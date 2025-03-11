//===- ElementwiseOpToLLVM.cpp ----------------------------------*- C++ -*-===//
//
// Copyright 2018-2020 Philippe Tillet
// Copyright 2020-2022 OpenAI
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.
//
//===----------------------------------------------------------------------===//
//
// This file is copied and modified from the triton project.
// https://github.com/triton-lang/triton
//
//===----------------------------------------------------------------------===//

#include "kapy/Conversion/KgpuToLLVM/ConvertUtils.h"
#include "kapy/Conversion/KgpuToLLVM/PTXBuilder.h"
#include "kapy/Conversion/KgpuToLLVM/Patterns.h"
#include "kapy/Dialect/Kapy/IR/Kapy.h"
#include "mlir/Conversion/LLVMCommon/Pattern.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"

using namespace mlir;
using namespace mlir::kapy;

template <typename OpT>
static Value createOp(OpT op, ConversionPatternRewriter &rewriter,
                      Type resultType, ValueRange operands) {
  return rewriter.create<OpT>(op->getLoc(), resultType, operands,
                              op->getAttrs());
}

static LLVM::LLVMFuncOp
getOrCreateExternFuncOp(OpBuilder &builder, Operation *op, StringRef funcName,
                        Type funcType, StringRef libName, StringRef libPath) {
  auto symbolAttr = StringAttr::get(op->getContext(), funcName);
  auto *symbolOp = SymbolTable::lookupNearestSymbolFrom(op, symbolAttr);
  if (auto funcOp = dyn_cast<LLVM::LLVMFuncOp>(symbolOp))
    return funcOp;

  auto *ip = op;
  if (!isa<LLVM::LLVMFuncOp>(ip))
    ip = ip->getParentOfType<LLVM::LLVMFuncOp>();
  builder.setInsertionPoint(ip);
  auto loc = op->getLoc();
  auto funcOp = builder.create<LLVM::LLVMFuncOp>(loc, funcName, funcType);
  funcOp->setAttr("libname", StringAttr::get(op->getContext(), libName));
  funcOp->setAttr("libpath", StringAttr::get(op->getContext(), libPath));
  return funcOp;
}

template <typename OpT> static bool isBF16Operands(OpT op) {
  static_assert(std::is_same_v<OpT, arith::MulFOp> ||
                std::is_same_v<OpT, arith::AddFOp> ||
                std::is_same_v<OpT, arith::SubFOp>);
  auto lhsType = getElementTypeOrSelf(op.getLhs().getType());
  auto rhsType = getElementTypeOrSelf(op.getRhs().getType());
  return lhsType.isBF16() && rhsType.isBF16();
}

struct ConversionDesc {
  std::string ptx;
  unsigned oldPackedBits;
  unsigned newPackedBits;
  unsigned numElements;
};

static const ConversionDesc F16_TO_F8E5M2_RN(bool isNative) {
  ConversionDesc desc;
  if (!isNative)
    desc = {"{                            \n"
            ".reg .b32 a<2>;              \n"
            "and.b32 a0, $1, 0xfffefffe;  \n"   // a0 &= 0xfffefffe
            "and.b32 a1, $2, 0xfffefffe;  \n"   // (strip lowest bit)
            "add.u32 a0, a0, 0x00800080;  \n"   // a0 += 0x00800080
            "add.u32 a1, a1, 0x00800080;  \n"   // (round to nearest)
            "prmt.b32 $0, a0, a1, 0x7531; \n\t" // output = a1a0
            "}",
            32, 32, 4};
  else
    desc = {"cvt.rn.satfinite.e5m2x2.f16x2 $0, $1; \n\t", 32, 16, 2};
  return desc;
}

static const ConversionDesc F16_TO_F8E5M2_RZ = {
    "{                            \n"
    ".reg .b32 a<2>;              \n"
    "and.b32 a0, $1, 0xfffefffe;  \n"   // a0 &= 0xfffefffe
    "and.b32 a1, $2, 0xfffefffe;  \n"   // (strip lowest bit)
    "prmt.b32 $0, a0, a1, 0x7531; \n\t" // output = a1a0
    "}",
    32, 32, 4};

static const ConversionDesc F8E5M2_TO_F16(bool isNative) {
  ConversionDesc desc;
  if (!isNative)
    desc = {"{                           \n"
            "prmt.b32 $0, 0, $2, 0x5140; \n\t"
            "prmt.b32 $1, 0, $2, 0x7362; \n\t"
            "}",
            32, 32, 4};
  else
    desc = {"cvt.rn.f16x2.e5m2x2 $0, $1; \n\t", 16, 32, 2};
  return desc;
}

static const ConversionDesc F8E5M2_TO_BF16(bool isNative) {
  ConversionDesc desc;
  if (!isNative)
    desc = {
        "{                                        \n"
        ".reg .b32 a<2>, b<2>, c<4>, d<4>, e112;  \n" // if input = 0xf1f2f3f4
        "mov.u32 e112, 0x77800000;                \n"
        "prmt.b32 a0, 0, $2, 0x5140;              \n" // a0 = 0xf300f400
        "prmt.b32 a1, 0, $2, 0x7362;              \n" // a1 = 0xf100f200
        "lop3.b32 b0, a0, 0x7fff7fff, 0, 0xc0;    \n" // b0 = a0 & 0x7fff7fff
        "lop3.b32 b1, a1, 0x7fff7fff, 0, 0xc0;    \n" // (strip sign)
        "shr.b32  b0, b0, 3;                      \n" // b0 >>= 3
        "shr.b32  b1, b1, 3;                      \n" // shift into bf16
                                                      // position
        "and.b32 c0, b0, 0xFFFF0000;              \n" // c0 = f3
        "shl.b32 c1, b0, 16;                      \n" // c1 = f4
        "and.b32 c2, b1, 0xFFFF0000;              \n" // c2 = f1
        "shl.b32 c3, b1, 16;                      \n" // c3 = f2
        "mul.f32 d0, c0, e112;                    \n" // d0 = c0 * 0x77800000
        "mul.f32 d1, c1, e112;                    \n" // d1 = c1 * 0x77800000
        "mul.f32 d2, c2, e112;                    \n" // d2 = c2 * 0x77800000
        "mul.f32 d3, c3, e112;                    \n" // d3 = c3 * 0x77800000
        "prmt.b32 b0, d0, d1, 0x3276;             \n" // b0 = 0xd3d4
        "prmt.b32 b1, d2, d3, 0x3276;             \n" // b1 = 0xd1d2
        "lop3.b32 $0, b0, 0x80008000, a0, 0xf8;   \n" // out0 =
                                                      // b0|(0x80008000&a0)
        "lop3.b32 $1, b1, 0x80008000, a1, 0xf8;   \n" // (restore sign)
        "}",
        32, 32, 4};
  else
    desc = {
        "{                                      \n"
        ".reg .b32 a<2>, b<2>;                  \n" // if input = 0xf1f2f3f4
        ".reg .b32 e112;                        \n"
        "mov.u32 e112, 0x77807780;              \n" // 2**112 represented as
                                                    // bf16x2
        "prmt.b32 a0, 0, $2, 0x5140;            \n" // a0 = 0xf300f400
        "prmt.b32 a1, 0, $2, 0x7362;            \n" // a1 = 0xf100f200
        "lop3.b32 b0, a0, 0x7fff7fff, 0, 0xc0;  \n" // b0 = a0 & 0x7fff7fff
        "lop3.b32 b1, a1, 0x7fff7fff, 0, 0xc0;  \n" // (strip sign)
        "shr.b32  b0, b0, 3;                    \n" // b0 >>= 3
        "shr.b32  b1, b1, 3;                    \n" // shift into bf16 position
        "lop3.b32 b0, b0, 0x80008000, a0, 0xf8; \n" // out0 = b0|(0x80008000&a0)
        "lop3.b32 b1, b1, 0x80008000, a1, 0xf8; \n" // (restore sign)
        "mul.rn.bf16x2 $0, b0, e112;            \n" // b0.exp += 2**7-2**4
        "mul.rn.bf16x2 $1, b1, e112;            \n" // exponent compensate = 112
        "}",
        32, 32, 4};
  return desc;
}

static const ConversionDesc BF16_TO_F8E5M2_RN(bool isNative) {
  ConversionDesc desc;
  if (!isNative)
    desc = {
        "{                                           \n" // bf16=fp8>>3 + 112<<7
        ".reg .u32 sign, sign<2>, nosign, nosign<2>; \n" // fp8_min = 0b00000000
        ".reg .u32 fp8_min, fp8_max, rn_;            \n" // fp8_max = 0b11111111
        "mov.u32 fp8_min, 0x38003800;                \n" // so bf16_min = 0x3800
        "mov.u32 fp8_max, 0x57e057e0;                \n" // so bf16_max = 0x57e0
        "mov.u32 rn_, 0x00100010;                    \n" // round to nearest
        "and.b32 sign0, $1, 0x80008000;              \n" // sign0=in0&0x80008000
        "and.b32 sign1, $2, 0x80008000;              \n" // (store sign)
        "prmt.b32 sign, sign0, sign1, 0x7531;        \n"
        "and.b32 nosign0, $1, 0x7fff7fff;            \n" // nosign0=in0&0x7fff7fff
        "and.b32 nosign1, $2, 0x7fff7fff;            \n" // (strip sign)

        // nosign = clamp(nosign, min, max)
        ".reg .u32 nosign_0_<2>, nosign_1_<2>;       \n"
        "and.b32 nosign_0_0, nosign0, 0xffff0000;    \n"
        "max.u32 nosign_0_0, nosign_0_0, 0x38000000; \n"
        "min.u32 nosign_0_0, nosign_0_0, 0x57e00000; \n"
        "and.b32 nosign_0_1, nosign0, 0x0000ffff;    \n"
        "max.u32 nosign_0_1, nosign_0_1, 0x3800;     \n"
        "min.u32 nosign_0_1, nosign_0_1, 0x57e0;     \n"
        "or.b32 nosign0, nosign_0_0, nosign_0_1;     \n"
        "and.b32 nosign_1_0, nosign1, 0xffff0000;    \n"
        "max.u32 nosign_1_0, nosign_1_0, 0x38000000; \n"
        "min.u32 nosign_1_0, nosign_1_0, 0x57e00000; \n"
        "and.b32 nosign_1_1, nosign1, 0x0000ffff;    \n"
        "max.u32 nosign_1_1, nosign_1_1, 0x3800;     \n"
        "min.u32 nosign_1_1, nosign_1_1, 0x57e0;     \n"
        "or.b32 nosign1, nosign_1_0, nosign_1_1;     \n"

        "add.u32 nosign0, nosign0, rn_;              \n" // nosign0 += rn_
        "add.u32 nosign1, nosign1, rn_;              \n" // (round to nearest)
        "sub.u32 nosign0, nosign0, 0x38003800;       \n" // nosign0-=0x38003800
        "sub.u32 nosign1, nosign1, 0x38003800;       \n" // (compensate offset)
        "shl.b32 nosign0, nosign0, 3;                \n" // nosign0 <<= 3
        "shl.b32 nosign1, nosign1, 3;                \n" // shift into to fp8e4
        "prmt.b32 nosign, nosign0, nosign1, 0x7531;  \n" // nosign0 = 0xf100f200
                                                         // nosign1 = 0xf300f400
                                                         // nosign = 0xf3f4f1f2
        "or.b32 $0, nosign, sign;                    \n" // restore sign
        "}",
        32, 32, 4};
  else
    desc = {"{                                       \n"
            ".reg .b16 a<2>;                         \n"
            ".reg .f32 b<2>;                         \n"
            "mov.b32 {a0, a1}, $1;                   \n"
            "cvt.f32.bf16 b0, a0;                    \n"
            "cvt.f32.bf16 b1, a1;                    \n"
            "cvt.rn.satfinite.e5m2x2.f32 $0, b1, b0; \n"
            "}",
            32, 16, 2};
  return desc;
}

/// F8E4M3 (x2) -> F16 (x2) (packed)
static const ConversionDesc F8E4M3_TO_F16 = {"{ \n"
                                             "cvt.rn.f16x2.e4m3x2 $0, $1; \n"
                                             "}",
                                             16, 32, 2};

/// F16 (x2) -> F8E4M3 (x2) (packed)
static const ConversionDesc F16_TO_F8E4M3_RN = {
    "{ \n"
    "cvt.rn.satfinite.e4m3x2.f16x2 $0, $1; \n"
    "}",
    32, 16, 2};

/// F8E4M3 (x2) -> BF16 (x2) (packed)
static const ConversionDesc F8E4M3_TO_BF16 = {
    "{                                       \n"
    ".reg .b32 a;                            \n"
    ".reg .f16 a<2>;                         \n"
    ".reg .b16 b<2>;                         \n"
    "cvt.rn.f16x2.e4m3x2 a, $1;              \n"
    "mov.b32 {a0, a1}, a;                    \n"
    "cvt.bf16.f16 b0, a0;                    \n"
    "cvt.bf16.f16 b1, a1;                    \n"
    "mov.b32 $0, {b0, b1};                   \n"
    "}",
    16, 32, 2};

/// BF16 (x2) -> F8E4M3 (x2) (packed)
static const ConversionDesc BF16_TO_F8E4M3_RN = {
    "{                                       \n"
    ".reg .b16 a<2>;                         \n"
    ".reg .f32 b<2>;                         \n"
    "mov.b32 {a0, a1}, $1;                   \n"
    "cvt.f32.bf16 b0, a0;                    \n"
    "cvt.f32.bf16 b1, a1;                    \n"
    "cvt.rn.satfinite.e4m3x2.f32 $0, b1, b0; \n"
    "}",
    32, 16, 2};

/// F32 (x2) -> F8E4M3 (x2) (packed)
static const ConversionDesc F32_TO_F8E4M3_RN = {
    "cvt.rn.satfinite.e4m3x2.f32 $0, $2, $1; \n", 32, 16, 2};

/// F32 (x2) -> F8E5M2 (x2) (packed)
static const ConversionDesc F32_TO_F8E5M2_RN = {
    "cvt.rn.satfinite.e5m2x2.f32 $0, $2, $1; \n", 32, 16, 2};

static const std::string S8_TO_BF16 =
    "{                                           \n"
    ".reg .s8 s<4>;                              \n"
    ".reg .f32 f<4>;                             \n"
    "mov.b32 {s0, s1, s2, s3}, $2;               \n" // unpack
    "cvt.rn.f32.s8 f0, s0;                       \n" // no s8->bf16 pre-Hopper
    "cvt.rn.f32.s8 f1, s1;                       \n" // fi[0:15] is always 0
    "cvt.rn.f32.s8 f2, s2;                       \n" //
    "cvt.rn.f32.s8 f3, s3;                       \n" //
    "prmt.b32 $0, f0, f1, 0x7632;                \n" // f32->bf16 + pack
    "prmt.b32 $1, f2, f3, 0x7632;                \n" //
    "}";

using ConversionFunc = std::function<SmallVector<Value>(
    ConversionPatternRewriter &, Location, ValueRange)>;

static ConversionFunc makeConversionFuncFromPTX(const std::string &ptx,
                                                Type oldType, Type newType,
                                                unsigned oldPackedBits = 32,
                                                unsigned newPackedBits = 32) {
  ConversionFunc convFunc = [&](ConversionPatternRewriter &rewriter,
                                Location loc, ValueRange oldElems) {
    auto *context = rewriter.getContext();
    auto numElems = oldElems.size();
    assert(numElems == 4 || numElems == 2);

    auto oldBitWidth = oldType.getIntOrFloatBitWidth();
    auto newBitWidth = newType.getIntOrFloatBitWidth();

    // First, we pack `oldElems` into 32-bit integers.
    auto oldVecWidth = oldPackedBits / oldBitWidth;
    auto oldVectorType = VectorType::get(oldVecWidth, oldType);
    auto oldNumVectors = numElems / oldVecWidth;
    SmallVector<Value> oldVectors(oldNumVectors, llvm_undef(oldVectorType));
    for (unsigned i = 0; i < numElems; ++i) {
      auto &oldVector = oldVectors[i / oldVecWidth];
      oldVector = llvm_insertelement(oldVectorType, oldVector, oldElems[i],
                                     arith_constant_i32(i % oldVecWidth));
    }
    auto oldPackType = rewriter.getIntegerType(oldPackedBits);
    for (unsigned i = 0; i < oldNumVectors; ++i)
      oldVectors[i] = llvm_bitcast(oldPackType, oldVectors[i]);

    // Then, we run the provided inline PTX.
    PTXBuilder builder;
    SmallVector<PTXBuilder::Operand *> ptxOperands;
    auto dstConstraint = newPackedBits == 16 ? "=h" : "=r";
    auto srcConstraint = oldPackedBits == 16 ? "h" : "r";
    auto newVecWidth = newPackedBits / newBitWidth;
    auto newNumVectors = numElems / newVecWidth;
    for (unsigned i = 0; i < newNumVectors; ++i)
      ptxOperands.push_back(builder.newOperand(dstConstraint));
    for (auto oldVector : oldVectors)
      ptxOperands.push_back(builder.newOperand(oldVector, srcConstraint));
    auto &inst = *builder.create(ptx);
    inst(ptxOperands, true);

    auto newVectorType = VectorType::get(newVecWidth, newType);
    SmallVector<Value> newVectors;
    if (newNumVectors == 1) {
      newVectors.push_back(builder.launch(rewriter, loc, newVectorType, false));
    } else {
      auto newStructType = LLVM::LLVMStructType::getLiteral(
          context, SmallVector<Type>(newNumVectors, newVectorType));
      auto newStruct = builder.launch(rewriter, loc, newStructType, false);
      for (unsigned i = 0; i < newNumVectors; ++i)
        newVectors.push_back(llvm_extractvalue(newVectorType, newStruct, i));
    }

    SmallVector<Value> newElems;
    for (unsigned i = 0; i < numElems; ++i) {
      auto newVector = newVectors[i / newVecWidth];
      newElems.push_back(llvm_extractelement(
          newType, newVector, arith_constant_i32(i % newVecWidth)));
    }
    return newElems;
  };

  return convFunc;
}

static Value cvtF16ToF32(RewriterBase &rewriter, Location loc, Value value) {
  PTXBuilder builder;
  auto &cvt = *builder.create("cvt.f32.f16");
  auto *dstOperand = builder.newOperand("=r");
  auto *srcOperand = builder.newOperand(value, "h");
  cvt(dstOperand, srcOperand);
  return builder.launch(rewriter, loc, rewriter.getF32Type(), false);
}

static Value cvtBF16ToF32(RewriterBase &rewriter, Location loc, Value value) {
  PTXBuilder builder;
  auto &cvt = *builder.create("cvt.f32.bf16");
  auto *dstOperand = builder.newOperand("=r");
  auto *srcOperand = builder.newOperand(value, "h");
  cvt(dstOperand, srcOperand);
  return builder.launch(rewriter, loc, rewriter.getF32Type(), false);
}

static Value cvtF32ToF16(RewriterBase &rewriter, Location loc, Value value,
                         RoundingMode roundingMode) {
  PTXBuilder builder;
  StringRef ptx;
  switch (roundingMode) {
  case RoundingMode::RZ:
    ptx = "cvt.rz.f16.f32";
    break;
  case RoundingMode::RN:
    ptx = "cvt.rn.f16.f32";
    break;
  }
  auto &cvt = *builder.create(ptx.str());
  auto *dstOperand = builder.newOperand("=h");
  auto *srcOperand = builder.newOperand(value, "r");
  cvt(dstOperand, srcOperand);
  return builder.launch(rewriter, loc, rewriter.getF16Type(), false);
}

static Value cvtF32ToBF16(RewriterBase &rewriter, Location loc, Value value,
                          RoundingMode roundingMode) {
  PTXBuilder builder;
  StringRef ptx;
  switch (roundingMode) {
  case RoundingMode::RZ:
    ptx = "cvt.rz.bf16.f32";
    break;
  case RoundingMode::RN:
    ptx = "cvt.rn.bf16.f32";
    break;
  }
  auto &cvt = *builder.create(ptx.str());
  auto *dstOperand = builder.newOperand("=h");
  auto *srcOperand = builder.newOperand(value, "r");
  cvt(dstOperand, srcOperand);
  return builder.launch(rewriter, loc, rewriter.getBF16Type(), false);
}

namespace {

/// Base pattern for elementwise op conversion. Unpack individual elements from
/// a llvm.struct via llvm.extractvalue, call ConcreteT::doConversion for each
/// element, and pack them back into a llvm.struct using llvm.insertvalue.
template <typename OpT, typename ConcreteT>
class ElementwiseOpConversion : public ConvertOpToLLVMPattern<OpT> {
public:
  using ConvertOpToLLVMPattern<OpT>::ConvertOpToLLVMPattern;
  using OpAdaptor = typename OpT::Adaptor;

  virtual LogicalResult
  matchAndRewrite(OpT op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    assert(op->getNumOperands() >= 1 && op->getNumResults() == 1);
    if (isa<RankedTensorType>(op->getResultTypes()[0]))
      return rewriteTensor(op, adaptor, rewriter);
    else
      return rewriteScalar(op, adaptor, rewriter);
  }

protected:
  LLVM::LLVMStructType getResultStructType(Operation *op) const {
    auto resultType =
        this->getTypeConverter()->convertType(op->getResultTypes()[0]);
    return cast<LLVM::LLVMStructType>(resultType);
  }

  Type getResultElementType(Operation *op) const {
    auto resultType = getElementTypeOrSelf(op->getResultTypes()[0]);
    return this->getTypeConverter()->convertType(resultType);
  }

private:
  LogicalResult rewriteTensor(OpT op, OpAdaptor adaptor,
                              ConversionPatternRewriter &rewriter) const {
    auto loc = op->getLoc();
    SmallVector<SmallVector<Value>> operandsValues;
    for (auto llvmStruct : adaptor.getOperands())
      operandsValues.push_back(unpackLLVMStruct(rewriter, loc, llvmStruct));

    auto resultValues = static_cast<const ConcreteT *>(this)->doConversion(
        op, adaptor, rewriter, MultipleValuesRange(operandsValues));
    if (resultValues.empty())
      return failure();

    auto resultType = getResultStructType(op);
    auto resultStruct =
        packToLLVMStruct(rewriter, loc, resultType, resultValues);
    rewriter.replaceOp(op, resultStruct);
    return success();
  }

  LogicalResult rewriteScalar(OpT op, OpAdaptor adaptor,
                              ConversionPatternRewriter &rewriter) const {
    SmallVector<SmallVector<Value>> operandsValues;
    for (auto operand : adaptor.getOperands())
      operandsValues.push_back(SmallVector<Value>{operand});

    auto resultValues = static_cast<const ConcreteT *>(this)->doConversion(
        op, adaptor, rewriter, MultipleValuesRange(operandsValues));
    if (resultValues.empty())
      return failure();

    assert(resultValues.size() == 1);
    rewriter.replaceOp(op, resultValues[0]);
    return success();
  }
};

template <typename OpT>
class UnaryOpConversion
    : public ElementwiseOpConversion<OpT, UnaryOpConversion<OpT>> {
public:
  using Base = ElementwiseOpConversion<OpT, UnaryOpConversion<OpT>>;
  using Base::Base;
  using OpAdaptor = typename Base::OpAdaptor;

  SmallVector<Value> doConversion(OpT op, OpAdaptor adaptor,
                                  ConversionPatternRewriter &rewriter,
                                  MultipleValuesRange operandsValues) const {
    assert(operandsValues.size() == 1);
    auto resultType = getResultElementType(op);
    SmallVector<Value> resultValues;
    for (unsigned i = 0; i < operandsValues[0].size(); ++i) {
      auto value = operandsValues[0][i];
      value = createOp(op, rewriter, resultType, value);
      resultValues.push_back(value);
    }
    return resultValues;
  }
};

template <typename OpT>
class BinaryOpConversion
    : public ElementwiseOpConversion<OpT, BinaryOpConversion<OpT>> {
public:
  using Base = ElementwiseOpConversion<OpT, UnaryOpConversion<OpT>>;
  using Base::Base;
  using OpAdaptor = typename Base::OpAdaptor;

  SmallVector<Value> doConversion(OpT op, OpAdaptor adaptor,
                                  ConversionPatternRewriter &rewriter,
                                  MultipleValuesRange operandsValues) const {
    assert(operandsValues.size() == 2);
    auto resultType = getResultElementType(op);
    SmallVector<Value> resultValues;
    for (unsigned i = 0; i < operandsValues[0].size(); ++i) {
      auto lhs = operandsValues[0][i];
      auto rhs = operandsValues[1][i];
      auto result = createOp(op, rewriter, resultType, {lhs, rhs});
      resultValues.push_back(result);
    }
    return resultValues;
  }
};

class FPToFPOpConversion
    : public ElementwiseOpConversion<FPToFPOp, FPToFPOpConversion> {
public:
  using Base = ElementwiseOpConversion<FPToFPOp, FPToFPOpConversion>;
  using Base::Base;

  SmallVector<Value> doConversion(FPToFPOp op, OpAdaptor adaptor,
                                  ConversionPatternRewriter &rewriter,
                                  MultipleValuesRange operandsValues) const {
    assert(operandsValues.size() == 1);
    auto loc = op.getLoc();
    auto resultType = getResultElementType(op);

    auto oldType = getElementTypeOrSelf(op.getSource().getType());
    auto newType = getElementTypeOrSelf(op.getResult().getType());
    auto nvidiaCC = getNvidiaCC(op->getParentOfType<ModuleOp>());
    auto roundingMode = op.getRoundingMode();

    if (oldType.isF32() && newType.isF16()) {
      assert(roundingMode.has_value());
      SmallVector<Value> resultValues;
      for (unsigned i = 0; i < operandsValues[0].size(); ++i) {
        auto value = operandsValues[0][i];
        value = cvtF32ToF16(rewriter, loc, value, roundingMode.value());
        resultValues.push_back(value);
      }
      return resultValues;
    }

    if (oldType.isF32() && newType.isBF16()) {
      assert(roundingMode.has_value());
      SmallVector<Value> resultValues;
      for (unsigned i = 0; i < operandsValues[0].size(); ++i) {
        auto value = operandsValues[0][i];
        value = cvtF32ToBF16(rewriter, loc, value, roundingMode.value());
        resultValues.push_back(value);
      }
      return resultValues;
    }

    bool isToF8 = newType.isFloat8E4M3() || newType.isFloat8E5M2();
    bool isF32ToF8 = oldType.isF32() && isToF8;
    bool isRZ = roundingMode.value() == RoundingMode::RZ;
    bool isF16Intermediate = isF32ToF8 && (nvidiaCC < 90 || isRZ);

    oldType = isF16Intermediate ? rewriter.getF16Type() : oldType;
    newType = newType.isF32() ? rewriter.getF16Type() : newType;

    auto [convFunc, numElems] =
        getConversionFunc(oldType, newType, nvidiaCC, roundingMode);

    SmallVector<Value> sourceValues;
    for (unsigned i = 0; i < operandsValues[0].size(); ++i)
      sourceValues.push_back(operandsValues[0][i]);
    if (isF16Intermediate)
      for (auto &value : sourceValues)
        value = cvtF32ToF16(rewriter, loc, value, RoundingMode::RZ);

    SmallVector<Value> resultValues;
    for (unsigned i = 0; i < sourceValues.size(); i += numElems) {
      SmallVector<Value> elements;
      for (unsigned j = 0; j < numElems; ++j)
        elements.push_back(sourceValues[i + j]);
      elements = convFunc(rewriter, loc, elements);
      resultValues.append(elements.begin(), elements.end());
    }
    if (newType.isF32())
      for (auto &value : resultValues)
        value = cvtF16ToF32(rewriter, loc, value);
    return resultValues;
  }

private:
  std::pair<ConversionFunc, unsigned>
  getConversionFunc(Type oldType, Type newType, int64_t nvidiaCC,
                    std::optional<RoundingMode> roundingMode) const {
    auto F8E4M3TypeID = TypeID::get<Float8E4M3Type>();
    auto F8E5M2TypeID = TypeID::get<Float8E5M2Type>();
    auto F16TypeID = TypeID::get<Float16Type>();
    auto BF16TypeID = TypeID::get<BFloat16Type>();
    auto F32TypeID = TypeID::get<Float64Type>();

    auto roundingModeNull = static_cast<RoundingMode>(-1);

    static DenseMap<std::tuple<TypeID, TypeID, RoundingMode>, ConversionDesc>
        convDescs = {
            // F8E4M3 to F16
            {{F8E4M3TypeID, F16TypeID, roundingModeNull}, F8E4M3_TO_F16},
            // F8E5M2 to F16
            {{F8E5M2TypeID, F16TypeID, roundingModeNull},
             F8E5M2_TO_F16(nvidiaCC >= 90)},
            // F16 to F8E4M3
            {{F16TypeID, F8E4M3TypeID, RoundingMode::RN}, F16_TO_F8E4M3_RN},
            // F16 to F8E5M2
            {{F16TypeID, F8E5M2TypeID, RoundingMode::RN},
             F16_TO_F8E5M2_RN(nvidiaCC >= 90)},
            {{F16TypeID, F8E5M2TypeID, RoundingMode::RZ}, F16_TO_F8E5M2_RZ},
            // F8E4M3 to BF16
            {{F8E4M3TypeID, BF16TypeID, roundingModeNull}, F8E4M3_TO_BF16},
            // F8E5M2 to BF16
            {{F8E5M2TypeID, BF16TypeID, roundingModeNull},
             F8E5M2_TO_BF16(nvidiaCC >= 90)},
            // BF16 to F8E4M3
            {{BF16TypeID, F8E4M3TypeID, RoundingMode::RN}, BF16_TO_F8E4M3_RN},
            // BF16 to F8E5M2
            {{BF16TypeID, F8E5M2TypeID, RoundingMode::RN},
             BF16_TO_F8E5M2_RN(nvidiaCC >= 90)},
            // F32 to F8E4M3
            {{F32TypeID, F8E4M3TypeID, RoundingMode::RN}, F32_TO_F8E4M3_RN},
            // F32 to F8E5M2
            {{F32TypeID, F8E5M2TypeID, RoundingMode::RN}, F32_TO_F8E5M2_RN}};

    std::tuple<TypeID, TypeID, RoundingMode> convKind = {
        oldType.getTypeID(), newType.getTypeID(),
        roundingMode.value_or(roundingModeNull)};

    oldType = typeConverter->convertType(oldType);
    newType = typeConverter->convertType(newType);
    auto convDesc = convDescs.lookup(convKind);
    auto convFunc = makeConversionFuncFromPTX(convDesc.ptx, oldType, newType,
                                              convDesc.oldPackedBits,
                                              convDesc.newPackedBits);
    return {convFunc, convDesc.numElements};
  }
};

class AbsFOpConversion
    : public ElementwiseOpConversion<math::AbsFOp, AbsFOpConversion> {
public:
  using Base = ElementwiseOpConversion<math::AbsFOp, AbsFOpConversion>;
  using Base::Base;

  SmallVector<Value> doConversion(math::AbsFOp op, OpAdaptor adaptor,
                                  ConversionPatternRewriter &rewriter,
                                  MultipleValuesRange operandsValues) const {
    assert(operandsValues.size() == 1);
    auto loc = op.getLoc();
    auto resultType = getResultElementType(op);
    SmallVector<Value> resultValues;
    if (resultType.isInteger()) {
      // Mask out the sign bit.
      auto bitWidth = getIntOrFloatBitWidth(op.getType());
      auto mask = (1 << (bitWidth - 1)) - 1;
      auto maskAttr = rewriter.getIntegerAttr(resultType, mask);
      auto maskOp = arith_constant(loc, maskAttr);
      for (unsigned i = 0; i < operandsValues[0].size(); ++i) {
        auto value = operandsValues[0][i];
        value = arith_andi(loc, value, maskOp);
        resultValues.push_back(value);
      }
    } else {
      for (unsigned i = 0; i < operandsValues[0].size(); ++i) {
        auto value = operandsValues[0][i];
        value = createOp(op, rewriter, resultType, value);
        resultValues.push_back(value);
      }
    }
    return resultValues;
  }
};

class ClampFOpConversion
    : public ElementwiseOpConversion<ClampFOp, ClampFOpConversion> {
public:
  using Base = ElementwiseOpConversion<ClampFOp, ClampFOpConversion>;
  using Base::Base;

  SmallVector<Value> doConversion(ClampFOp op, OpAdaptor adaptor,
                                  ConversionPatternRewriter &rewriter,
                                  MultipleValuesRange operandsValues) const {
    assert(operandsValues.size() == 3);
    auto loc = op.getLoc();
    auto resultType = getResultElementType(op);
    SmallVector<Value> resultValues;
    for (unsigned i = 0; i < operandsValues[0].size(); ++i) {
      auto value = operandsValues[0][i];
      auto low = operandsValues[1][i];
      auto high = operandsValues[2][i];
      if (op.getPropagateNan()) {
        value = arith_maximumf(resultType, value, low);
        value = arith_minimumf(resultType, value, high);
      } else {
        value = arith_maxnumf(resultType, value, low);
        value = arith_minnumf(resultType, value, high);
      }
      resultValues.push_back(value);
    }
    return resultValues;
  }
};

class DivFOpConversion
    : public ElementwiseOpConversion<arith::DivFOp, DivFOpConversion> {
public:
  using Base = ElementwiseOpConversion<arith::DivFOp, DivFOpConversion>;
  using Base::Base;

  SmallVector<Value> doConversion(arith::DivFOp op, OpAdaptor adaptor,
                                  ConversionPatternRewriter &rewriter,
                                  MultipleValuesRange operandsValues) const {
    assert(operandsValues.size() == 2);
    auto loc = op.getLoc();
    auto resultType = getResultElementType(op);
    SmallVector<Value> resultValues;
    for (unsigned i = 0; i < operandsValues[0].size(); ++i) {
      auto lhs = operandsValues[0][i];
      auto rhs = operandsValues[1][i];
      PTXBuilder builder;
      auto &div = *builder.create("div");
      auto bitWidth = resultType.getIntOrFloatBitWidth();
      switch (bitWidth) {
      case 32:
        div.o("full").o("f32");
        break;
      case 64:
        div.o("rn").o("f64");
        break;
      default:
        llvm_unreachable("unsupported bit width");
      }
      auto *dstOperand = builder.newOperand(bitWidth == 32 ? "=r" : "=l");
      auto *lhsOperand = builder.newOperand(lhs, bitWidth == 32 ? "r" : "l");
      auto *rhsOperand = builder.newOperand(rhs, bitWidth == 32 ? "r" : "l");
      div(dstOperand, lhsOperand, rhsOperand);
      auto result = builder.launch(rewriter, loc, resultType, false);
      resultValues.push_back(result);
    }
    return resultValues;
  }
};

class MulFOpConversion
    : public ElementwiseOpConversion<arith::MulFOp, MulFOpConversion> {
public:
  using Base = ElementwiseOpConversion<arith::MulFOp, MulFOpConversion>;
  using Base::Base;

  SmallVector<Value> doConversion(arith::MulFOp op, OpAdaptor adaptor,
                                  ConversionPatternRewriter &rewriter,
                                  MultipleValuesRange operandsValues) const {
    assert(operandsValues.size() == 2);
    auto loc = op.getLoc();
    auto resultType = getResultElementType(op);
    SmallVector<Value> resultValues;
    if (isBF16Operands(op)) {
      for (unsigned i = 0; i < operandsValues[0].size(); ++i) {
        auto lhs = operandsValues[0][i];
        auto rhs = operandsValues[1][i];
        PTXBuilder builder;
        auto &mul = *builder.create(" { .reg .b16 c;                 \n"
                                    "   mov.b16 c, 0x8000U;          \n"
                                    "   fma.rn.bf16 $0, $1, $2, c; } \n");
        auto *dstOperand = builder.newOperand("=h");
        auto *lhsOperand = builder.newOperand(lhs, "h");
        auto *rhsOperand = builder.newOperand(rhs, "h");
        mul(dstOperand, lhsOperand, rhsOperand);
        auto result = builder.launch(rewriter, loc, resultType, false);
        resultValues.push_back(result);
      }
    } else {
      for (unsigned i = 0; i < operandsValues[0].size(); ++i) {
        auto lhs = operandsValues[0][i];
        auto rhs = operandsValues[1][i];
        auto result = createOp(op, rewriter, resultType, {lhs, rhs});
        resultValues.push_back(result);
      }
    }
    return resultValues;
  }
};

class AddFOpConversion
    : public ElementwiseOpConversion<arith::AddFOp, AddFOpConversion> {
public:
  using Base = ElementwiseOpConversion<arith::AddFOp, AddFOpConversion>;
  using Base::Base;

  SmallVector<Value> doConversion(arith::AddFOp op, OpAdaptor adaptor,
                                  ConversionPatternRewriter &rewriter,
                                  MultipleValuesRange operandsValues) const {
    assert(operandsValues.size() == 2);
    auto loc = op.getLoc();
    auto resultType = getResultElementType(op);
    SmallVector<Value> resultValues;
    if (isBF16Operands(op)) {
      for (unsigned i = 0; i < operandsValues[0].size(); ++i) {
        auto lhs = operandsValues[0][i];
        auto rhs = operandsValues[1][i];
        PTXBuilder builder;
        auto &add = *builder.create("{ .reg .b16 c;                 \n"
                                    "  mov.b16 c, 0x3f80U;          \n"
                                    "  fma.rn.bf16 $0, $1, c, $2; } \n");
        auto *dstOperand = builder.newOperand("=h");
        auto *lhsOperand = builder.newOperand(lhs, "h");
        auto *rhsOperand = builder.newOperand(rhs, "h");
        add(dstOperand, lhsOperand, rhsOperand);
        auto result = builder.launch(rewriter, loc, resultType, false);
        resultValues.push_back(result);
      }
    } else {
      for (unsigned i = 0; i < operandsValues[0].size(); ++i) {
        auto lhs = operandsValues[0][i];
        auto rhs = operandsValues[1][i];
        auto result = createOp(op, rewriter, resultType, {lhs, rhs});
        resultValues.push_back(result);
      }
    }
    return resultValues;
  }
};

class SubFOpConversion
    : public ElementwiseOpConversion<arith::SubFOp, SubFOpConversion> {
public:
  using Base = ElementwiseOpConversion<arith::SubFOp, SubFOpConversion>;
  using Base::Base;

  SmallVector<Value> doConversion(arith::SubFOp op, OpAdaptor adaptor,
                                  ConversionPatternRewriter &rewriter,
                                  MultipleValuesRange operandsValues) const {
    assert(operandsValues.size() == 2);
    auto loc = op.getLoc();
    auto resultType = getResultElementType(op);
    SmallVector<Value> resultValues;
    if (isBF16Operands(op)) {
      for (unsigned i = 0; i < operandsValues[0].size(); ++i) {
        auto lhs = operandsValues[0][i];
        auto rhs = operandsValues[1][i];
        PTXBuilder builder;
        auto &sub = *builder.create(" { .reg.b16 c;                  \n"
                                    "   mov.b16 c, 0xbf80U;          \n"
                                    "   fma.rn.bf16 $0, $2, c, $1; } \n");
        auto *dstOperand = builder.newOperand("=h");
        auto *lhsOperand = builder.newOperand(lhs, "h");
        auto *rhsOperand = builder.newOperand(rhs, "h");
        sub(dstOperand, lhsOperand, rhsOperand);
        auto result = builder.launch(rewriter, loc, resultType, false);
        resultValues.push_back(result);
      }
    } else {
      for (unsigned i = 0; i < operandsValues[0].size(); ++i) {
        auto lhs = operandsValues[0][i];
        auto rhs = operandsValues[1][i];
        auto result = createOp(op, rewriter, resultType, {lhs, rhs});
        resultValues.push_back(result);
      }
    }
    return resultValues;
  }
};

class SIToFPOpConversion
    : public ElementwiseOpConversion<arith::SIToFPOp, SIToFPOpConversion> {
public:
  using Base = ElementwiseOpConversion<arith::SIToFPOp, SIToFPOpConversion>;
  using Base::Base;

  SmallVector<Value> doConversion(arith::SIToFPOp op, OpAdaptor adaptor,
                                  ConversionPatternRewriter &rewriter,
                                  MultipleValuesRange operandsValues) const {
    assert(operandsValues.size() == 1);
    auto loc = op.getLoc();
    auto resultType = getResultElementType(op);
    auto oldType = getElementTypeOrSelf(op.getIn().getType());
    auto newType = getElementTypeOrSelf(op.getType());
    SmallVector<Value> resultValues;
    if (oldType.isInteger(8) && newType.isBF16() &&
        operandsValues[0].size() >= 4) {
      auto convFunc = makeConversionFuncFromPTX(S8_TO_BF16, oldType, newType);
      for (unsigned i = 0; i < operandsValues[0].size(); i += 4) {
        SmallVector<Value> elements{
            operandsValues[0][i + 0], operandsValues[0][i + 1],
            operandsValues[0][i + 2], operandsValues[0][i + 3]};
        elements = convFunc(rewriter, loc, elements);
        resultValues.append(elements.begin(), elements.end());
      }
    } else if (newType.isBF16()) {
      auto f32Type = rewriter.getF32Type();
      for (unsigned i = 0; i < operandsValues[0].size(); ++i) {
        auto value = operandsValues[0][i];
        value = createOp(op, rewriter, f32Type, value);
        value = cvtF32ToBF16(rewriter, loc, value, RoundingMode::RN);
        resultValues.push_back(value);
      }
    } else {
      for (unsigned i = 0; i < operandsValues[0].size(); ++i) {
        auto value = operandsValues[0][i];
        value = createOp(op, rewriter, resultType, value);
        resultValues.push_back(value);
      }
    }
    return resultValues;
  }
};

class FPToSIOpConversion
    : public ElementwiseOpConversion<arith::FPToSIOp, FPToSIOpConversion> {
public:
  using Base = ElementwiseOpConversion<arith::FPToSIOp, FPToSIOpConversion>;
  using Base::Base;

  SmallVector<Value> doConversion(arith::FPToSIOp op, OpAdaptor adaptor,
                                  ConversionPatternRewriter &rewriter,
                                  MultipleValuesRange operandsValues) const {
    assert(operandsValues.size() == 1);
    auto loc = op.getLoc();
    auto resultType = getResultElementType(op);
    auto oldType = getElementTypeOrSelf(op.getIn().getType());
    auto newType = getElementTypeOrSelf(op.getType());
    SmallVector<Value> resultValues;
    if (oldType.isBF16()) {
      for (unsigned i = 0; i < operandsValues[0].size(); ++i) {
        auto value = operandsValues[0][i];
        value = cvtBF16ToF32(rewriter, loc, value);
        value = createOp(op, rewriter, resultType, value);
        resultValues.push_back(value);
      }
    } else {
      for (unsigned i = 0; i < operandsValues[0].size(); ++i) {
        auto value = operandsValues[0][i];
        value = createOp(op, rewriter, resultType, value);
        resultValues.push_back(value);
      }
    }
    return resultValues;
  }
};

class ExtFOpConversion
    : public ElementwiseOpConversion<arith::ExtFOp, ExtFOpConversion> {
public:
  using Base = ElementwiseOpConversion<arith::ExtFOp, ExtFOpConversion>;
  using Base::Base;

  SmallVector<Value> doConversion(arith::ExtFOp op, OpAdaptor adaptor,
                                  ConversionPatternRewriter &rewriter,
                                  MultipleValuesRange operandsValues) const {
    assert(operandsValues.size() == 1);
    auto loc = op.getLoc();
    auto resultType = getResultElementType(op);
    auto oldType = getElementTypeOrSelf(op.getIn().getType());
    auto newType = getElementTypeOrSelf(op.getType());
    SmallVector<Value> resultValues;
    if (oldType.isBF16()) {
      for (unsigned i = 0; i < operandsValues[0].size(); ++i) {
        auto value = operandsValues[0][i];
        value = cvtBF16ToF32(rewriter, loc, value);
        resultValues.push_back(value);
      }
    } else {
      for (unsigned i = 0; i < operandsValues[0].size(); ++i) {
        auto value = operandsValues[0][i];
        value = createOp(op, rewriter, resultType, value);
        resultValues.push_back(value);
      }
    }
    return resultValues;
  }
};

class TruncFOpConversion
    : public ElementwiseOpConversion<arith::TruncFOp, TruncFOpConversion> {
public:
  using Base = ElementwiseOpConversion<arith::TruncFOp, TruncFOpConversion>;
  using Base::Base;

  SmallVector<Value> doConversion(arith::TruncFOp op, OpAdaptor adaptor,
                                  ConversionPatternRewriter &rewriter,
                                  MultipleValuesRange operandsValues) const {
    assert(operandsValues.size() == 1);
    auto loc = op.getLoc();
    auto resultType = getResultElementType(op);
    auto oldType = getElementTypeOrSelf(op.getIn().getType());
    auto newType = getElementTypeOrSelf(op.getType());
    SmallVector<Value> resultValues;
    if (newType.isBF16()) {
      for (unsigned i = 0; i < operandsValues[0].size(); ++i) {
        auto value = operandsValues[0][i];
        value = cvtF32ToBF16(rewriter, loc, value, RoundingMode::RN);
        resultValues.push_back(value);
      }
    } else {
      for (unsigned i = 0; i < operandsValues[0].size(); ++i) {
        auto value = operandsValues[0][i];
        value = createOp(op, rewriter, resultType, value);
        resultValues.push_back(value);
      }
    }
    return resultValues;
  }
};

class ExpApproxOpConversion
    : public ElementwiseOpConversion<math::ExpOp, ExpApproxOpConversion> {
public:
  using Base = ElementwiseOpConversion<math::ExpOp, ExpApproxOpConversion>;
  using Base::Base;

  SmallVector<Value> doConversion(math::ExpOp op, OpAdaptor adaptor,
                                  ConversionPatternRewriter &rewriter,
                                  MultipleValuesRange operandsValues) const {
    assert(operandsValues.size() == 1);
    auto loc = op.getLoc();
    auto resultType = getResultElementType(op);

    // For non-F32 input, call __nv_expf for higher-precision calculation.
    if (!resultType.isF32())
      return {};

    const double log2e = 1.4426950408889634;
    SmallVector<Value> resultValues;
    for (unsigned i = 0; i < operandsValues[0].size(); ++i) {
      auto value = operandsValues[0][i];
      value = arith_mulf(value, arith_constant_f32(log2e));
      PTXBuilder builder;
      auto &ex2 = *builder.create("ex2");
      ex2.o("approx").o("f32");
      auto *dstOperand = builder.newOperand("=f");
      auto *srcOperand = builder.newOperand(value, "f");
      ex2(dstOperand, srcOperand);
      value = builder.launch(rewriter, loc, resultType, false);
      resultValues.push_back(value);
    }
    return resultValues;
  }
};

class FmaOpConversion
    : public ElementwiseOpConversion<math::FmaOp, FmaOpConversion> {
public:
  using Base = ElementwiseOpConversion<math::FmaOp, FmaOpConversion>;
  using Base::Base;

  SmallVector<Value> doConversion(math::FmaOp op, OpAdaptor adaptor,
                                  ConversionPatternRewriter &rewriter,
                                  MultipleValuesRange operandsValues) const {
    assert(operandsValues.size() == 3);
    auto loc = op.getLoc();
    auto resultType = getResultElementType(op);
    SmallVector<Value> resultValues;
    for (unsigned i = 0; i < operandsValues[0].size(); ++i) {
      auto lhs = operandsValues[0][i];
      auto rhs = operandsValues[1][i];
      auto acc = operandsValues[2][i];
      auto result = createOp(op, rewriter, resultType, {lhs, rhs, acc});
      resultValues.push_back(result);
    }
    return resultValues;
  }
};

class ElementwiseExternLibOpConversion
    : public ElementwiseOpConversion<ElementwiseExternLibOp,
                                     ElementwiseExternLibOpConversion> {
public:
  using Base = ElementwiseOpConversion<ElementwiseExternLibOp,
                                       ElementwiseExternLibOpConversion>;
  using Base::Base;

  SmallVector<Value> doConversion(ElementwiseExternLibOp op, OpAdaptor adaptor,
                                  ConversionPatternRewriter &rewriter,
                                  MultipleValuesRange operandsValues) const {
    assert(operandsValues.size() >= 1);
    auto loc = op.getLoc();
    auto resultType = getResultElementType(op);

    SmallVector<Type> inputTypes;
    for (unsigned i = 0; i < operandsValues.size(); ++i)
      inputTypes.push_back(operandsValues[i][0].getType());

    auto funcName = op.getSymName();
    auto funcType = LLVM::LLVMFunctionType::get(resultType, inputTypes);
    auto funcOp = getOrCreateExternFuncOp(rewriter, op, funcName, funcType,
                                          op.getLibName(), op.getLibPath());

    SmallVector<Value> resultValues;
    for (unsigned i = 0; i < operandsValues[0].size(); ++i) {
      SmallVector<Value> inputs;
      for (unsigned j = 0; j < operandsValues.size(); ++j)
        inputs.push_back(operandsValues[j][i]);
      auto callOp = rewriter.create<LLVM::CallOp>(loc, funcOp, inputs);
      resultValues.push_back(callOp.getResult());
    }
    return resultValues;
  }
};

class ElementwiseInlineAsmOpConversion
    : public ElementwiseOpConversion<ElementwiseInlineAsmOp,
                                     ElementwiseInlineAsmOpConversion> {
public:
  using Base = ElementwiseOpConversion<ElementwiseInlineAsmOp,
                                       ElementwiseInlineAsmOpConversion>;
  using Base::Base;

  SmallVector<Value> doConversion(ElementwiseInlineAsmOp op, OpAdaptor adaptor,
                                  ConversionPatternRewriter &rewriter,
                                  MultipleValuesRange operandsValues) const {
    assert(operandsValues.size() >= 1);
    auto loc = op.getLoc();
    auto resultType = getResultElementType(op);
    SmallVector<Value> resultValues;
    for (unsigned i = 0; i < operandsValues[0].size(); ++i) {
      SmallVector<Value> inputs;
      for (unsigned j = 0; j < operandsValues.size(); ++j)
        inputs.push_back(operandsValues[j][i]);
      auto inlineAsmOp = rewriter.create<LLVM::InlineAsmOp>(
          loc, resultType, inputs, op.getAsmString(), op.getConstraints(),
          !op.isPure(), false);
      resultValues.push_back(inlineAsmOp.getResult(0));
    }
    return resultValues;
  }
};

} // namespace

void kapy::populateElementwiseOpToLLVMConversionPatterns(
    const LLVMTypeConverter &typeConverter, RewritePatternSet &patterns) {
  // ExpApproxOpConversion will try using ex2.approx if the input type is F32.
  // For other input types, ExpApproxOpConversion will return failure and
  // UnaryOpConversion<math::ExpOp> defined below will call __nv_expf for
  // higher-precision calculation.
  patterns.add<ExpApproxOpConversion>(typeConverter);

  patterns.add<BinaryOpConversion<arith::AddIOp>>(typeConverter);
  patterns.add<AddFOpConversion>(typeConverter);
  patterns.add<BinaryOpConversion<arith::SubIOp>>(typeConverter);
  patterns.add<SubFOpConversion>(typeConverter);
  patterns.add<BinaryOpConversion<arith::MulFOp>>(typeConverter);
  patterns.add<MulFOpConversion>(typeConverter);
  patterns.add<BinaryOpConversion<arith::DivUIOp>,
               BinaryOpConversion<arith::DivSIOp>>(typeConverter);
  patterns.add<DivFOpConversion>(typeConverter);
  patterns.add<BinaryOpConversion<arith::FloorDivSIOp>>(typeConverter);
  patterns.add<BinaryOpConversion<arith::CeilDivUIOp>,
               BinaryOpConversion<arith::CeilDivSIOp>>(typeConverter);
  patterns.add<BinaryOpConversion<arith::RemUIOp>,
               BinaryOpConversion<arith::RemSIOp>>(typeConverter);
  patterns.add<BinaryOpConversion<arith::RemFOp>>(typeConverter);
  patterns.add<BinaryOpConversion<arith::AndIOp>, //
               BinaryOpConversion<arith::OrIOp>,  //
               BinaryOpConversion<arith::XOrIOp>>(typeConverter);
  patterns.add<BinaryOpConversion<arith::ShLIOp>>(typeConverter);
  patterns.add<BinaryOpConversion<arith::ShRUIOp>,
               BinaryOpConversion<arith::ShRSIOp>>(typeConverter);
  patterns.add<BinaryOpConversion<arith::MaxUIOp>,
               BinaryOpConversion<arith::MaxSIOp>>(typeConverter);
  patterns.add<BinaryOpConversion<arith::MaximumFOp>,
               BinaryOpConversion<arith::MaxNumFOp>>(typeConverter);
  patterns.add<BinaryOpConversion<arith::MinUIOp>,
               BinaryOpConversion<arith::MinSIOp>>(typeConverter);
  patterns.add<BinaryOpConversion<arith::MinimumFOp>,
               BinaryOpConversion<arith::MinNumFOp>>(typeConverter);
  patterns.add<BinaryOpConversion<arith::CmpIOp>,
               BinaryOpConversion<arith::CmpFOp>>(typeConverter);
  patterns.add<UnaryOpConversion<arith::TruncIOp>>(typeConverter);
  patterns.add<TruncFOpConversion>(typeConverter);
  patterns.add<UnaryOpConversion<arith::ExtUIOp>,
               UnaryOpConversion<arith::ExtSIOp>>(typeConverter);
  patterns.add<ExtFOpConversion>(typeConverter);
  patterns.add<SIToFPOpConversion>(typeConverter);
  patterns.add<FPToSIOpConversion>(typeConverter);
  patterns.add<UnaryOpConversion<arith::BitcastOp>>(typeConverter);

  patterns.add<UnaryOpConversion<math::ExpOp>, //
               UnaryOpConversion<math::Exp2Op>>(typeConverter);
  patterns.add<UnaryOpConversion<math::FloorOp>, //
               UnaryOpConversion<math::CeilOp>>(typeConverter);
  patterns.add<UnaryOpConversion<math::SinOp>, //
               UnaryOpConversion<math::CosOp>>(typeConverter);
  patterns.add<UnaryOpConversion<math::LogOp>, //
               UnaryOpConversion<math::Log2Op>>(typeConverter);
  patterns.add<UnaryOpConversion<math::ErfOp>>(typeConverter);
  patterns.add<UnaryOpConversion<math::AbsIOp>>(typeConverter);
  patterns.add<AbsFOpConversion>(typeConverter);
  patterns.add<UnaryOpConversion<math::SqrtOp>>(typeConverter);
  patterns.add<UnaryOpConversion<math::RsqrtOp>>(typeConverter);
  patterns.add<FmaOpConversion>(typeConverter);

  patterns.add<FPToFPOpConversion>(typeConverter);
  patterns.add<ClampFOpConversion>(typeConverter);
  patterns.add<ElementwiseExternLibOpConversion>(typeConverter);
  patterns.add<ElementwiseInlineAsmOpConversion>(typeConverter);
}
