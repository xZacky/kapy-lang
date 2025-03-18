//===- MatmulOpToLLVM.cpp ---------------------------------------*- C++ -*-===//
//
// This file implements class to make MatmulOp to LLVM compatible.
//
//===----------------------------------------------------------------------===//

#include "kapy/Conversion/KgpuToLLVM/ConvertUtils.h"
#include "kapy/Conversion/KgpuToLLVM/PTXBuilder.h"
#include "kapy/Conversion/KgpuToLLVM/Patterns.h"
#include "kapy/Dialect/Kapy/IR/Kapy.h"
#include "kapy/Dialect/Kgpu/IR/Kgpu.h"
#include "mlir/Conversion/LLVMCommon/Pattern.h"

using namespace mlir;
using namespace mlir::kapy;
using namespace mlir::LLVM;

static void permute(SmallVectorImpl<int64_t> &elements,
                    ArrayRef<unsigned> permutation) {
  SmallVector<int64_t> permuted;
  for (unsigned i = 0; i < elements.size(); ++i)
    permuted.push_back(elements[permutation[i]]);
  elements = std::move(permuted);
}

static void bitcast(OpBuilder rewriter, Location loc,
                    SmallVectorImpl<Value> &words) {
  SmallVector<Value> casted;
  for (auto word : words) {
    if (auto vectorType = dyn_cast<VectorType>(word.getType())) {
      if (vectorType.getElementType().isBF16()) {
        // bf16x2 -> i32
        assert(vectorType.getNumElements() == 2);
        auto i32Type = rewriter.getIntegerType(32);
        casted.push_back(llvm_bitcast(i32Type, word));
        continue;
      }
      if (vectorType.getElementType().isInteger(8)) {
        // f8x4 -> i32
        assert(vectorType.getNumElements() == 4);
        auto i32Type = rewriter.getIntegerType(32);
        casted.push_back(llvm_bitcast(i32Type, word));
        continue;
      }
    }
    if (word.getType().isF32()) {
      // tf32 -> i32
      auto i32Type = rewriter.getIntegerType(32);
      casted.push_back(llvm_bitcast(i32Type, word));
      continue;
    }
  }
  words = std::move(casted);
}

namespace {

class MatmulOpConversion : public ConvertOpToLLVMPattern<MatmulOp> {
public:
  using ConvertOpToLLVMPattern<MatmulOp>::ConvertOpToLLVMPattern;

  virtual LogicalResult
  matchAndRewrite(MatmulOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    assert(op->getNumOperands() == 3 && op->getNumResults() == 1);
    auto loc = op.getLoc();

    auto lhsType = op.getLhs().getType();
    auto rhsType = op.getRhs().getType();
    auto accType = op.getAcc().getType();
    auto lhsLayout = getLayout<FragmentsLayoutAttr>(lhsType);
    auto rhsLayout = getLayout<FragmentsLayoutAttr>(rhsType);
    auto accLayout = getLayout<FragmentsLayoutAttr>(accType);
    auto lhsLoopSpace = lhsLayout.getLoopSpace(lhsType.getShape());
    auto rhsLoopSpace = rhsLayout.getLoopSpace(rhsType.getShape());
    auto accLoopSpace = accLayout.getLoopSpace(accType.getShape());

    std::array<unsigned, 2> lhsSimdShape, rhsSimdShape, accSimdShape;
    switch (op.getMatmulImplWay()) {
    case MatmulImplWay::MMA_M16N8K8_F16:
    case MatmulImplWay::MMA_M16N8K8_TF32:
      lhsSimdShape = {2, 2};
      rhsSimdShape = {2, 1};
      accSimdShape = {2, 2};
      break;
    case MatmulImplWay::MMA_M16N8K16_F16:
    case MatmulImplWay::MMA_M16N8K16_F8:
      lhsSimdShape = {2, 4};
      rhsSimdShape = {4, 1};
      accSimdShape = {2, 2};
      break;
    }

    DenseMap<std::pair<int64_t, int64_t>, SmallVector<int64_t, 8>>
        lhsCoordsToLoopIvs;
    for (int64_t loopIv0 = 0; loopIv0 < lhsLoopSpace[0]; ++loopIv0) {
      for (int64_t loopIv1 = 0; loopIv1 < lhsLoopSpace[1]; ++loopIv1) {
        auto loopIv = loopIv0 * lhsLoopSpace[1] + loopIv1;
        auto coordM = loopIv0 / lhsSimdShape[0];
        auto coordK = loopIv1 / lhsSimdShape[1];
        lhsCoordsToLoopIvs[{coordM, coordK}].push_back(loopIv);
      }
    }
    DenseMap<std::pair<int64_t, int64_t>, SmallVector<int64_t, 4>>
        rhsCoordsToLoopIvs;
    for (int64_t loopIv0 = 0; loopIv0 < rhsLoopSpace[0]; ++loopIv0) {
      for (int64_t loopIv1 = 0; loopIv1 < rhsLoopSpace[1]; ++loopIv1) {
        auto loopIv = loopIv0 + loopIv1 * rhsLoopSpace[0];
        auto coordK = loopIv0 / rhsSimdShape[0];
        auto coordN = loopIv1 / rhsSimdShape[1];
        rhsCoordsToLoopIvs[{coordK, coordN}].push_back(loopIv);
      }
    }
    DenseMap<std::pair<int64_t, int64_t>, SmallVector<int64_t, 4>>
        accCoordsToLoopIvs;
    for (int64_t loopIv0 = 0; loopIv0 < accLoopSpace[0]; ++loopIv0) {
      for (int64_t loopIv1 = 0; loopIv1 < accLoopSpace[1]; ++loopIv1) {
        auto loopIv = loopIv0 * accLoopSpace[1] + loopIv1;
        auto coordM = loopIv0 / accSimdShape[0];
        auto coordN = loopIv1 / accSimdShape[1];
        accCoordsToLoopIvs[{coordM, coordN}].push_back(loopIv);
      }
    }

    auto sizeM = accType.getShape()[0];
    auto sizeN = accType.getShape()[1];
    auto sizeK = lhsType.getShape()[1];
    auto stepM = 16;
    auto stepN = 8;
    auto stepK = 8;
    switch (op.getMatmulImplWay()) {
    case MatmulImplWay::MMA_M16N8K8_F16:
    case MatmulImplWay::MMA_M16N8K8_TF32:
      break;
    case MatmulImplWay::MMA_M16N8K16_F16:
    case MatmulImplWay::MMA_M16N8K16_F8:
      stepK = 16;
      break;
    }

    switch (op.getMatmulImplWay()) {
    case MatmulImplWay::MMA_M16N8K8_F16:
      break;
    case MatmulImplWay::MMA_M16N8K8_TF32:
      // [0, 1] -> [0, 2]
      // [2, 3]    [1, 3]
      for (unsigned m = 0; m < sizeM / stepM; ++m)
        for (unsigned k = 0; k < sizeK / stepK; ++k)
          permute(lhsCoordsToLoopIvs[{m, k}], {0, 2, 1, 3});
      break;
    case MatmulImplWay::MMA_M16N8K16_F16:
      // [0, 1, 2, 3] -> [0, 1, 4, 5]
      // [4, 5, 6, 7]    [2, 3, 6, 7]
      for (unsigned m = 0; m < sizeM / stepM; ++m)
        for (unsigned k = 0; k < sizeK / stepK; ++k)
          permute(lhsCoordsToLoopIvs[{m, k}], {0, 1, 4, 5, 2, 3, 6, 7});
      break;
    case MatmulImplWay::MMA_M16N8K16_F8:
      break;
    }

    auto lhsStruct = adaptor.getLhs();
    auto rhsStruct = adaptor.getRhs();
    auto accStruct = adaptor.getAcc();
    auto lhsElemType = getLhsElementType(op);
    auto rhsElemType = getRhsElementType(op);
    auto accElemType = getAccElementType(op);
    for (unsigned m = 0; m < sizeM / stepM; ++m) {
      for (unsigned n = 0; n < sizeN / stepN; ++n) {
        for (unsigned k = 0; k < sizeK / stepK; ++k) {
          auto lhsLoopIvs = lhsCoordsToLoopIvs[{m, k}];
          auto rhsLoopIvs = rhsCoordsToLoopIvs[{k, n}];
          auto accLoopIvs = accCoordsToLoopIvs[{m, n}];
          SmallVector<Value> lhsElems, rhsElems, accElems;
          for (auto iv : lhsLoopIvs)
            lhsElems.push_back(llvm_extractvalue(lhsElemType, lhsStruct, iv));
          for (auto iv : rhsLoopIvs)
            rhsElems.push_back(llvm_extractvalue(rhsElemType, rhsStruct, iv));
          for (auto iv : accLoopIvs)
            accElems.push_back(llvm_extractvalue(accElemType, accStruct, iv));
          auto lhsWordTypes = getLhsWordTypes(op);
          auto rhsWordTypes = getRhsWordTypes(op);
          auto accWordTypes = getAccWordTypes(op);
          SmallVector<Value> lhsWords, rhsWords, accWords;
          for (unsigned i = 0; i < lhsWordTypes.size(); ++i) {
            if (auto vectorType = dyn_cast<VectorType>(lhsWordTypes[i])) {
              Value lhsWord = llvm_undef(vectorType);
              for (unsigned j = 0; j < vectorType.getNumElements(); ++j)
                lhsWord = llvm_insertelement(
                    vectorType, lhsWord,
                    lhsElems[i * vectorType.getNumElements() + j],
                    arith_constant_i32(j));
              lhsWords.push_back(lhsWord);
            } else {
              lhsWords.push_back(lhsElems[i]);
            }
          }
          for (unsigned i = 0; i < rhsWordTypes.size(); ++i) {
            if (auto vectorType = dyn_cast<VectorType>(rhsWordTypes[i])) {
              Value rhsWord = llvm_undef(vectorType);
              for (unsigned j = 0; j < vectorType.getNumElements(); ++j)
                rhsWord = llvm_insertelement(
                    vectorType, rhsWord,
                    rhsElems[i * vectorType.getNumElements() + j],
                    arith_constant_i32(j));
              rhsWords.push_back(rhsWord);
            } else {
              rhsWords.push_back(rhsElems[i]);
            }
          }
          for (unsigned i = 0; i < accWordTypes.size(); ++i) {
            if (auto vectorType = dyn_cast<VectorType>(accWordTypes[i])) {
              Value accWord = llvm_undef(vectorType);
              for (unsigned j = 0; j < vectorType.getNumElements(); ++j)
                accWord = llvm_insertelement(
                    vectorType, accWord,
                    accElems[i * vectorType.getNumElements() + j],
                    arith_constant_i32(j));
              accWords.push_back(accWord);
            } else {
              accWords.push_back(accElems[i]);
            }
          }
          bitcast(rewriter, loc, lhsWords);
          bitcast(rewriter, loc, rhsWords);

          PTXBuilder builder;
          auto &mma = *builder.create(getPTXString(op));
          auto *dstOperand = builder.newListOperand();
          auto *lhsOperand = builder.newListOperand();
          auto *rhsOperand = builder.newListOperand();
          auto *accOperand = builder.newListOperand();
          for (unsigned i = 0; i < accWords.size(); ++i)
            dstOperand->listPushBack(builder.newOperand("=r"));
          for (unsigned i = 0; i < lhsWords.size(); ++i)
            lhsOperand->listPushBack(builder.newOperand(lhsWords[i], "r"));
          for (unsigned i = 0; i < rhsWords.size(); ++i)
            rhsOperand->listPushBack(builder.newOperand(rhsWords[i], "r"));
          for (unsigned i = 0; i < accWords.size(); ++i)
            accOperand->listPushBack(builder.newOperand(accWords[i], "r"));
          mma(dstOperand, lhsOperand, rhsOperand, accOperand);
          auto mmaWordsType =
              LLVMStructType::getLiteral(getContext(), accWordTypes);
          auto mmaWords = builder.launch(rewriter, loc, mmaWordsType, false);
          SmallVector<Value> mmaElems;
          for (unsigned i = 0; i < accWordTypes.size(); ++i) {
            Value mmaWord = llvm_extractvalue(accWordTypes[i], mmaWords, i);
            if (auto vectorType = dyn_cast<VectorType>(mmaWord.getType())) {
              for (unsigned j = 0; j < vectorType.getNumElements(); ++j)
                mmaElems.push_back(llvm_extractelement(accElemType, mmaWord,
                                                       arith_constant_i32(j)));
            } else {
              mmaElems.push_back(mmaWord);
            }
          }
          auto structType = accStruct.getType();
          for (auto it : llvm::enumerate(accLoopIvs))
            accStruct = llvm_insertvalue(structType, accStruct,
                                         mmaElems[it.index()], it.value());
        }
      }
    }

    rewriter.replaceOp(op, accStruct);
    return success();
  }

private:
  std::string getPTXString(MatmulOp op) const {
    switch (op.getMatmulImplWay()) {
    case MatmulImplWay::MMA_M16N8K8_F16: {
      if (op.getLhs().getType().getElementType().isF16()) {
        if (op.getType().getElementType().isF16())
          return "mma.sync.aligned.m16n8k8.row.col.f16.f16.f16.f16";
        else
          return "mma.sync.aligned.m16n8k8.row.col.f32.f16.f16.f32";
      } else {
        return "mma.sync.aligned.m16n8k8.row.col.f32.bf16.bf16.f32";
      }
    }
    case MatmulImplWay::MMA_M16N8K8_TF32: {
      if (op.getType().getElementType().isF16())
        return "mma.sync.aligned.m16n8k8.row.col.f16.tf32.tf32.f16";
      else
        return "mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32";
    }
    case MatmulImplWay::MMA_M16N8K16_F16: {
      if (op.getLhs().getType().getElementType().isF16()) {
        if (op.getType().getElementType().isF16())
          return "mma.sync.aligned.m16n8k16.row.col.f16.f16.f16.f16";
        else
          return "mma.sync.aligned.m16n8k16.row.col.f16.f16.f16.f16";
      } else {
        return "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32";
      }
    }
    case MatmulImplWay::MMA_M16N8K16_F8: {
      std::string lhs = op.getLhs().getType().getElementType().isFloat8E4M3()
                            ? ".e4m3"
                            : ".e5m2";
      std::string rhs = op.getRhs().getType().getElementType().isFloat8E4M3()
                            ? ".e4m3"
                            : ".e5m2";
      if (op.getType().getElementType().isF16())
        return "mma.sync.aligned.m16n8k16.row.col.f16" + lhs + rhs + ".f16";
      else
        return "mma.sync.aligned.m16n8k16.row.col.f32" + lhs + rhs + ".f32";
    }
    }
    llvm_unreachable("unsupported matmul implement way");
  }

  Type getLhsElementType(MatmulOp op) const {
    auto elementType = op.getLhs().getType().getElementType();
    return typeConverter->convertType(elementType);
  }

  Type getRhsElementType(MatmulOp op) const {
    auto elementType = op.getRhs().getType().getElementType();
    return typeConverter->convertType(elementType);
  }

  Type getAccElementType(MatmulOp op) const {
    auto elementType = op.getAcc().getType().getElementType();
    return typeConverter->convertType(elementType);
  }

  SmallVector<Type, 4> getLhsWordTypes(MatmulOp op) const {
    auto *context = op.getContext();
    switch (op.getMatmulImplWay()) {
    case MatmulImplWay::MMA_M16N8K8_F16: {
      if (op.getLhs().getType().getElementType().isF16()) {
        auto f16Type = FloatType::getF16(context);
        return SmallVector<Type>(2, VectorType::get(2, f16Type));
      } else {
        auto bf16Type = FloatType::getBF16(context);
        return SmallVector<Type>(2, VectorType::get(2, bf16Type));
      }
    }
    case MatmulImplWay::MMA_M16N8K16_F16: {
      if (op.getLhs().getType().getElementType().isF16()) {
        auto f16Type = FloatType::getF16(context);
        return SmallVector<Type>(4, VectorType::get(2, f16Type));
      } else {
        auto bf16Type = FloatType::getBF16(context);
        return SmallVector<Type>(4, VectorType::get(2, bf16Type));
      }
    }
    case MatmulImplWay::MMA_M16N8K8_TF32: {
      auto f32Type = FloatType::getF32(context);
      return SmallVector<Type>(4, f32Type);
    }
    case MatmulImplWay::MMA_M16N8K16_F8: {
      auto i8Type = IntegerType::get(context, 8);
      return SmallVector<Type>(2, VectorType::get(4, i8Type));
    }
    }
    llvm_unreachable("unsupported matmul implement way");
  }

  SmallVector<Type, 2> getRhsWordTypes(MatmulOp op) const {
    auto *context = op.getContext();
    switch (op.getMatmulImplWay()) {
    case MatmulImplWay::MMA_M16N8K8_F16: {
      if (op.getRhs().getType().getElementType().isF16()) {
        auto f16Type = FloatType::getF16(context);
        return SmallVector<Type>(1, VectorType::get(2, f16Type));
      } else {
        auto bf16Type = FloatType::getBF16(context);
        return SmallVector<Type>(1, VectorType::get(2, bf16Type));
      }
    }
    case MatmulImplWay::MMA_M16N8K16_F16: {
      if (op.getRhs().getType().getElementType().isF16()) {
        auto f16Type = FloatType::getF16(context);
        return SmallVector<Type>(2, VectorType::get(2, f16Type));
      } else {
        auto bf16Type = FloatType::getBF16(context);
        return SmallVector<Type>(2, VectorType::get(2, bf16Type));
      }
    }
    case MatmulImplWay::MMA_M16N8K8_TF32: {
      auto f32Type = FloatType::getF32(context);
      return SmallVector<Type>(2, f32Type);
    }
    case MatmulImplWay::MMA_M16N8K16_F8: {
      auto i8Type = IntegerType::get(context, 8);
      return SmallVector<Type>(1, VectorType::get(4, i8Type));
    }
    }
    llvm_unreachable("unsupported matmul implement way");
  }

  SmallVector<Type, 4> getAccWordTypes(MatmulOp op) const {
    auto *context = op.getContext();
    if (op.getType().getElementType().isF16()) {
      auto f16Type = FloatType::getF16(context);
      return SmallVector<Type>(2, VectorType::get(2, f16Type));
    } else {
      auto f32Type = FloatType::getF32(context);
      return SmallVector<Type>(4, f32Type);
    }
  }
};

} // namespace

void kapy::populateMatmulOpToLLVMConversionPattern(
    const LLVMTypeConverter &typeConverter, RewritePatternSet &patterns) {
  patterns.add<MatmulOpConversion>(typeConverter);
}
