//===- LdSharedOpToLLVM.cpp -------------------------------------*- C++ -*-===//
//
// This file implements class to make LdSharedOp to LLVM compatible.
//
//===----------------------------------------------------------------------===//

#include "kapy/Conversion/KgpuToLLVM/ConvertUtils.h"
#include "kapy/Conversion/KgpuToLLVM/PTXBuilder.h"
#include "kapy/Conversion/KgpuToLLVM/Patterns.h"
#include "kapy/Dialect/Kapy/IR/Kapy.h"
#include "kapy/Dialect/Kgpu/IR/Kgpu.h"
#include "kapy/Support/CommonUtils.h"
#include "mlir/Conversion/LLVMCommon/Pattern.h"

using namespace mlir;
using namespace mlir::kapy;
using namespace mlir::LLVM;

namespace {

class LdSharedOpConversion : public ConvertOpToLLVMPattern<LdSharedOp> {
public:
  using ConvertOpToLLVMPattern<LdSharedOp>::ConvertOpToLLVMPattern;

  virtual LogicalResult
  matchAndRewrite(LdSharedOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    assert(op->getNumOperands() == 1 && op->getNumResults() == 1);
    auto loc = op.getLoc();

    auto sourceType = op.getSource().getType();
    auto sharedLayout = getLayout<SwizzlingLayoutAttr>(sourceType);
    auto bankParam = sharedLayout.getBankParam();
    auto lineParam = sharedLayout.getLineParam();

    auto tensorLayout = getLayout<FragmentsLayoutAttr>(op.getType());
    auto laneLoops = tensorLayout.getLaneLoops();
    auto loopSpace = tensorLayout.getLoopSpace(sourceType.getShape());
    auto map = tensorLayout.getAffineMap(sourceType.getShape(), 1);

    auto bitWidth = getIntOrFloatBitWidth(sourceType);
    auto simdSize = laneLoops[sharedLayout.isColMajor() ? 0 : 1];
    simdSize = std::min(simdSize, bankParam * 32 / bitWidth);
    auto numWords = simdSize * bitWidth / 32;

    DenseMap<int64_t, SmallVector<int64_t, 8>> instIdToLoopIvs;
    for (int64_t loopIv0 = 0; loopIv0 < loopSpace[0]; ++loopIv0) {
      for (int64_t loopIv1 = 0; loopIv1 < loopSpace[1]; ++loopIv1) {
        int64_t instId;
        if (sharedLayout.isRowMajor())
          instId = (loopIv0 * loopSpace[1] + loopIv1) / simdSize;
        else
          instId = (loopIv0 + loopIv1 * loopSpace[0]) / simdSize;
        int64_t loopIv;
        if (tensorLayout.isRowMajor())
          loopIv = loopIv0 * loopSpace[1] + loopIv1;
        else
          loopIv = loopIv0 + loopIv1 * loopSpace[0];
        instIdToLoopIvs[instId].push_back(loopIv);
      }
    }
    auto numInsts = instIdToLoopIvs.size();

    auto pointerType = LLVMPointerType::get(getContext(), 3);
    auto elementType = getSourceElementType(op);
    auto i32Type = rewriter.getIntegerType(32);
    auto lessThan = arith::CmpIPredicate::ult;
    auto initConst = generateInitConstant(sourceType.getElementType(),
                                          op.getPaddingOption());

    Value sourceStruct = adaptor.getSource();
    Value pointer = llvm_extractvalue(pointerType, sourceStruct, 0);
    Value start0 = llvm_extractvalue(i32Type, sourceStruct, 1);
    Value start1 = llvm_extractvalue(i32Type, sourceStruct, 2);
    Value end0 = llvm_extractvalue(i32Type, sourceStruct, 3);
    Value end1 = llvm_extractvalue(i32Type, sourceStruct, 4);
    Value stride0 = arith_constant_i32(sharedLayout.getStride0());
    Value stride1 = arith_constant_i32(sharedLayout.getStride1());
    Value bankValue = arith_constant_i32(bankParam);
    Value lineValue = arith_constant_i32(lineParam);
    Value byteWidth = arith_constant_i32(bitWidth / 8);
    Value const128 = arith_constant_i32(128);
    Value four = arith_constant_i32(4);
    Value bankSize = arith_constant_i32(32 / bitWidth);
    Value lineSize = arith_constant_i32(1024 / bitWidth);
    Value laneId = rewriter.create<LaneIdOp>(loc);

    SmallVector<Value> pointers, predicates;
    for (int64_t instId = 0; instId < numInsts; ++instId) {
      Value loopIv = arith_constant_i32(instIdToLoopIvs[instId][0]);
      Value index0 =
          expandAffineExpr(rewriter, loc, map.getResult(0), {laneId, loopIv});
      Value index1 =
          expandAffineExpr(rewriter, loc, map.getResult(1), {laneId, loopIv});
      index0 = arith_addi(start0, index0);
      index1 = arith_addi(start1, index1);
      // before swizzling
      Value elemOffset = arith_addi(arith_muli(index0, stride0), //
                                    arith_muli(index1, stride1));
      Value byteOffset = arith_muli(elemOffset, byteWidth);
      Value bankId = arith_divui(arith_remui(byteOffset, const128), four);
      Value lineId = arith_divui(byteOffset, const128);
      // apply swizzling
      Value xorResult = arith_xori(arith_divui(bankId, bankValue),
                                   arith_remui(lineId, lineValue));
      Value newBankId = arith_addi(arith_muli(xorResult, bankValue),
                                   arith_remui(bankId, bankValue));
      Value bankOffset = arith_muli(newBankId, bankSize);
      Value lineOffset = arith_muli(lineId, lineSize);
      pointers.push_back(
          llvm_getelementptr(pointerType, elementType, pointer,
                             arith_addi(bankOffset, lineOffset)));
      predicates.push_back(arith_andi(arith_cmpi(lessThan, index0, end0),
                                      arith_cmpi(lessThan, index1, end1)));
    }

    SmallVector<Value> resultValues(product(loopSpace));
    for (int64_t instId = 0; instId < numInsts; ++instId) {
      PTXBuilder builder;
      auto &ld = builder.create("ld")->o("volatile", op.isVolatile()).shared();
      ld.v(numWords, numWords > 1).b(32);

      auto *dstOperand = builder.newListOperand();
      for (unsigned i = 0; i < numWords; ++i)
        dstOperand->listPushBack(builder.newOperand("=r", initConst));
      auto *srcOperand = builder.newAddressOperand(pointers[instId], "l");
      ld(dstOperand, srcOperand).predicate(predicates[instId], "b");

      SmallVector<Type> wordTypes(numWords, i32Type);
      Type wordsType;
      if (numWords > 1)
        wordsType = LLVMStructType::getLiteral(getContext(), wordTypes);
      else
        wordsType = wordTypes[0];
      auto words = builder.launch(rewriter, loc, wordsType);
      for (unsigned i = 0; i < numWords; ++i) {
        Value word;
        if (numWords > 1)
          word = llvm_extractvalue(i32Type, words, i);
        else
          word = words;
        if (bitWidth < 32) {
          auto vectorSize = 32 / bitWidth;
          auto vectorType = VectorType::get(vectorSize, elementType);
          word = llvm_bitcast(vectorType, word);
          for (unsigned j = 0; j < vectorSize; ++j) {
            auto loopIv = instIdToLoopIvs[instId][i * vectorSize + j];
            resultValues[loopIv] =
                llvm_extractelement(elementType, word, arith_constant_i32(j));
          }
        } else {
          auto loopIv = instIdToLoopIvs[instId][i];
          resultValues[loopIv] = llvm_bitcast(elementType, word);
        }
      }
    }

    auto resultType = getResultStructType(op);
    packAndReplace(rewriter, op, resultType, resultValues);
    return success();
  }

private:
  Type getSourceElementType(LdSharedOp op) const {
    auto elementType = op.getSource().getType().getElementType();
    return typeConverter->convertType(elementType);
  }

  LLVMStructType getResultStructType(LdSharedOp op) const {
    auto resultType = typeConverter->convertType(op.getType());
    return cast<LLVMStructType>(resultType);
  }
};

} // namespace

void kapy::populateLdSharedOpToLLVMConversionPattern(
    const LLVMTypeConverter &typeConverter, RewritePatternSet &patterns) {
  patterns.add<LdSharedOpConversion>(typeConverter);
}
