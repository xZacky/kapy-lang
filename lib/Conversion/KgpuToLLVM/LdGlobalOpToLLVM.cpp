//===- LdGlobalOpToLLVM.cpp -------------------------------------*- C++ -*-===//
//
// This file implements class to make LdGlobalOp to LLVM compatible.
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

class LdGlobalOpConversion : public ConvertOpToLLVMPattern<LdGlobalOp> {
public:
  using ConvertOpToLLVMPattern<LdGlobalOp>::ConvertOpToLLVMPattern;

  virtual LogicalResult
  matchAndRewrite(LdGlobalOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    assert(op->getNumOperands() == 1 && op->getNumResults() == 1);
    auto loc = op.getLoc();

    auto sourceType = op.getSource().getType();
    auto globalLayout = getLayout<Strided2dLayoutAttr>(sourceType);

    auto tensorLayout = getLayout<FragmentsLayoutAttr>(op.getType());
    auto laneLoops = tensorLayout.getLaneLoops();
    auto loopSpace = tensorLayout.getLoopSpace(sourceType.getShape());
    auto map = tensorLayout.getAffineMap(sourceType.getShape(), 1);

    auto bitWidth = getIntOrFloatBitWidth(sourceType);
    auto simdSize = laneLoops[globalLayout.isColMajor() ? 0 : 1];
    auto numWords = simdSize * bitWidth / 32;

    DenseMap<int64_t, SmallVector<int64_t, 8>> instIdToLoopIvs;
    for (int64_t loopIv0 = 0; loopIv0 < loopSpace[0]; ++loopIv0) {
      for (int64_t loopIv1 = 0; loopIv1 < loopSpace[1]; ++loopIv1) {
        int64_t instId;
        if (globalLayout.isRowMajor())
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

    auto pointerType = LLVMPointerType::get(getContext(), 1);
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
    Value one = arith_constant_i32(1);
    Value stride0 = globalLayout.isRowMajor()
                        ? llvm_extractvalue(i32Type, sourceStruct, 5)
                        : one;
    Value stride1 = globalLayout.isColMajor()
                        ? llvm_extractvalue(i32Type, sourceStruct, 6)
                        : one;
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
      pointers.push_back(
          llvm_getelementptr(pointerType, elementType, pointer,
                             arith_addi(arith_muli(index0, stride0),
                                        arith_muli(index1, stride1))));
      predicates.push_back(arith_andi(arith_cmpi(lessThan, index0, end0),
                                      arith_cmpi(lessThan, index1, end1)));
    }

    SmallVector<Value> resultValues(product(loopSpace));
    for (int64_t instId = 0; instId < numInsts; ++instId) {
      PTXBuilder builder;
      auto &ld = builder.create("ld")->o("volatile", op.isVolatile()).global();
      if (!op.isVolatile()) {
        if (op.getCacheModifier() != CacheModifier::NONE)
          ld.o(stringifyCacheModifier(op.getCacheModifier()).str());
        else
          ld.o("L1::" + stringifyEvictPriority(op.getEvictPriority()).str());
      }
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
  Type getSourceElementType(LdGlobalOp op) const {
    auto elementType = op.getSource().getType().getElementType();
    return typeConverter->convertType(elementType);
  }

  LLVMStructType getResultStructType(LdGlobalOp op) const {
    auto resultType = typeConverter->convertType(op.getType());
    return cast<LLVMStructType>(resultType);
  }
};

} // namespace

void kapy::populateLdGlobalOpToLLVMConversionPattern(
    const LLVMTypeConverter &typeConverter, RewritePatternSet &patterns) {
  patterns.add<LdGlobalOpConversion>(typeConverter);
}
