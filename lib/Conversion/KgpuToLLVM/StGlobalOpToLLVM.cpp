//===- StGlobalOpToLLVM.cpp -------------------------------------*- C++ -*-===//
//
// This file implements class to make StGlobalOp to LLVM compatible.
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

namespace {

class StGlobalOpConversion : public ConvertOpToLLVMPattern<StGlobalOp> {
public:
  using ConvertOpToLLVMPattern<StGlobalOp>::ConvertOpToLLVMPattern;

  virtual LogicalResult
  matchAndRewrite(StGlobalOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    assert(op->getNumOperands() == 2 && op->getNumResults() == 0);
    auto loc = op.getLoc();

    auto targetType = op.getTarget().getType();
    auto globalLayout = getLayout<Strided2dLayoutAttr>(targetType);

    auto sourceType = op.getSource().getType();
    auto tensorLayout = getLayout<FragmentsLayoutAttr>(sourceType);
    auto laneLoops = tensorLayout.getLaneLoops();
    auto loopSpace = tensorLayout.getLoopSpace(sourceType.getShape());
    auto map = tensorLayout.getAffineMap(sourceType.getShape(), 2);

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
    auto voidType = LLVMVoidType::get(getContext());
    auto lessThan = arith::CmpIPredicate::ult;

    Value sourceStruct = adaptor.getSource();
    Value targetStruct = adaptor.getTarget();
    Value pointer = llvm_extractvalue(pointerType, targetStruct, 0);
    Value start0 = llvm_extractvalue(i32Type, targetStruct, 1);
    Value start1 = llvm_extractvalue(i32Type, targetStruct, 2);
    Value end0 = llvm_extractvalue(i32Type, targetStruct, 3);
    Value end1 = llvm_extractvalue(i32Type, targetStruct, 4);
    Value one = arith_constant_i32(1);
    Value stride0 = globalLayout.isRowMajor()
                        ? llvm_extractvalue(i32Type, targetStruct, 5)
                        : one;
    Value stride1 = globalLayout.isColMajor()
                        ? llvm_extractvalue(i32Type, targetStruct, 6)
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

    for (int64_t instId = 0; instId < numInsts; ++instId) {
      PTXBuilder builder;
      auto &st = builder.create("st")->o("volatile", op.isVolatile()).global();
      if (!op.isVolatile()) {
        if (op.getCacheModifier() != CacheModifier::NONE)
          st.o(stringifyCacheModifier(op.getCacheModifier()).str());
        else
          st.o("L1::" + stringifyEvictPriority(op.getEvictPriority()).str());
      }
      st.v(numWords, numWords > 1).b(32);

      auto *dstOperand = builder.newAddressOperand(pointers[instId], "l");
      SmallVector<Value> words;
      for (unsigned i = 0; i < numWords; ++i) {
        if (bitWidth < 32) {
          auto vectorSize = 32 / bitWidth;
          auto vectorType = VectorType::get(vectorSize, elementType);
          Value word = llvm_undef(vectorType);
          for (unsigned j = 0; j < vectorSize; ++j) {
            auto loopIv = instIdToLoopIvs[instId][i * vectorSize + j];
            Value value = llvm_extractvalue(elementType, sourceStruct, loopIv);
            word = llvm_insertelement(vectorType, word, value,
                                      arith_constant_i32(j));
          }
          word = llvm_bitcast(i32Type, word);
          words.push_back(word);
        } else {
          auto loopIv = instIdToLoopIvs[instId][i];
          Value value = llvm_extractvalue(elementType, sourceStruct, loopIv);
          if (value.getType().isF32())
            value = llvm_bitcast(i32Type, value);
          words.push_back(value);
        }
      }
      auto *srcOperand = builder.newListOperand();
      for (auto word : words)
        srcOperand->listPushBack(builder.newOperand(word, "r"));
      st(dstOperand, srcOperand).predicate(predicates[instId], "b");
      builder.launch(rewriter, loc, voidType);
    }

    rewriter.eraseOp(op);
    return success();
  }

private:
  Type getSourceElementType(StGlobalOp op) const {
    auto elementType = getElementTypeOrSelf(op.getSource().getType());
    return typeConverter->convertType(elementType);
  }
};

} // namespace

void kapy::populateStGlobalOpToLLVMConversionPattern(
    const LLVMTypeConverter &typeConverter, RewritePatternSet &patterns) {
  patterns.add<StGlobalOpConversion>(typeConverter);
}
