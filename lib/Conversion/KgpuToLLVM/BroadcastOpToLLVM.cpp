//===- BroadcastOpToLLVM.cpp ------------------------------------*- C++ -*-===//
//
// This file implements class to make BroadcastOp to LLVM compatible.
//
//===----------------------------------------------------------------------===//

#include "kapy/Conversion/KgpuToLLVM/ConvertUtils.h"
#include "kapy/Conversion/KgpuToLLVM/Patterns.h"
#include "kapy/Dialect/Kapy/IR/Kapy.h"
#include "kapy/Dialect/Kgpu/IR/Kgpu.h"
#include "kapy/Support/CommonUtils.h"
#include "mlir/Conversion/LLVMCommon/Pattern.h"

using namespace mlir;
using namespace mlir::kapy;
using namespace mlir::LLVM;

namespace {

class BroadcastOpConversion : public ConvertOpToLLVMPattern<BroadcastOp> {
public:
  using ConvertOpToLLVMPattern<BroadcastOp>::ConvertOpToLLVMPattern;

  virtual LogicalResult
  matchAndRewrite(BroadcastOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    assert(op->getNumOperands() == 1 && op->getNumResults() == 1);
    auto loc = op.getLoc();

    auto sourceType = op.getSource().getType();
    auto resultType = op.getType();
    auto sourceLayout = getLayout<FragmentsLayoutAttr>(sourceType);
    auto resultLayout = getLayout<FragmentsLayoutAttr>(resultType);
    auto oldMap = sourceLayout.getAffineMap(sourceType.getShape(), 3);
    auto newMap = resultLayout.getAffineMap(resultType.getShape(), 2);

    auto sourceValues = unpackLLVMStruct(rewriter, loc, adaptor.getSource());
    auto axis = getBroadcastAxis(op);
    auto elementType = getSourceElementType(op);
    auto loopSize = product(resultLayout.getLoopSpace(resultType.getShape()));

    SmallVector<Value> resultValues;
    for (int64_t loopIv = 0; loopIv < loopSize; ++loopIv) {
      auto indices = newMap.compose({0, loopIv});
      indices[axis] = 0;
      auto inputId = oldMap.compose(indices)[0];
      resultValues.push_back(sourceValues[inputId]);
    }

    auto structType = getResultStructType(op);
    packAndReplace(rewriter, op, structType, resultValues);
    return success();
  }

private:
  unsigned getBroadcastAxis(BroadcastOp op) const {
    auto oldShape = op.getSource().getType().getShape();
    auto newShape = op.getType().getShape();
    for (unsigned i = 0; i < 2; ++i)
      if (oldShape[i] == 1 && newShape[i] != 1)
        return i;
    llvm_unreachable("can not find the broadcast axis");
  }

  Type getSourceElementType(BroadcastOp op) const {
    auto elementType = getElementTypeOrSelf(op.getSource().getType());
    return typeConverter->convertType(elementType);
  }

  LLVMStructType getResultStructType(BroadcastOp op) const {
    auto resultType = typeConverter->convertType(op.getType());
    return cast<LLVMStructType>(resultType);
  }
};

} // namespace

void kapy::populateBroadcastOpToLLVMConversionPattern(
    const LLVMTypeConverter &typeConverter, RewritePatternSet &patterns) {
  patterns.add<BroadcastOpConversion>(typeConverter);
}
