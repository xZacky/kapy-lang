//===- ArangeOpToLLVM.cpp ---------------------------------------*- C++ -*-===//
//
// This file implements class to make ArangeOp to LLVM compatible.
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

class ArangeOpConversion : public ConvertOpToLLVMPattern<ArangeOp> {
public:
  using ConvertOpToLLVMPattern<ArangeOp>::ConvertOpToLLVMPattern;

  virtual LogicalResult
  matchAndRewrite(ArangeOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    assert(op->getNumOperands() == 0 && op->getNumResults() == 1);
    auto loc = op.getLoc();
    auto tensorType = op.getType();
    auto tensorLayout = getLayout<FragmentsLayoutAttr>(tensorType);
    auto map = tensorLayout.getAffineMap(tensorType.getShape(), 1);
    auto loopSize = product(tensorLayout.getLoopSpace(tensorType.getShape()));
    auto axis = op.getAxis();
    auto start = arith_constant_i32(op.getStart());
    auto laneId = rewriter.create<LaneIdOp>(loc);
    SmallVector<Value> resultValues;
    for (int64_t i = 0; i < loopSize; ++i) {
      Value loopIv = arith_constant_i32(i);
      if (axis == 0) {
        Value index =
            expandAffineExpr(rewriter, loc, map.getResult(0), {laneId, loopIv});
        resultValues.push_back(arith_addi(start, index));
      } else {
        Value index =
            expandAffineExpr(rewriter, loc, map.getResult(1), {laneId, loopIv});
        resultValues.push_back(arith_addi(start, index));
      }
    }
    auto structType = getResultStructType(op);
    packAndReplace(rewriter, op, structType, resultValues);
    return success();
  }

private:
  LLVMStructType getResultStructType(ArangeOp op) const {
    auto resultType = typeConverter->convertType(op.getType());
    return cast<LLVMStructType>(resultType);
  }
};

} // namespace

void kapy::populateArangeOpToLLVMConversionPattern(
    const LLVMTypeConverter &typeConverter, RewritePatternSet &patterns) {
  patterns.add<ArangeOpConversion>(typeConverter);
}
