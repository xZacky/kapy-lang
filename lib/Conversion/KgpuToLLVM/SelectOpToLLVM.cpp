//===- SelectOpToLLVM.cpp ---------------------------------------*- C++ -*-===//
//
// This file implements class to make arith::SelectOp to LLVM compatible.
//
//===----------------------------------------------------------------------===//

#include "kapy/Conversion/KgpuToLLVM/ConvertUtils.h"
#include "kapy/Conversion/KgpuToLLVM/Patterns.h"
#include "kapy/Dialect/Kapy/IR/Kapy.h"
#include "mlir/Conversion/LLVMCommon/Pattern.h"

using namespace mlir;
using namespace mlir::kapy;

namespace {

class SelectOpConversion : public ConvertOpToLLVMPattern<arith::SelectOp> {
public:
  using ConvertOpToLLVMPattern<arith::SelectOp>::ConvertOpToLLVMPattern;

  virtual LogicalResult
  matchAndRewrite(arith::SelectOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    assert(op->getNumOperands() == 3 && op->getNumResults() == 1);
    auto loc = op.getLoc();
    auto resultType = getResultStructType(op);
    auto condition = op.getCondition();
    auto trueValue = op.getTrueValue();
    auto falseValue = op.getFalseValue();
    auto trueType = cast<RankedTensorType>(trueValue.getType());
    auto falseType = cast<RankedTensorType>(falseValue.getType());
    if (isa<RankedTensorType>(condition.getType())) {
      SmallVector<SmallVector<Value>> operandsValues;
      for (auto llvmStruct : adaptor.getOperands())
        operandsValues.push_back(unpackLLVMStruct(rewriter, loc, llvmStruct));

      auto resultValues = doConversion(op, adaptor, rewriter,
                                       MultipleValuesRange(operandsValues));
      auto resultStruct =
          packToLLVMStruct(rewriter, loc, resultType, resultValues);
      rewriter.replaceOp(op, resultStruct);
      return success();
    } else {
      condition = adaptor.getCondition();
      trueValue = adaptor.getTrueValue();
      falseValue = adaptor.getFalseValue();
      if (inRegisterFile(trueType) && inRegisterFile(falseType)) {
        SmallVector<SmallVector<Value>> operandsValues;
        operandsValues.push_back(SmallVector<Value>{condition});
        operandsValues.push_back(unpackLLVMStruct(rewriter, loc, trueValue));
        operandsValues.push_back(unpackLLVMStruct(rewriter, loc, falseValue));

        auto resultValues = doConversion(op, adaptor, rewriter,
                                         MultipleValuesRange(operandsValues));
        auto resultStruct =
            packToLLVMStruct(rewriter, loc, resultType, resultValues);
        rewriter.replaceOp(op, resultStruct);
        return success();
      } else {
        auto newOp = arith_select(resultType, condition, trueValue, falseValue);
        rewriter.replaceOp(op, newOp);
        return success();
      }
    }
  }

private:
  LLVM::LLVMStructType getResultStructType(arith::SelectOp op) const {
    auto resultType = typeConverter->convertType(op.getType());
    return cast<LLVM::LLVMStructType>(resultType);
  }

  Type getResultElementType(arith::SelectOp op) const {
    auto resultType = getElementTypeOrSelf(op.getType());
    return typeConverter->convertType(resultType);
  }

  SmallVector<Value> doConversion(arith::SelectOp op, OpAdaptor adaptor,
                                  ConversionPatternRewriter &rewriter,
                                  MultipleValuesRange operandsValues) const {
    assert(operandsValues.size() == 3);
    auto loc = op.getLoc();
    auto resultType = getResultElementType(op);
    SmallVector<Value> resultValues;
    if (operandsValues[0].size() == 1) {
      auto condition = operandsValues[0][0];
      for (unsigned i = 0; i < operandsValues[1].size(); ++i) {
        auto trueValue = operandsValues[1][i];
        auto falseValue = operandsValues[2][i];
        resultValues.push_back(
            arith_select(resultType, condition, trueValue, falseValue));
      }
    } else {
      for (unsigned i = 0; i < operandsValues[0].size(); ++i) {
        auto condition = operandsValues[0][i];
        auto trueValue = operandsValues[1][i];
        auto falseValue = operandsValues[2][i];
        resultValues.push_back(
            arith_select(resultType, condition, trueValue, falseValue));
      }
    }
    return resultValues;
  }
};

} // namespace

void kapy::populateSelectOpToLLVMConversionPattern(
    const LLVMTypeConverter &typeConverter, RewritePatternSet &patterns) {
  patterns.add<SelectOpConversion>(typeConverter);
}
