//===- TransposeOpToLLVM.cpp ------------------------------------*- C++ -*-===//
//
// This file implements class to make TransposeOp to LLVM compatible.
//
//===----------------------------------------------------------------------===//

#include "kapy/Conversion/KgpuToLLVM/Patterns.h"
#include "kapy/Dialect/Kapy/IR/Kapy.h"
#include "mlir/Conversion/LLVMCommon/Pattern.h"

using namespace mlir;
using namespace mlir::kapy;

namespace {

class TransposeOpConversion : public ConvertOpToLLVMPattern<TransposeOp> {
public:
  using ConvertOpToLLVMPattern<TransposeOp>::ConvertOpToLLVMPattern;

  virtual LogicalResult
  matchAndRewrite(TransposeOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    assert(op->getNumOperands() == 1 && op->getNumResults() == 1);
    rewriter.replaceOp(op, op.getSource());
    return success();
  }
};

} // namespace

void kapy::populateTransposeOpToLLVMConversionPattern(
    const LLVMTypeConverter &typeConverter, RewritePatternSet &patterns) {
  patterns.add<TransposeOpConversion>(typeConverter);
}
