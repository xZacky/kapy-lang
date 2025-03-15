//===- SvSharedOpToLLVM.cpp -------------------------------------*- C++ -*-===//
//
// This file implements class to make SvSharedOp to LLVM compatible.
//
//===----------------------------------------------------------------------===//

#include "kapy/Conversion/KgpuToLLVM/ConvertUtils.h"
#include "kapy/Conversion/KgpuToLLVM/Patterns.h"
#include "kapy/Dialect/Kapy/IR/Kapy.h"
#include "mlir/Conversion/LLVMCommon/Pattern.h"

using namespace mlir;
using namespace mlir::kapy;
using namespace mlir::LLVM;

namespace {

class SvSharedOpConversion : public ConvertOpToLLVMPattern<SvSharedOp> {
public:
  using ConvertOpToLLVMPattern<SvSharedOp>::ConvertOpToLLVMPattern;

  virtual LogicalResult
  matchAndRewrite(SvSharedOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    assert(op->getNumOperands() == 5 && op->getNumResults() == 1);
    auto loc = op.getLoc();
    SmallVector<Value> i32Values;
    auto i32Type = rewriter.getIntegerType(32);
    Value llvmStruct = adaptor.getSource();
    Value start0 = llvm_extractvalue(i32Type, llvmStruct, 1);
    Value start1 = llvm_extractvalue(i32Type, llvmStruct, 2);
    i32Values.push_back(arith_addi(start0, adaptor.getStart0()));
    i32Values.push_back(arith_addi(start1, adaptor.getStart1()));
    i32Values.push_back(arith_addi(start0, adaptor.getEnd0()));
    i32Values.push_back(arith_addi(start1, adaptor.getEnd1()));
    auto structType = llvmStruct.getType();
    for (auto it : llvm::enumerate(i32Values))
      llvmStruct =
          llvm_insertvalue(structType, llvmStruct, it.value(), it.index() + 1);
    rewriter.replaceOp(op, llvmStruct);
    return success();
  }
};

} // namespace

void kapy::populateSvSharedOpToLLVMConversionPattern(
    const LLVMTypeConverter &typeConverter, RewritePatternSet &patterns) {
  patterns.add<SvSharedOpConversion>(typeConverter);
}
