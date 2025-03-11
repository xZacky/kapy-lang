//===- MkGlobalOpToLLVM.cpp -------------------------------------*- C++ -*-===//
//
// This file implements class to make MkGlobalOp to LLVM compatible.
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

class MkGlobalOpConversion : public ConvertOpToLLVMPattern<MkGlobalOp> {
public:
  using ConvertOpToLLVMPattern::ConvertOpToLLVMPattern;

  virtual LogicalResult
  matchAndRewrite(MkGlobalOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    assert(op->getNumOperands() == 5 && op->getNumResults() == 1);
    auto loc = op.getLoc();
    auto structType = typeConverter->convertType(op.getType());
    auto pointerType = LLVMPointerType::get(getContext(), 1);
    Value llvmStruct = llvm_undef(structType);
    auto pointer = llvm_inttoptr(pointerType, adaptor.getAddress());
    llvmStruct = llvm_insertvalue(structType, llvmStruct, pointer, 0);
    auto zero = arith_constant_i32(0);
    SmallVector<Value> i32Values;
    i32Values.push_back(adaptor.getStride0());
    i32Values.push_back(adaptor.getStride1());
    i32Values.push_back(zero);
    i32Values.push_back(zero);
    i32Values.push_back(adaptor.getSize0());
    i32Values.push_back(adaptor.getSize1());
    for (auto it : llvm::enumerate(i32Values)) {
      auto value = it.value();
      auto index = it.index() + 1;
      llvmStruct = llvm_insertvalue(structType, llvmStruct, value, index);
    }
    rewriter.replaceOp(op, llvmStruct);
    return success();
  }
};

} // namespace

void kapy::populateMkGlobalOpToLLVMConversionPattern(
    const LLVMTypeConverter &typeConverter, RewritePatternSet &patterns) {
  patterns.add<MkGlobalOpConversion>(typeConverter);
}
