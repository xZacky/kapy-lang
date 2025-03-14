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
  using ConvertOpToLLVMPattern<MkGlobalOp>::ConvertOpToLLVMPattern;

  virtual LogicalResult
  matchAndRewrite(MkGlobalOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    assert(op->getNumOperands() == 6 && op->getNumResults() == 1);
    auto loc = op.getLoc();
    auto structType = getResultStructType(op);
    auto pointerType = LLVMPointerType::get(getContext(), 1);
    auto elementType = getResultElementType(op);
    Value llvmStruct = llvm_undef(structType);
    Value pointer = llvm_getelementptr(pointerType, elementType,   //
                                       adaptor.getGlobalAddress(), //
                                       adaptor.getDynamicOffset());
    llvmStruct = llvm_insertvalue(structType, llvmStruct, pointer, 0);
    Value zero = arith_constant_i32(0);
    SmallVector<Value> i32Values;
    i32Values.push_back(zero);
    i32Values.push_back(zero);
    i32Values.push_back(adaptor.getSize0());
    i32Values.push_back(adaptor.getSize1());
    i32Values.push_back(adaptor.getStride0());
    i32Values.push_back(adaptor.getStride1());
    for (auto it : llvm::enumerate(i32Values))
      llvmStruct =
          llvm_insertvalue(structType, llvmStruct, it.value(), it.index() + 1);
    rewriter.replaceOp(op, llvmStruct);
    return success();
  }

private:
  LLVMStructType getResultStructType(MkGlobalOp op) const {
    auto resultType = typeConverter->convertType(op.getType());
    return cast<LLVMStructType>(resultType);
  }

  Type getResultElementType(MkGlobalOp op) const {
    auto elementType = op.getType().getElementType();
    return typeConverter->convertType(elementType);
  }
};

} // namespace

void kapy::populateMkGlobalOpToLLVMConversionPattern(
    const LLVMTypeConverter &typeConverter, RewritePatternSet &patterns) {
  patterns.add<MkGlobalOpConversion>(typeConverter);
}
