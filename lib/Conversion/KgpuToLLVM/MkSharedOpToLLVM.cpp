//===- MkSharedOpToLLVM.cpp -------------------------------------*- C++ -*-===//
//
// This file implements class to make MkSharedOp to LLVM compatible.
//
//===----------------------------------------------------------------------===//

#include "kapy/Conversion/KgpuToLLVM/ConvertUtils.h"
#include "kapy/Conversion/KgpuToLLVM/Patterns.h"
#include "kapy/Dialect/Kapy/IR/Kapy.h"
#include "mlir/Conversion/LLVMCommon/Pattern.h"

using namespace mlir;
using namespace mlir::kapy;
using namespace mlir::LLVM;

static Value getAddressOfSharedArray(OpBuilder &rewriter,
                                     FunctionOpInterface funcOp) {
  LLVM::GlobalOp sharedArray;
  auto module = funcOp->getParentOfType<ModuleOp>();
  module.walk([&](LLVM::GlobalOp op) {
    if (op.getSymName() == "shared_array")
      sharedArray = op;
  });
  if (funcOp.getVisibility() == SymbolTable::Visibility::Public) {
    auto loc = funcOp.getLoc();
    return llvm_addressof(sharedArray);
  }
  return funcOp.getArgument(funcOp.getNumArguments() - 1);
}

static Value getPointerToSharedArray(OpBuilder &rewriter, Operation *op) {
  auto pointerType = LLVM::LLVMPointerType::get(rewriter.getContext(), 3);
  auto funcOp = op->getParentOfType<FunctionOpInterface>();
  auto loc = op->getLoc();
  auto i8Type = rewriter.getIntegerType(8);
  auto pointer = getAddressOfSharedArray(rewriter, funcOp);
  assert(op->hasAttr("kapy.offset"));
  auto offset = arith_constant_i32(getOffset(op));
  return llvm_getelementptr(pointerType, i8Type, pointer, offset);
}

namespace {

class MkSharedOpConversion : public ConvertOpToLLVMPattern<MkSharedOp> {
public:
  using ConvertOpToLLVMPattern<MkSharedOp>::ConvertOpToLLVMPattern;

  virtual LogicalResult
  matchAndRewrite(MkSharedOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    assert(op->getNumOperands() == 0 && op->getNumResults() == 1);
    auto loc = op.getLoc();
    auto structType = getResultStructType(op);
    auto pointerType = LLVMPointerType::get(getContext(), 3);
    Value llvmStruct = llvm_undef(structType);
    Value pointer = getPointerToSharedArray(rewriter, op);
    llvmStruct = llvm_insertvalue(structType, llvmStruct, pointer, 0);
    Value zero = arith_constant_i32(0);
    Value end0 = arith_constant_i32(op.getType().getShape()[0]);
    Value end1 = arith_constant_i32(op.getType().getShape()[1]);
    SmallVector<Value> i32Values;
    i32Values.push_back(zero);
    i32Values.push_back(zero);
    i32Values.push_back(end0);
    i32Values.push_back(end1);
    for (auto it : llvm::enumerate(i32Values))
      llvmStruct =
          llvm_insertvalue(structType, llvmStruct, it.value(), it.index() + 1);
    rewriter.replaceOp(op, llvmStruct);
    return success();
  }

private:
  LLVMStructType getResultStructType(MkSharedOp op) const {
    auto resultType = typeConverter->convertType(op.getType());
    return cast<LLVMStructType>(resultType);
  }
};

} // namespace

void kapy::populateMkSharedOpToLLVMConversionPattern(
    const LLVMTypeConverter &typeConverter, RewritePatternSet &patterns) {
  patterns.add<MkSharedOpConversion>(typeConverter);
}
