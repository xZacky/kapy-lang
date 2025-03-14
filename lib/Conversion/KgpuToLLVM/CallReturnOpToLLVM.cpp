//===- CallReturnOpToLLVM.cpp -----------------------------------*- C++ -*-===//
//
// Copyright 2018-2020 Philippe Tillet
// Copyright 2020-2022 OpenAI
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.
//
//===----------------------------------------------------------------------===//
//
// This file is modified from the triton project.
// https://github.com/triton-lang/triton
//
//===----------------------------------------------------------------------===//

#include "kapy/Conversion/KgpuToLLVM/ConvertUtils.h"
#include "kapy/Conversion/KgpuToLLVM/Patterns.h"
#include "kapy/Dialect/Kapy/IR/Kapy.h"
#include "mlir/Conversion/LLVMCommon/Pattern.h"

using namespace mlir;
using namespace mlir::kapy;

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

class ReturnOpConversion : public ConvertOpToLLVMPattern<ReturnOp> {
public:
  using ConvertOpToLLVMPattern<ReturnOp>::ConvertOpToLLVMPattern;

  virtual LogicalResult
  matchAndRewrite(ReturnOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto funcOp = op->getParentOfType<LLVM::LLVMFuncOp>();
    if (funcOp->hasAttr("nvvm.kernel")) {
      if (op.getNumOperands())
        return rewriter.notifyMatchFailure(
            op, "kernel function do not support return with operands");
      rewriter.replaceOpWithNewOp<LLVM::ReturnOp>(op, TypeRange(), ValueRange(),
                                                  op->getAttrs());
    } else {
      LLVM::ReturnOp newOp;
      auto loc = op.getLoc();
      if (adaptor.getOperands().size() < 2) {
        newOp = rewriter.create<LLVM::ReturnOp>(loc, adaptor.getOperands());
      } else {
        auto structType =
            getTypeConverter()->packFunctionResults(funcOp.getResultTypes());
        Value llvmStruct = llvm_undef(structType);
        for (auto it : llvm::enumerate(adaptor.getOperands()))
          llvmStruct =
              llvm_insertvalue(structType, llvmStruct, it.value(), it.index());
        newOp = rewriter.create<LLVM::ReturnOp>(loc, llvmStruct);
      }
      newOp->setAttrs(op->getAttrs());
      rewriter.replaceOp(op, newOp->getResults());
    }
    return success();
  }
};

class CallOpConversion : public ConvertOpToLLVMPattern<CallOp> {
public:
  using ConvertOpToLLVMPattern<CallOp>::ConvertOpToLLVMPattern;

  virtual LogicalResult
  matchAndRewrite(CallOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto operands = promoteOperands(op, adaptor, rewriter);
    auto newOp = convertCallOpToLLVMCallOp(op, operands, rewriter);
    if (!newOp)
      return failure();
    auto results = getCallOpResults(newOp, op.getNumResults(), rewriter);
    return success();
  }

private:
  SmallVector<Value>
  promoteOperands(CallOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const {
    // Get the last argument of the caller, which is the current stack pointer
    // of shared memory and append it to the operands of the CallOp.
    auto caller = op->getParentOfType<FunctionOpInterface>();
    auto operands = getTypeConverter()->promoteOperands(
        op.getLoc(), op.getOperands(), adaptor.getOperands(), rewriter);
    if (!caller->hasAttr("kapy.offset"))
      operands.push_back(getAddressOfSharedArray(rewriter, caller));
    else
      operands.push_back(getPointerToSharedArray(rewriter, op));
    return operands;
  }

  LLVM::CallOp
  convertCallOpToLLVMCallOp(CallOp op, ArrayRef<Value> operands,
                            ConversionPatternRewriter &rewriter) const {
    auto loc = op.getLoc();
    if (op.getNumResults() == 0) {
      return rewriter.create<LLVM::CallOp>(loc, TypeRange(), operands,
                                           op->getAttrs());
    } else {
      auto structType =
          getTypeConverter()->packFunctionResults(op.getResultTypes());
      return rewriter.create<LLVM::CallOp>(loc, structType, operands,
                                           op->getAttrs());
    }
  }

  SmallVector<Value>
  getCallOpResults(LLVM::CallOp op, unsigned numResults,
                   ConversionPatternRewriter &rewriter) const {
    SmallVector<Value> results;
    if (numResults < 2) {
      results.append(op.result_begin(), op.result_end());
    } else {
      for (unsigned i = 0; i < numResults; ++i) {
        auto loc = op.getLoc();
        results.push_back(llvm_extractvalue(op.getResult(), i));
      }
    }
    return results;
  }
};

} // namespace

void kapy::populateCallReturnOpToLLVMConversionPatterns(
    const LLVMTypeConverter &typeConverter, RewritePatternSet &patterns) {
  patterns.add<ReturnOpConversion>(typeConverter);
  patterns.add<CallOpConversion>(typeConverter);
}
