//===- FuncOpToLLVM.cpp -----------------------------------------*- C++ -*-===//
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

#include "kapy/Conversion/KgpuToLLVM/Patterns.h"
#include "kapy/Dialect/Kapy/IR/Kapy.h"
#include "mlir/Conversion/LLVMCommon/Pattern.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"

namespace mlir {

FailureOr<LLVM::LLVMFuncOp>
convertFuncOpToLLVMFuncOp(FunctionOpInterface funcOp,
                          ConversionPatternRewriter &rewriter,
                          const LLVMTypeConverter &typeConverter);

}

using namespace mlir;
using namespace mlir::kapy;
using namespace mlir::LLVM;

static bool isKernel(FunctionOpInterface funcOp) {
  return funcOp.getVisibility() == SymbolTable::Visibility::Public;
}

namespace {

class FuncOpConversion : public ConvertOpToLLVMPattern<FuncOp> {
public:
  using ConvertOpToLLVMPattern::ConvertOpToLLVMPattern;

  virtual LogicalResult
  matchAndRewrite(FuncOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    if (!isKernel(op))
      op = amendFuncOp(op, rewriter);
    auto newOp = *convertFuncOpToLLVMFuncOp(op, rewriter, *getTypeConverter());
    if (!newOp)
      return failure();
    auto *context = op.getContext();
    if (isKernel(op)) {
      auto u1Type = IntegerType::get(context, 1, IntegerType::Unsigned);
      newOp->setAttr("nvvm.kernel", rewriter.getIntegerAttr(u1Type, 1));
      newOp.setLinkage(Linkage::External);
    } else {
      // The noinline attribute will be used by the LLVM codegen to prevent
      // inlining.
      auto noinlineAttr = rewriter.getStringAttr("noinline");
      newOp.setPassthroughAttr(ArrayAttr::get(context, noinlineAttr));
      newOp.setLinkage(Linkage::External);
    }
    // Set an attribute for reqntid, it could be used in latter LLVM codegen for
    // nvvm.annotation metadata.
    auto numWarps = getNumWarps(op->getParentOfType<ModuleOp>());
    auto reqntidAttr = rewriter.getDenseI32ArrayAttr(numWarps * 32);
    newOp->setAttr("nvvm.reqntid", reqntidAttr);
    rewriter.eraseOp(op);
    return success();
  }

private:
  /// Only retain attributes that are not constructed by LLVMFuncOp::build.
  /// If `filterArgAttrs` is set, also filter out argument attributes.
  static void filterFuncAttributes(FuncOp op, bool filterArgAttrs,
                                   SmallVectorImpl<NamedAttribute> &attrs) {
    for (auto attr : op->getAttrs()) {
      if (attr.getName() == SymbolTable::getSymbolAttrName() ||
          attr.getName() == op.getFunctionTypeAttrName() ||
          attr.getName() == "std.varargs" ||
          (filterArgAttrs && attr.getName() == op.getArgAttrsAttrName()))
        continue;
      attrs.push_back(attr);
    }
  }

  FuncOp amendFuncOp(FuncOp op, ConversionPatternRewriter &rewriter) const {
    auto *context = op.getContext();
    // Push back a variable that indicates the current stack pointer of shared
    // memory to the function arguments.
    auto pointerType = LLVMPointerType::get(rewriter.getContext(), 3);
    // 1. Modify the function type to add the new argument.
    auto funcType = op.getFunctionType();
    auto inputs = llvm::to_vector(funcType.getInputs());
    inputs.push_back(pointerType);
    funcType = FunctionType::get(context, inputs, funcType.getResults());
    // 2. Modify the argument attributes to add the new argument.
    SmallVector<NamedAttribute> attrs;
    filterFuncAttributes(op, true, attrs);
    auto argAttrs = llvm::to_vector(op.getAllArgAttrs());
    argAttrs.emplace_back(DictionaryAttr::get(context));
    auto arrayAttr = rewriter.getArrayAttr(argAttrs);
    attrs.push_back(rewriter.getNamedAttr(op.getArgAttrsAttrName(), arrayAttr));
    // 3. Add a new argument to the region.
    auto loc = op.getLoc();
    auto newOp = rewriter.create<FuncOp>(loc, op.getName(), funcType, attrs);
    auto &region = op.getBody();
    rewriter.inlineRegionBefore(region, newOp.getBody(), newOp.getBody().end());
    rewriter.eraseOp(op);
    return newOp;
  }
};

} // namespace

void kapy::populateFuncOpToLLVMConversionPattern(
    const LLVMTypeConverter &typeConverter, RewritePatternSet &patterns) {
  patterns.add<FuncOpConversion>(typeConverter);
}
