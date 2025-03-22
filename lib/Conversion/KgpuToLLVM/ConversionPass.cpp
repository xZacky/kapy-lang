//===- ConversionPass.cpp ---------------------------------------*- C++ -*-===//
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

#include "kapy/Analysis/AllocAnalysis.h"
#include "kapy/Analysis/BlockAnalysis.h"
#include "kapy/Conversion/KgpuToLLVM/ConversionTarget.h"
#include "kapy/Conversion/KgpuToLLVM/Passes.h"
#include "kapy/Conversion/KgpuToLLVM/Patterns.h"
#include "kapy/Conversion/KgpuToLLVM/TypeConverter.h"
#include "kapy/Dialect/Kapy/IR/Kapy.h"
#include "kapy/Dialect/Kgpu/IR/Kgpu.h"
#include "mlir/Conversion/ControlFlowToLLVM/ControlFlowToLLVM.h"
#include "mlir/Conversion/LLVMCommon/ConversionTarget.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/LLVMIR/NVVMDialect.h"
#include "mlir/Pass/PassManager.h"

using namespace mlir;
using namespace mlir::kapy;

namespace {

#define GEN_PASS_DEF_CONVERTKGPUTOLLVM
#include "kapy/Conversion/KgpuToLLVM/Passes.h.inc"

class ConvertKgpuToLLVMPass
    : public impl::ConvertKgpuToLLVMBase<ConvertKgpuToLLVMPass> {
public:
  virtual void runOnOperation() override {
    allocSharedArray();
    insertBarriers();
    lowerFuncOps();
    lowerKgpuOps();
  }

private:
  void allocSharedArray() {
    auto module = getOperation();
    if (getSize(module) == 0)
      return;
    OpBuilder builder(module.getBodyRegion());
    auto loc = module.getLoc();
    // Set array size 0 and external linkage indicates that we use dynamic
    // shared allocation to allow a larger shared memory size for each kernel.
    builder.create<LLVM::GlobalOp>(
        loc, LLVM::LLVMArrayType::get(builder.getIntegerType(8), 0), false,
        LLVM::Linkage::External, "shared_array", Attribute(), 128,
        static_cast<unsigned>(NVVM::NVVMMemorySpace::kSharedMemorySpace));
  }

  void insertBarriers() {
    auto module = getOperation();
    ModuleAllocAnalysis allocAnalysis(module);
    ModuleBlockAnalysis blockAnalysis(&allocAnalysis);
    blockAnalysis.run();
  }

  void lowerFuncOps() {
    auto *context = &getContext();
    LowerToLLVMOptions options(context);
    KgpuToLLVMTypeConverter typeConverter(context, options);
    LLVMConversionTarget convTarget(*context);
    RewritePatternSet patterns(context);
    populateFuncOpToLLVMConversionPattern(typeConverter, patterns);

    auto module = getOperation();
    if (failed(applyPartialConversion(module, convTarget, std::move(patterns))))
      return signalPassFailure();
  }

  void lowerKgpuOps() {
    auto *context = &getContext();
    LowerToLLVMOptions options(context);
    KgpuToLLVMTypeConverter typeConverter(context, options);
    KgpuToLLVMConversionTarget convTarget(context, typeConverter);
    RewritePatternSet patterns(context);
    populateElementwiseOpToLLVMConversionPatterns(typeConverter, patterns);
    populateSelectOpToLLVMConversionPattern(typeConverter, patterns);
    populateMkGlobalOpToLLVMConversionPattern(typeConverter, patterns);
    populateSvGlobalOpToLLVMConversionPattern(typeConverter, patterns);
    populateLdGlobalOpToLLVMConversionPattern(typeConverter, patterns);
    populateStGlobalOpToLLVMConversionPattern(typeConverter, patterns);
    populateMkSharedOpToLLVMConversionPattern(typeConverter, patterns);
    populateSvSharedOpToLLVMConversionPattern(typeConverter, patterns);
    populateLdSharedOpToLLVMConversionPattern(typeConverter, patterns);
    populateStSharedOpToLLVMConversionPattern(typeConverter, patterns);
    populateLdMatrixOpToLLVMConversionPattern(typeConverter, patterns);
    populateCpAsyncOpToLLVMConversionPatterns(typeConverter, patterns);
    populateSplatLikeOpToLLVMConversionPatterns(typeConverter, patterns);
    populateBroadcastOpToLLVMConversionPattern(typeConverter, patterns);
    populateTransposeOpToLLVMConversionPattern(typeConverter, patterns);
    populateArangeOpToLLVMConversionPattern(typeConverter, patterns);
    populateMatmulOpToLLVMConversionPattern(typeConverter, patterns);
    populateReduceOpToLLVMConversionPattern(typeConverter, patterns);
    populateChangeOpToLLVMConversionPattern(typeConverter, patterns);
    populateCallReturnOpToLLVMConversionPatterns(typeConverter, patterns);
    cf::populateControlFlowToLLVMConversionPatterns(typeConverter, patterns);

    auto module = getOperation();
    if (failed(applyPartialConversion(module, convTarget, std::move(patterns))))
      return signalPassFailure();
  }
};

} // namespace

std::unique_ptr<Pass> kapy::createConvertKgpuToLLVMPass() {
  return std::make_unique<ConvertKgpuToLLVMPass>();
}

void kapy::registerConvertKgpuToLLVMPass() {
  registerPass([]() { return createConvertKgpuToLLVMPass(); });
}
