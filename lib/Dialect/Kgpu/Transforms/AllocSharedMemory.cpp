//===- AllocSharedMemory.cpp ------------------------------------*- C++ -*-===//
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
// This file is copied and modified from the triton project.
// https://github.com/triton-lang/triton
//
//===----------------------------------------------------------------------===//

#include "kapy/Analysis/AllocAnalysis.h"
#include "kapy/Dialect/Kgpu/IR/Kgpu.h"
#include "kapy/Dialect/Kgpu/Transforms/Passes.h"

using namespace mlir;
using namespace mlir::kapy;

namespace {

#define GEN_PASS_DEF_KGPUALLOCSHAREDMEMORY
#include "kapy/Dialect/Kgpu/Transforms/Passes.h.inc"

class KgpuAllocSharedMemoryPass
    : public impl::KgpuAllocSharedMemoryBase<KgpuAllocSharedMemoryPass> {
public:
  virtual void runOnOperation() override {
    auto module = getOperation();
    ModuleAllocAnalysis analysis(module);
    auto i64Type = IntegerType::get(&getContext(), 64);
    module.walk([&](FunctionOpInterface funcOp) {
      funcOp.walk([&](Operation *op) {
        auto *info = analysis.getData(funcOp);
        auto id = info->getBufferId(op);
        int64_t offset = -1;
        if (id != AllocInfo::INVALID_ID) {
          offset = info->getOffset(id);
        } else if (op->getNumResults() == 1) {
          auto result = op->getResult(0);
          auto id = info->getBufferId(result);
          if (id != AllocInfo::INVALID_ID)
            offset = info->getOffset(id);
        }
        if (offset == -1)
          return;
        op->setAttr(offsetAttrName, IntegerAttr::get(i64Type, offset));
      });
    });
    auto size = analysis.getAllocatedSize();
    module->setAttr(sizeAttrName, IntegerAttr::get(i64Type, size));
  }
};

} // namespace

std::unique_ptr<Pass> kapy::createKgpuAllocSharedMemoryPass() {
  return std::make_unique<KgpuAllocSharedMemoryPass>();
}
