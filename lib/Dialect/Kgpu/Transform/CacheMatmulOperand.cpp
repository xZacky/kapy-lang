//===- CacheMatmulOperand.cpp -----------------------------------*- C++ -*-===//
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

#include "kapy/Analysis/Layout.h"
#include "kapy/Dialect/Kapy/IR/Kapy.h"
#include "kapy/Dialect/Kgpu/IR/Kgpu.h"
#include "kapy/Dialect/Kgpu/Transform/Passes.h"

using namespace mlir;
using namespace mlir::kapy;

namespace {

#define GEN_PASS_DEF_KGPUCACHEMATMULOPERAND
#include "kapy/Dialect/Kgpu/Transform/Passes.h.inc"

class KgpuCacheMatmulOperandPass
    : public impl::KgpuCacheMatmulOperandBase<KgpuCacheMatmulOperandPass> {
public:
  virtual void runOnOperation() override {
    auto module = getOperation();
    module.walk([&](ChangeOp changeOp) {
      OpBuilder builder(changeOp);
      auto loc = changeOp.getLoc();
      auto operand = changeOp.getOperand();
      auto operandType = operand.getType();
      auto resultType = changeOp.getType();
      auto mmopdLayout =
          dyn_cast<MmOperandLayoutAttr>(resultType.getEncoding());
      if (!mmopdLayout)
        return;
      auto nvmmaLayout =
          dyn_cast<NvidiaMmaLayoutAttr>(operandType.getEncoding());
      if (isNvidiaMmaToMmOperandShortcut(nvmmaLayout, mmopdLayout))
        return;
      auto fragsLayout =
          dyn_cast<FragmentsLayoutAttr>(operandType.getEncoding());
      if (!fragsLayout)
        return;
      bool needTranspose = fragsLayout.getMajorAxis() == 0;
      auto shmemLayout = getSharedMemLayout(resultType, needTranspose);
      auto memrefType = KapyMemRefType::get(
          resultType.getShape(), resultType.getElementType(), shmemLayout);
      auto allocOp = builder.create<LocalAllocOp>(loc, memrefType, operand);
      auto loadOp = builder.create<LocalLoadOp>(loc, resultType, allocOp);
      changeOp.replaceAllUsesWith(loadOp.getResult());
      changeOp.erase();
    });
  }
};

} // namespace

std::unique_ptr<Pass> kapy::createKgpuCacheMatmulOperandPass() {
  return std::make_unique<KgpuCacheMatmulOperandPass>();
}
