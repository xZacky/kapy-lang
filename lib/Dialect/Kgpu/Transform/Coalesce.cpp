//===- Coalesce.cpp ---------------------------------------------*- C++ -*-===//
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

#include "kapy/Analysis/Integer.h"
#include "kapy/Analysis/Layout.h"
#include "kapy/Dialect/Kapy/IR/Kapy.h"
#include "kapy/Dialect/Kapy/IR/Utils.h"
#include "kapy/Dialect/Kgpu/IR/Kgpu.h"
#include "kapy/Dialect/Kgpu/Transform/Passes.h"

using namespace mlir;
using namespace mlir::kapy;

static Value getMemRef(Operation *op) {
  if (auto loadOp = dyn_cast<LoadOp>(op))
    return loadOp.getSource();
  if (auto storeOp = dyn_cast<StoreOp>(op))
    return storeOp.getTarget();
  if (auto rmwOp = dyn_cast<AtomicRMWOp>(op))
    return rmwOp.getSource();
  if (auto casOp = dyn_cast<AtomicCASOp>(op))
    return casOp.getSource();
  return Value();
}

static int64_t getVectorWidth(Operation *op,
                              ModuleIntegerInfoAnalysis &analysis) {
  auto memref = getMemRef(op);
  auto bitWidth = getIntOrFloatBitWidth(memref.getType());
  auto alignment = analysis.getIntegerInfo(memref)->getDivisibility();
  return std::min<int64_t>(alignment, 128 / bitWidth);
}

static Attribute chooseLayout(ModuleIntegerInfoAnalysis &analysis,
                              Operation *op, int64_t numWarps) {
  auto memref = getMemRef(op);
  auto memrefType = cast<KapyMemRefType>(memref.getType());
  auto glmemLayout = cast<GlobalMemLayoutAttr>(memrefType.getEncoding());
  auto rank = memrefType.getRank();
  unsigned majorAxis = rank - 1;
  for (auto it : llvm::enumerate(glmemLayout.getStrides()))
    if (it.value() == 1)
      majorAxis = it.index();
  auto vecWidth = getVectorWidth(op, analysis);
  auto numElems = memrefType.getNumElements();
  vecWidth = std::min(vecWidth, ceilDiv(numElems, numLanes));
  SmallVector<int64_t, 2> laneLoops(rank, 1);
  laneLoops[majorAxis] = vecWidth;
  auto *context = op->getContext();
  auto shape = memrefType.getShape();
  bool needTranspose = rank == 2 && majorAxis == 0;
  return getFragmentsLayout(context, laneLoops, shape, numWarps, needTranspose);
}

static void updateLayout(Operation *op, Attribute layout) {
  OpBuilder builder(op);
  auto loc = op->getLoc();
  SmallVector<Value> newOperands;
  for (auto operand : op->getOperands()) {
    if (auto tensorType = dyn_cast<RankedTensorType>(operand.getType())) {
      tensorType = cloneWith(tensorType, layout);
      newOperands.push_back(builder.create<ChangeOp>(loc, tensorType, operand));
    } else {
      newOperands.push_back(operand);
    }
  }
  SmallVector<Type> newTypes;
  for (auto type : op->getResultTypes()) {
    if (auto tensorType = dyn_cast<RankedTensorType>(type)) {
      tensorType = cloneWith(tensorType, layout);
      newTypes.push_back(tensorType);
    } else {
      newTypes.push_back(type);
    }
  }
  auto *newOp = builder.create(loc, op->getName().getIdentifier(), newOperands,
                               newTypes, op->getAttrs());
  for (unsigned i = 0; i < op->getNumResults(); ++i) {
    Value result = op->getResult(i);
    Value newResult = newOp->getResult(i);
    newResult = builder.create<ChangeOp>(loc, result.getType(), newResult);
    result.replaceAllUsesWith(newResult);
  }
  op->erase();
}

namespace {

#define GEN_PASS_DEF_KGPUCOALESCE
#include "kapy/Dialect/Kgpu/Transform/Passes.h.inc"

class KgpuCoalescePass : public impl::KgpuCoalesceBase<KgpuCoalescePass> {
public:
  virtual void runOnOperation() override {
    auto module = getOperation();
    ModuleIntegerInfoAnalysis analysis(module);
    auto numWarps = getNumWarps(module);
    // For each memory access operation, we determine what layout it should have
    // for best memory coalescing.
    module.walk([&](Operation *op) {
      if (!isa<LoadOp, StoreOp, AtomicRMWOp, AtomicCASOp>(op))
        return;
      updateLayout(op, chooseLayout(analysis, op, numWarps));
    });
  }
};

} // namespace

std::unique_ptr<Pass> kapy::createKgpuCoalescePass() {
  return std::make_unique<KgpuCoalescePass>();
}
