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
#include "kapy/Analysis/Utils.h"
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
  return Value();
}

static SmallVector<unsigned, 4> getOrder(ArrayRef<int64_t> strides) {
  SmallVector<unsigned, 4> order;
  auto rank = strides.size();
  unsigned contiguousAxis = rank;
  for (unsigned i = 0; i < rank; ++i)
    if (strides[i] == 1)
      contiguousAxis = i;
    else
      order.push_back(i);
  // Currently must have a contiguous axis.
  assert(contiguousAxis < rank);
  order.push_back(contiguousAxis);
  return order;
}

static int64_t getVectorWidth(Operation *op,
                              ModuleIntegerInfoAnalysis &analysis) {
  auto memref = getMemRef(op);
  auto bitWidth = getIntOrFloatBitWidth(memref.getType());
  auto alignment = analysis.getIntegerInfo(memref)->getDivisibility();
  return std::min<int64_t>(alignment, 128 / bitWidth);
}

static void setLayout(ModuleIntegerInfoAnalysis &analysis, Operation *dstOp,
                      Value dstMem, int64_t numWarps,
                      llvm::MapVector<Operation *, Attribute> &layouts) {
  auto dstType = cast<KapyMemRefType>(dstMem.getType());
  auto dstLayout = cast<GlobalMemLayoutAttr>(dstType.getEncoding());
  auto dstOrder = getOrder(dstLayout.getStrides());

  auto dstShape = dstType.getShape();
  auto haveSameShape = [dstShape](Value curMem) {
    return cast<KapyMemRefType>(curMem.getType()).getShape() == dstShape;
  };

  SetVector<Operation *> sameLayoutOps;
  sameLayoutOps.insert(dstOp);
  for (auto *curOp : multiRootGetSlice(dstOp)) {
    auto curMem = getMemRef(curOp);
    if (!curMem || !haveSameShape(curMem) || sameLayoutOps.contains(curOp))
      continue;
    auto curType = cast<KapyMemRefType>(curMem.getType());
    auto curLayout = cast<GlobalMemLayoutAttr>(curType.getEncoding());
    if (getOrder(curLayout.getStrides()) == dstOrder)
      sameLayoutOps.insert(curOp);
  }

  auto dstVecWidth = getVectorWidth(dstOp, analysis);
  for (auto *curOp : sameLayoutOps) {
    if (curOp == dstOp)
      continue;
    auto curVecWidth = getVectorWidth(curOp, analysis);
    dstVecWidth = std::max(dstVecWidth, curVecWidth);
  }
  auto numElems = product(dstShape);
  dstVecWidth = std::min(dstVecWidth, ceilDiv(numElems, numLanes));
  if (!isa<LoadOp>(dstOp)) {
    // For operations can result in a global memory write, we should enforce
    // that each thread handles at most 128 bits, which is the widest
    // available vectorized store width. Otherwise, the store will have gaps
    // in the memory write at warp level, resulting in worse performance.
    // For loads, we can expect that the gaps won't matter due to L1 cache.
    dstVecWidth = std::min(dstVecWidth, getVectorWidth(dstOp, analysis));
  }

  auto rank = dstShape.size();
  SmallVector<int64_t, 4> loopsPerLane(rank, 1);
  loopsPerLane[dstOrder[rank - 1]] = dstVecWidth;

  layouts[dstOp] = getRegistersLayout(dstOp->getContext(), loopsPerLane,
                                      dstShape, dstOrder, numWarps);
}

static void coalesceOp(Operation *op, Attribute layout) {
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

    // For each memory access operation, we determine what layout the it should
    // have for best memory coalescing.
    llvm::MapVector<Operation *, Attribute> layouts;
    module.walk([&](Operation *op) {
      auto memref = getMemRef(op);
      if (!memref)
        return;
      setLayout(analysis, op, memref, numWarps, layouts);
    });

    for (auto [op, layout] : layouts)
      coalesceOp(op, layout);
  }
};

} // namespace

std::unique_ptr<Pass> kapy::createKgpuCoalescePass() {
  return std::make_unique<KgpuCoalescePass>();
}
