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

#include "kapy/Analysis/Layout.h"
#include "kapy/Dialect/Kapy/IR/Kapy.h"
#include "kapy/Dialect/Kapy/IR/Utils.h"
#include "kapy/Dialect/Kgpu/IR/Kgpu.h"
#include "kapy/Dialect/Kgpu/Transform/Passes.h"

using namespace mlir;
using namespace mlir::kapy;

static OpOperand *getMemRef(Operation *op) {
  auto effectOp = dyn_cast<MemoryEffectOpInterface>(op);
  if (!effectOp)
    return nullptr;
  SmallVector<SideEffects::EffectInstance<MemoryEffects::Effect>> effects;
  effectOp.getEffects(effects);
  for (auto &effect : effects)
    if (isa<MemoryEffects::Read, MemoryEffects::Write>(effect.getEffect()) &&
        effect.getResource() == GlobalMemory::get())
      return effect.getEffectValue<OpOperand *>();
  return nullptr;
}

static int64_t getVectorWidth(OpOperand *memref) {
  auto alignment = getAlignment(memref);
  auto bitWidth = getIntOrFloatBitWidth(memref->get().getType());
  return std::min<int64_t>(alignment * 8 / bitWidth, 128 / bitWidth);
}

static Attribute chooseLayout(OpOperand *memref, int64_t numWarps) {
  auto memrefType = cast<KapyMemRefType>(memref->get().getType());
  auto glmemLayout = cast<GlobalMemLayoutAttr>(memrefType.getEncoding());
  auto rank = memrefType.getRank();
  // Initialize major axis as rank, that means no contiguous axis.
  unsigned majorAxis = rank;
  for (auto it : llvm::enumerate(glmemLayout.getStrides()))
    if (it.value() == 1)
      majorAxis = it.index();
  // Currently we assume that must have a contiguous axis.
  if (majorAxis == rank)
    llvm_unreachable("can not find a contiguous axis");
  auto vecWidth = getVectorWidth(memref);
  auto numElems = memrefType.getNumElements();
  vecWidth = std::min(vecWidth, ceilDiv(numElems, numLanes));
  SmallVector<int64_t, 2> laneLoops(rank, 1);
  laneLoops[majorAxis] = vecWidth;
  auto *context = memref->getOwner()->getContext();
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
    Value oldResult = op->getResult(i);
    Value newResult = newOp->getResult(i);
    newResult = builder.create<ChangeOp>(loc, oldResult.getType(), newResult);
    oldResult.replaceAllUsesWith(newResult);
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
    auto numWarps = getNumWarps(module);
    // For each memory access operation, we determine what layout it should have
    // for best memory coalescing and update it.
    module.walk([&](Operation *op) {
      if (auto *memref = getMemRef(op))
        updateLayout(op, chooseLayout(memref, numWarps));
    });
  }
};

} // namespace

std::unique_ptr<Pass> kapy::createKgpuCoalescePass() {
  return std::make_unique<KgpuCoalescePass>();
}
