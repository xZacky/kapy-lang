//===- CoalescePass.cpp -----------------------------------------*- C++ -*-===//
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

#include "kapy/Analysis/AnalysisUtils.h"
#include "kapy/Analysis/LayoutUtils.h"
#include "kapy/Dialect/Kapy/IR/Kapy.h"
#include "kapy/Dialect/Kgpu/IR/Kgpu.h"
#include "kapy/Dialect/Kgpu/Transforms/Passes.h"
#include "kapy/Dialect/Kgpu/Transforms/TransformUtils.h"
#include "kapy/Support/CommonUtils.h"

using namespace mlir;
using namespace mlir::kapy;

static int64_t getVecWidth(Operation *op, RankedTensorType globalType) {
  auto alignment = getAlignment(op);
  auto bitWidth = getIntOrFloatBitWidth(globalType);
  auto vecWidth = std::min<int64_t>(alignment * 8 / bitWidth, 128 / bitWidth);
  auto numElems = globalType.getNumElements();
  vecWidth = std::min<int64_t>(vecWidth, ceilDiv(numElems, warpSize));
  vecWidth = std::max<int64_t>(vecWidth, 1);
  return vecWidth;
}

static FragmentsLayoutAttr getGlobalAccessLayout(Operation *op,
                                                 RankedTensorType globalType) {
  auto globalLayout = getLayout<Strided2dLayoutAttr>(globalType);
  // Initialize major axis as 2, that means no contiguous axis.
  unsigned j = 2;
  if (globalLayout.getStrideX() == 1)
    j = 0;
  if (globalLayout.getStrideY() == 1)
    j = 1;
  // Currently we assume that must have a contiguous axis.
  if (j == 2)
    llvm_unreachable("can not find a contiguous axis");
  auto vecWidth = getVecWidth(op, globalType);
  SmallVector<int64_t, 2> laneLoops{1, 1};
  laneLoops[j] = vecWidth;
  return getFragmentsLayout(laneLoops, globalType, j == 1);
}

static FragmentsLayoutAttr getSharedAccessLayout(Operation *op,
                                                 RankedTensorType sharedType) {
  auto bitWidth = getIntOrFloatBitWidth(sharedType);
  auto vecWidth = 128 / bitWidth;
  SmallVector<int64_t, 2> laneLoops{1, vecWidth};
  return getFragmentsLayout(laneLoops, sharedType);
}

namespace {

#define GEN_PASS_DEF_KGPUCOALESCE
#include "kapy/Dialect/Kgpu/Transforms/Passes.h.inc"

class KgpuCoalescePass : public impl::KgpuCoalesceBase<KgpuCoalescePass> {
public:
  virtual void runOnOperation() override {
    updateGlobalTensorType();
    processGlobalAccessOps();
    processSharedAccessOps();
    updateSharedTensorType();
  }

private:
  void updateGlobalTensorType();

  void coalesceOp(Operation *op, FragmentsLayoutAttr layout);

  void processGlobalAccessOps();

  void processSharedAccessOps();

  void updateSharedTensorType();
};

void KgpuCoalescePass::updateGlobalTensorType() {
  auto module = getOperation();
  module.walk([](MkGlobalOp op) {
    SmallVector<int64_t, 2> shape;
    for (auto size : {op.getSizeX(), op.getSizeY()}) {
      auto constantOp = size.getDefiningOp<arith::ConstantOp>();
      if (!constantOp) {
        shape.push_back(ShapedType::kDynamic);
        continue;
      }
      auto intAttr = dyn_cast<IntegerAttr>(constantOp.getValue());
      if (!intAttr) {
        shape.push_back(ShapedType::kDynamic);
        continue;
      }
      shape.push_back(intAttr.getInt());
    }
    SmallVector<int64_t, 2> strides;
    for (auto stride : {op.getStrideX(), op.getStrideY()}) {
      auto constantOp = stride.getDefiningOp<arith::ConstantOp>();
      if (!constantOp) {
        strides.push_back(ShapedType::kDynamic);
        continue;
      }
      auto intAttr = dyn_cast<IntegerAttr>(constantOp.getValue());
      if (!intAttr) {
        strides.push_back(ShapedType::kDynamic);
        continue;
      }
      strides.push_back(intAttr.getInt());
    }
    auto *context = op.getContext();
    auto layout = Strided2dLayoutAttr::get(context, strides[0], strides[1]);
    DenseSet<Value> seen;
    propagateMemoryLayout(op.getResult(), layout, seen);
  });
}

void KgpuCoalescePass::coalesceOp(Operation *op, FragmentsLayoutAttr layout) {
  OpBuilder builder(op);
  auto *context = op->getContext();
  auto loc = op->getLoc();

  SmallVector<Value> newOperands;
  for (auto operand : op->getOperands()) {
    auto tensorType = dyn_cast<RankedTensorType>(operand.getType());
    if (!tensorType || !inRegisterFile(tensorType)) {
      newOperands.push_back(operand);
    } else {
      tensorType = cloneWithLayout(tensorType, layout);
      newOperands.push_back(builder.create<ChangeOp>(loc, tensorType, operand));
    }
  }

  SmallVector<Type> newTypes;
  for (auto type : op->getResultTypes()) {
    auto tensorType = dyn_cast<RankedTensorType>(type);
    if (!tensorType || !inRegisterFile(tensorType)) {
      newTypes.push_back(type);
    } else {
      tensorType = cloneWithLayout(tensorType, layout);
      newTypes.push_back(tensorType);
    }
  }

  auto opName = op->getName().getIdentifier();
  auto *newOp =
      builder.create(loc, opName, newOperands, newTypes, op->getAttrs());
  for (unsigned i = 0; i < op->getNumResults(); ++i) {
    auto oldResult = op->getResult(i);
    auto newResult = newOp->getResult(i);
    oldResult.replaceAllUsesWith(
        builder.create<ChangeOp>(loc, oldResult.getType(), newResult));
  }
  op->erase();
}

void KgpuCoalescePass::processGlobalAccessOps() {
  auto module = getOperation();
  module.walk([&](Operation *op) {
    auto effectOp = dyn_cast<MemoryEffectOpInterface>(op);
    if (!effectOp)
      return;
    SmallVector<SideEffects::EffectInstance<MemoryEffects::Effect>> effects;
    effectOp.getEffects(effects);
    for (auto &effect : effects) {
      if (effect.getResource() == GlobalMemory::get() &&
          isa<MemoryEffects::Read, MemoryEffects::Write>(effect.getEffect())) {
        auto globalType = cast<RankedTensorType>(effect.getValue().getType());
        coalesceOp(op, getGlobalAccessLayout(op, globalType));
        break;
      }
    }
  });
}

void KgpuCoalescePass::processSharedAccessOps() {
  auto module = getOperation();
  module.walk([&](Operation *op) {
    if (isa<LdMatrixOp, CpAsyncGlobalToSharedOp>(op))
      return;
    auto effectOp = dyn_cast<MemoryEffectOpInterface>(op);
    if (!effectOp)
      return;
    SmallVector<SideEffects::EffectInstance<MemoryEffects::Effect>> effects;
    effectOp.getEffects(effects);
    for (auto &effect : effects) {
      if (effect.getResource() == SharedMemory::get() &&
          isa<MemoryEffects::Read, MemoryEffects::Write>(effect.getEffect())) {
        auto sharedType = cast<RankedTensorType>(effect.getValue().getType());
        coalesceOp(op, getSharedAccessLayout(op, sharedType));
        break;
      }
    }
  });
}

void KgpuCoalescePass::updateSharedTensorType() {
  auto module = getOperation();
  module.walk([](MkSharedOp op) {
    if (hasLayout(op.getResult().getType()))
      return;
    auto slice = multiRootGetSlice(op);
    bool rowMajor = true;
    for (auto *op : slice) {
      if (auto cpAsyncOp = dyn_cast<CpAsyncGlobalToSharedOp>(op)) {
        auto type = cast<RankedTensorType>(cpAsyncOp.getSource().getType());
        auto layout = getLayout<Strided2dLayoutAttr>(type);
        rowMajor = layout.isRowMajor();
        // Currently we assume that all the same.
        break;
      }
    }
    auto *context = op.getContext();
    auto layout = SwizzlingLayoutAttr::get(context, rowMajor);
    DenseSet<Value> seen;
    propagateMemoryLayout(op.getResult(), layout, seen);
  });
}

} // namespace

std::unique_ptr<Pass> kapy::createKgpuCoalescePass() {
  return std::make_unique<KgpuCoalescePass>();
}
