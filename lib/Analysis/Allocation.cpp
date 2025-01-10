//===- Allocation.cpp -------------------------------------------*- C++ -*-===//
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

#include "kapy/Analysis/Allocation.h"
#include "kapy/Analysis/OpHelpers.h"
#include "kapy/Dialect/Kapy/IR/Kapy.h"
#include "kapy/Dialect/Kapy/IR/Utils.h"
#include "kapy/Dialect/Kgpu/IR/Kgpu.h"

#include "mlir/Analysis/Liveness.h"

using namespace mlir;
using namespace mlir::kapy;

void Allocation::run(
    DenseMap<FunctionOpInterface, Allocation> &funcAllocations) {
  AllocationAnalysis(operation, this, &funcAllocations);
}

void AllocationAnalysis::addExplicit(Operation *op) {
  if (auto localAllocOp = dyn_cast<LocalAllocOp>(op)) {
    auto memrefType = localAllocOp.getType();
    auto numElems = product(memrefType.getShape());
    auto bitWidth = getIntOrFloatBitWidth(memrefType);
    auto size = numElems * ceilDiv<unsigned>(bitWidth, 8);
    Value result = localAllocOp.getResult();
    allocation->addBuffer<Buffer::BufferKind::Explicit>(result, size, 128);
  }
}

void AllocationAnalysis::addScratch(Operation *op) {
  if (auto reduceOp = dyn_cast<ReduceOp>(op)) {
    ReduceOpHelper helper(reduceOp);
    auto size = helper.getScratchSizeInBytes();
    if (size > 0)
      allocation->addBuffer<Buffer::BufferKind::Scratch>(op, size, 128);
  } else if (auto changeOp = dyn_cast<ChangeOp>(op)) {
    ChangeOpHelper helper(changeOp);
    auto size = helper.getScratchSizeInBytes();
    if (size > 0)
      allocation->addBuffer<Buffer::BufferKind::Scratch>(op, size, 128);
  }
}

void AllocationAnalysis::addVirtual(Operation *op) {
  if (auto callOp = dyn_cast<CallOpInterface>(op)) {
    auto *callable = callOp.resolveCallable();
    auto funcOp = dyn_cast_or_null<FunctionOpInterface>(callable);
    if (!funcOp)
      return;
    auto size = (*funcAllocations)[funcOp].getAllocatedSize();
    if (size > 0)
      allocation->addBuffer<Buffer::BufferKind::Virtual>(op, size, 128);
  }
}

void AllocationAnalysis::addBuffers() {
  operation->walk<WalkOrder::PreOrder>([&](Operation *op) {
    addExplicit(op);
    addScratch(op);
    addVirtual(op);
  });
}

void AllocationAnalysis::resolveExplicits(
    function_ref<Interval<OpId>(Value)> getLiveness) {
  for (auto [value, buffer] : allocation->explicits)
    bufferLivenesses[buffer] = getLiveness(value);
}

void AllocationAnalysis::resolveScratchsAndVirtuals(
    const DenseMap<Operation *, OpId> &opIds) {
  auto processBuffers = [&](const DenseMap<Operation *, Buffer *> &buffers) {
    for (auto [op, buffer] : buffers) {
      auto opId = opIds.lookup(op);
      bufferLivenesses.insert({buffer, Interval(opId, opId + 1)});
    }
  };
  processBuffers(allocation->scratchs);
  processBuffers(allocation->virtuals);
}

void AllocationAnalysis::resolveLivenesses() {
  // Assign an id to each operation using post-order traversal.
  // To achieve the correct liveness, the parent operation's id should be larger
  // than each of its child operation's id.
  //
  // Example:
  //
  //   %5 = kgpu.change %4
  //   %6 = scf.for ... iter_args(%arg0 = %0) -> ... {
  //     %2 = kgpu.change %5
  //     ...
  //     scf.yield %arg0
  //   }
  //
  // %5 is defined in the parent region and used in the child region, and is not
  // passed as a block argument. %6 should have an id larger than its child
  // operations, otherwise %5's liveness ends before the child operation's
  // liveness ends.
  DenseMap<Operation *, OpId> opIds;
  operation->walk<WalkOrder::PostOrder>(
      [&](Operation *op) { opIds[op] = opIds.size(); });

  // Analyze liveness of explicit buffers.
  Liveness liveness(operation);
  auto getLiveness = [&](Value value) {
    auto liveOps = liveness.resolveLiveness(value);
    auto minId = std::numeric_limits<OpId>::max();
    auto maxId = std::numeric_limits<OpId>::min();
    std::for_each(liveOps.begin(), liveOps.end(), [&](Operation *op) {
      if (opIds[op] < minId)
        minId = opIds[op];
      if (opIds[op] + 1 > maxId)
        maxId = opIds[op];
    });
    return Interval(minId, maxId);
  };

  resolveExplicits(getLiveness);
  resolveScratchsAndVirtuals(opIds);
}

void AllocationAnalysis::computeAndAllocate() {
  SmallVector<Buffer *> buffers;
  for (auto *buffer : llvm::make_first_range(bufferLivenesses))
    buffers.emplace_back(buffer);

  // NOTE: The original paper doesn't consider interference between the bumped
  // ranges. Buffers that previously do not interference with could interfere
  // after offset bumping if their liveness ranges overlap. Therefore, we rerun
  // the interference graph algorithm after bumping so that we regroup the
  // buffers and color them again. Since we always increase the buffer offset
  // and keep reducing conflicts, we will eventually reach a fixed point.
  DenseMap<Buffer *, DenseSet<Buffer *>> interference;
  buildInterference(buffers, interference);
  do {
    allocate(buffers, interference);
    buildInterference(buffers, interference);
  } while (!interference.empty());
}

void AllocationAnalysis::buildInterference(
    ArrayRef<Buffer *> buffers,
    DenseMap<Buffer *, DenseSet<Buffer *>> &interference) {
  // Reset interference graph.
  interference.clear();
  for (auto *buffer0 : buffers) {
    for (auto *buffer1 : buffers) {
      if (buffer0 == buffer1)
        continue;
      Interval memory0(buffer0->offset, buffer0->offset + buffer0->size);
      Interval memory1(buffer1->offset, buffer1->offset + buffer1->size);
      auto liveness0 = bufferLivenesses.lookup(buffer0);
      auto liveness1 = bufferLivenesses.lookup(buffer1);
      if (liveness0.intersects(liveness1) && memory0.intersects(memory1))
        interference[buffer0].insert(buffer1);
    }
  }
}

void AllocationAnalysis::allocate(
    ArrayRef<Buffer *> buffers,
    const DenseMap<Buffer *, DenseSet<Buffer *>> &interference) {
  // Reset allocated size.
  allocation->allocatedSize = 0;
  // First-fit graph coloring.
  // Neighbors are nodes that interference with each other. We color a node by
  // finding the index of the first available non-neighboring node or the first
  // neighboring node without any color.
  DenseMap<Buffer *, int> colors;
  for (auto *buffer : buffers) {
    // color < 0 means uncolored
    colors[buffer] = buffer == buffers[0] ? 0 : -1;
  }
  SmallVector<bool> available(buffers.size());
  for (auto *buffer0 : buffers) {
    std::fill(available.begin(), available.end(), true);
    for (auto *buffer1 : interference.lookup(buffer0)) {
      auto color = colors[buffer1];
      if (color >= 0)
        available[color] = false;
    }
    auto it = std::find(available.begin(), available.end(), true);
    colors[buffer0] = std::distance(available.begin(), it);
  }
  // Finalize allocation.
  // Example:
  //
  //   all alignments = 2
  //   color0: [0, 7), [0, 8), [0, 15) -> [0, 7), [0, 8), [0, 15)
  //   color1: [7, 9) -> [16, 16 + 9 - 7) -> [16, 18)
  //   color2: [8, 12) -> [18, 18 + 12 - 8) -> [18, 22)
  //
  for (auto *buffer0 : buffers) {
    int64_t newOffset = 0;
    for (auto *buffer1 : interference.lookup(buffer0))
      newOffset = std::max(newOffset, buffer1->offset + buffer1->size);
    if (colors.lookup(buffer0) != 0)
      buffer0->setOffsetAligned(newOffset);
    allocation->allocatedSize =
        std::max(allocation->allocatedSize, buffer0->offset + buffer0->size);
  }
}
