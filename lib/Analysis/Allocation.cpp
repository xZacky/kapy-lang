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

namespace mlir {
namespace kapy {

class AllocationAnalysis {
public:
  AllocationAnalysis(FunctionOpInterface funcOp, Allocation *allocation)
      : funcOp(funcOp), allocation(allocation) {}

  void run(DenseMap<FunctionOpInterface, Allocation> &funcToAllocation) {
    addBuffers(funcToAllocation);
    resolveLiveness();
    computeAndAllocate();
  }

private:
  using Buffer = Allocation::Buffer;
  using OpId = int64_t;

  FunctionOpInterface funcOp;
  Allocation *allocation;
  llvm::MapVector<Buffer *, Interval<OpId>> bufferToLiveness;

  void addBuffers(DenseMap<FunctionOpInterface, Allocation> &funcToAllocation);

  void resolveLiveness();

  // Compute the shared memory offsets and allocate for all the related buffers
  // while considering interference.
  // Paper: Algorithms for Compile-time Memory Optimization
  // https://dl.acm.org/doi/pdf/10.5555/314500.315082
  void computeAndAllocate();
};

} // namespace kapy
} // namespace mlir

using namespace mlir;
using namespace mlir::kapy;

void Allocation::run(
    DenseMap<FunctionOpInterface, Allocation> &funcToAllocation) {
  AllocationAnalysis(funcOp, this).run(funcToAllocation);
}

void AllocationAnalysis::addBuffers(
    DenseMap<FunctionOpInterface, Allocation> &funcToAllocation) {
  funcOp->walk<WalkOrder::PreOrder>([&](Operation *op) {
    if (auto allocOp = dyn_cast<LocalAllocOp>(op)) {
      auto memrefType = allocOp.getType();
      auto numElems = product(memrefType.getShape());
      auto bitWidth = getIntOrFloatBitWidth(memrefType);
      auto size = numElems * ceilDiv<unsigned>(bitWidth, 8);
      Value result = allocOp.getResult();
      allocation->addBuffer<Buffer::BufferKind::EXPLICIT>(result, size);
    } else if (auto reduceOp = dyn_cast<ReduceOp>(op)) {
      ReduceOpHelper helper(reduceOp);
      auto size = helper.getScratchSizeInBytes();
      if (size > 0)
        allocation->addBuffer<Buffer::BufferKind::SCRATCH>(op, size);
    } else if (auto changeOp = dyn_cast<ChangeOp>(op)) {
      ChangeOpHelper helper(changeOp);
      auto size = helper.getScratchSizeInBytes();
      if (size > 0)
        allocation->addBuffer<Buffer::BufferKind::SCRATCH>(op, size);
    } else if (auto callOp = dyn_cast<CallOpInterface>(op)) {
      auto *callable = callOp.resolveCallable();
      auto funcOp = dyn_cast_if_present<FunctionOpInterface>(callable);
      if (!funcOp)
        return;
      auto size = funcToAllocation[funcOp].getAllocatedSize();
      if (size > 0)
        allocation->addBuffer<Buffer::BufferKind::VIRTUAL>(op, size);
    }
  });
}

void AllocationAnalysis::resolveLiveness() {
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
  DenseMap<Operation *, OpId> opToId;
  funcOp->walk<WalkOrder::PostOrder>(
      [&](Operation *op) { opToId[op] = opToId.size(); });

  // Analyze liveness of explicit buffers.
  Liveness liveness(funcOp);
  auto getLiveness = [&](Value value) {
    auto liveOps = liveness.resolveLiveness(value);
    auto minId = std::numeric_limits<OpId>::max();
    auto maxId = std::numeric_limits<OpId>::min();
    std::for_each(liveOps.begin(), liveOps.end(), [&](Operation *op) {
      if (opToId[op] < minId)
        minId = opToId[op];
      if (opToId[op] + 1 > maxId)
        maxId = opToId[op];
    });
    return Interval(minId, maxId);
  };
  for (auto [value, buffer] : allocation->explicits)
    bufferToLiveness[buffer] = getLiveness(value);

  // Analyze liveness of scratch and virtual buffers.
  auto processBuffers = [&](const DenseMap<Operation *, Buffer *> &buffers) {
    for (auto [op, buffer] : buffers) {
      auto opId = opToId.lookup(op);
      bufferToLiveness.insert({buffer, Interval(opId, opId + 1)});
    }
  };
  processBuffers(allocation->scratchs);
  processBuffers(allocation->virtuals);
}

void AllocationAnalysis::computeAndAllocate() {
  SmallVector<Buffer *> buffers;
  for (auto [buffer, liveness] : bufferToLiveness)
    buffers.emplace_back(buffer);

  // NOTE: The original paper doesn't consider interference between the bumped
  // ranges. Buffers that previously do not interference with could interfere
  // after offset bumping if their liveness ranges overlap. Therefore, we rerun
  // the interference graph algorithm after bumping so that we regroup the
  // buffers and color them again. Since we always increase the buffer offset
  // and keep reducing conflicts, we will eventually reach a fixed point.
  DenseMap<Buffer *, DenseSet<Buffer *>> graph;

  // Build a interference graph of all the buffers.
  auto buildGraph = [&]() {
    // Reset interference graph.
    graph.clear();
    for (auto *bufferA : buffers) {
      for (auto *bufferB : buffers) {
        if (bufferA == bufferB)
          continue;
        auto memoryA = allocation->getInterval(bufferA->id);
        auto memoryB = allocation->getInterval(bufferB->id);
        auto livenessA = bufferToLiveness.lookup(bufferA);
        auto livenessB = bufferToLiveness.lookup(bufferB);
        if (livenessA.intersects(livenessB) && memoryA.intersects(memoryB))
          graph[bufferA].insert(bufferB);
      }
    }
  };

  // Try to allocate shared memory while considering interference.
  auto tryToAllocate = [&]() {
    // Reset allocated size.
    allocation->allocatedSize = 0;
    // First-fit graph coloring.
    // Neighbors are nodes that interference with each other. We color a node by
    // finding the position of the first available non-neighboring node or the
    // first neighboring node without any color.
    DenseMap<Buffer *, int> bufferToColor;
    for (auto *buffer : buffers) {
      // color < 0 means uncolored
      bufferToColor[buffer] = buffer == buffers[0] ? 0 : -1;
    }
    SmallVector<bool> available(buffers.size());
    for (auto *bufferA : buffers) {
      std::fill(available.begin(), available.end(), true);
      for (auto *bufferB : graph.lookup(bufferA)) {
        auto color = bufferToColor[bufferB];
        if (color >= 0)
          available[color] = false;
      }
      auto it = std::find(available.begin(), available.end(), true);
      bufferToColor[bufferA] = std::distance(available.begin(), it);
    }
    // Perform allocation.
    // Example:
    //
    //   all alignments = 2
    //   color0: [0, 7), [0, 8), [0, 15) -> [0, 7), [0, 8), [0, 15)
    //   color1: [7, 9) -> [16, 16 + 9 - 7) -> [16, 18)
    //   color2: [8, 12) -> [18, 18 + 12 - 8) -> [18, 22)
    //
    for (auto *bufferA : buffers) {
      int64_t newOffset = 0;
      if (bufferToColor.lookup(bufferA) == 0) {
        bufferA->setOffsetAligned(newOffset);
      } else {
        for (auto *bufferB : graph.lookup(bufferA))
          newOffset = std::max(newOffset, bufferB->offset + bufferB->size);
        bufferA->setOffsetAligned(newOffset);
      }
      allocation->allocatedSize =
          std::max(allocation->allocatedSize, bufferA->offset + bufferA->size);
    }
  };

  // Keep reducing conflicts until reach a fixed point.
  buildGraph();
  do {
    tryToAllocate();
    buildGraph();
  } while (!graph.empty());
}
