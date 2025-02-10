//===- AllocAnalysis.cpp ----------------------------------------*- C++ -*-===//
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
#include "kapy/Analysis/OpHelpers.h"
#include "kapy/Dialect/Kapy/IR/Kapy.h"
#include "kapy/Dialect/Kgpu/IR/Kgpu.h"
#include "kapy/Support/CommonUtils.h"
#include "mlir/Analysis/Liveness.h"

namespace mlir {
namespace kapy {

class AllocAnalysis {
public:
  AllocAnalysis(FunctionOpInterface funcOp, AllocInfo *info)
      : funcOp(funcOp), info(info) {}

  void run(DenseMap<FunctionOpInterface, AllocInfo> &funcToInfo) {
    addBuffers(funcToInfo);
    resolveLiveness();
    computeAndAllocate();
  }

private:
  using Buffer = AllocInfo::Buffer;
  using OpId = int;

  FunctionOpInterface funcOp;
  AllocInfo *info;
  llvm::MapVector<Buffer *, Interval<OpId>> bufferToLiveness;

  void addBuffers(DenseMap<FunctionOpInterface, AllocInfo> &funcToInfo);

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

void AllocInfo::run(DenseMap<FunctionOpInterface, AllocInfo> &funcToInfo) {
  AllocAnalysis(funcOp, this).run(funcToInfo);
}

void AllocAnalysis::addBuffers(
    DenseMap<FunctionOpInterface, AllocInfo> &funcToInfo) {
  funcOp->walk<WalkOrder::PreOrder>([&](Operation *op) {
    if (auto allocOp = dyn_cast<AllocSharedOp>(op)) {
      auto sharedType = allocOp.getType();
      auto bitWidth = getIntOrFloatBitWidth(sharedType);
      auto size = sharedType.getNumElements() * ceilDiv<unsigned>(bitWidth, 8);
      auto result = allocOp.getResult();
      info->addBuffer<Buffer::BufferKind::EXPLICIT>(result, size);
    } else if (auto reduceOp = dyn_cast<ReduceOp>(op)) {
      ReduceOpHelper helper(reduceOp);
      auto size = helper.getScratchSizeInBytes();
      if (size > 0)
        info->addBuffer<Buffer::BufferKind::SCRATCH>(op, size);
    } else if (auto changeOp = dyn_cast<ChangeOp>(op)) {
      ChangeOpHelper helper(changeOp);
      auto size = helper.getScratchSizeInBytes();
      if (size > 0)
        info->addBuffer<Buffer::BufferKind::SCRATCH>(op, size);
    } else if (auto callOp = dyn_cast<CallOpInterface>(op)) {
      auto *callable = callOp.resolveCallable();
      auto funcOp = dyn_cast_if_present<FunctionOpInterface>(callable);
      if (!funcOp)
        return;
      auto size = funcToInfo[funcOp].getAllocatedSize();
      if (size > 0)
        info->addBuffer<Buffer::BufferKind::VIRTUAL>(op, size);
    }
  });
}

void AllocAnalysis::resolveLiveness() {
  // Assign an id to each operation using post-order traversal.
  // To achieve the correct liveness, the parent operation's id should be larger
  // than each its child operation's id.
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
  // passed as a block argument. %6 should have an id larger than each its child
  // operation, otherwise %5's liveness ends before its child operations.
  DenseMap<Operation *, OpId> opToId;
  funcOp->walk([&](Operation *op) { opToId[op] = opToId.size(); });

  // Analyze liveness of explicit buffers.
  Liveness liveness(funcOp);
  auto getLiveness = [&](Value value) {
    auto ops = liveness.resolveLiveness(value);
    auto minId = std::numeric_limits<OpId>::max();
    auto maxId = std::numeric_limits<OpId>::min();
    std::for_each(ops.begin(), ops.end(), [&](Operation *op) {
      if (opToId[op] < minId)
        minId = opToId[op];
      if (opToId[op] + 1 > maxId)
        maxId = opToId[op];
    });
    return Interval(minId, maxId);
  };
  for (auto [value, buffer] : info->explicits)
    bufferToLiveness[buffer] = getLiveness(value);

  // Analyze liveness of scratch and virtual buffers.
  auto processBuffers = [&](const DenseMap<Operation *, Buffer *> &buffers) {
    for (auto [op, buffer] : buffers) {
      auto id = opToId.lookup(op);
      bufferToLiveness.insert({buffer, Interval(id, id + 1)});
    }
  };
  processBuffers(info->scratchs);
  processBuffers(info->virtuals);
}

void AllocAnalysis::computeAndAllocate() {
  SmallVector<Buffer *> buffers;
  for (auto *buffer : llvm::make_first_range(bufferToLiveness))
    buffers.push_back(buffer);

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
        auto livenessA = bufferToLiveness.lookup(bufferA);
        auto livenessB = bufferToLiveness.lookup(bufferB);
        auto memRangeA = info->getInterval(bufferA->id);
        auto memRangeB = info->getInterval(bufferB->id);
        if (livenessA.intersects(livenessB) && memRangeA.intersects(memRangeB))
          graph[bufferA].insert(bufferB);
      }
    }
  };

  // Try to allocate shared memory while considering interference.
  auto tryToAllocate = [&]() {
    // Reset allocated size.
    info->allocatedSize = 0;
    // First-fit graph coloring.
    // Neighbors are nodes that interference with each other. We color a node by
    // finding the position of the first available non-neighboring node or first
    // neighboring node without any color.
    DenseMap<Buffer *, int> bufferToColor;
    // color < 0 means uncolored
    for (auto *buffer : buffers)
      bufferToColor[buffer] = buffer == buffers[0] ? 0 : -1;
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
      uint64_t offset = 0;
      if (bufferToColor.lookup(bufferA) == 0) {
        bufferA->setOffsetAligned(offset);
      } else {
        for (auto *bufferB : graph.lookup(bufferA))
          offset = std::max(offset, bufferB->offset + bufferB->size);
        bufferA->setOffsetAligned(offset);
      }
      info->allocatedSize =
          std::max(info->allocatedSize, bufferA->offset + bufferA->size);
    }
  };

  // Keep reducing conflicts until reach a fixed point.
  buildGraph();
  do {
    tryToAllocate();
    buildGraph();
  } while (!graph.empty());
}
