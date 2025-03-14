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
#include "kapy/Analysis/AliasAnalysis.h"
#include "kapy/Analysis/AnalysisUtils.h"
#include "kapy/Dialect/Kapy/IR/Kapy.h"
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
    computeAndAlloc();
  }

private:
  using Buffer = AllocInfo::Buffer;
  using OpId = int64_t;

  FunctionOpInterface funcOp;
  AllocInfo *info = nullptr;
  llvm::MapVector<Buffer *, Interval<OpId>> bufferToLiveness;

  void addBuffers(DenseMap<FunctionOpInterface, AllocInfo> &funcToInfo);

  void addAlias(Value value, AliasAnalysis &analysis);

  void resolveLiveness();

  // Compute the shared memory offsets and allocate for all the related buffers
  // while considering interference.
  // Paper: Algorithms for Compile-time Memory Optimization
  // https://dl.acm.org/doi/pdf/10.5555/314500.315082
  void computeAndAlloc();
};

} // namespace kapy
} // namespace mlir

using namespace mlir;
using namespace mlir::kapy;

void AllocInfo::run(DenseMap<FunctionOpInterface, AllocInfo> &funcToInfo) {
  AllocAnalysis(funcOp, this).run(funcToInfo);
}

static uint64_t getSizeInBytes(int64_t numElems, unsigned bitWidth) {
  return numElems * bitWidth / 8;
}

void AllocAnalysis::addBuffers(
    DenseMap<FunctionOpInterface, AllocInfo> &funcToInfo) {
  funcOp->walk<WalkOrder::PreOrder>([&](Operation *op) {
    if (auto mkSharedOp = dyn_cast<MkSharedOp>(op)) {
      auto sharedType = mkSharedOp.getType();
      auto numElems = sharedType.getNumElements();
      auto bitWidth = getIntOrFloatBitWidth(sharedType);
      auto size = getSizeInBytes(numElems, bitWidth);
      auto result = mkSharedOp.getResult();
      info->addBuffer<Buffer::BufferKind::EXPLICIT>(result, size);
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
  auto solver = createDataFlowSolver();
  auto *analysis = solver->load<AliasAnalysis>();
  funcOp->walk([&](Operation *op) {
    if (op->hasTrait<OpTrait::IsIsolatedFromAbove>())
      if (failed(solver->initializeAndRun(op)))
        llvm_unreachable("failed to run alias analysis");
  });
  funcOp->walk<WalkOrder::PreOrder>([&](Operation *op) {
    for (auto operand : op->getOperands())
      addAlias(operand, *analysis);
    for (auto result : op->getResults())
      addAlias(result, *analysis);
  });
}

void AllocAnalysis::addAlias(Value value, AliasAnalysis &analysis) {
  auto aliasSet = analysis.getLatticeElement(value)->getValue().getAliasSet();
  if (!aliasSet.empty())
    for (auto aliased : aliasSet)
      info->addAlias(value, aliased);
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

  Liveness funcLiveness(funcOp);
  auto getLiveness = [&](Value value) {
    auto ops = funcLiveness.resolveLiveness(value);
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

  // Analyze liveness of explicit buffers.
  for (auto [value, buffer] : info->explicits)
    bufferToLiveness[buffer] = getLiveness(value);

  // Analyze liveness of aliased buffers.
  for (auto [value, buffers] : info->aliasSets) {
    auto liveness = getLiveness(value);
    for (auto *buffer : buffers) {
      auto minId = liveness.start();
      auto maxId = liveness.end();
      if (bufferToLiveness.contains(buffer)) {
        minId = std::min(minId, bufferToLiveness[buffer].start());
        maxId = std::max(maxId, bufferToLiveness[buffer].end());
      }
      bufferToLiveness[buffer] = Interval(minId, maxId);
    }
  }

  // Analyze liveness of virtual buffers.
  for (auto [op, buffer] : info->virtuals) {
    auto id = opToId.lookup(op);
    bufferToLiveness.insert({buffer, Interval(id, id + 1)});
  }
}

void AllocAnalysis::computeAndAlloc() {
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
    for (auto *buffer0 : buffers) {
      for (auto *buffer1 : buffers) {
        if (buffer0 == buffer1)
          continue;
        auto liveness0 = bufferToLiveness.lookup(buffer0);
        auto liveness1 = bufferToLiveness.lookup(buffer1);
        auto memRange0 = info->getInterval(buffer0->id);
        auto memRange1 = info->getInterval(buffer1->id);
        if (liveness0.intersects(liveness1) && memRange0.intersects(memRange1))
          graph[buffer0].insert(buffer1);
      }
    }
  };

  // Try to allocate shared memory while considering interference.
  auto tryToAlloc = [&]() {
    // Reset allocated size.
    info->allocatedSize = 0;
    // First-fit graph coloring.
    // Neighbors are nodes that interference with each other. We color a node by
    // finding the position of the first available non-neighboring node or first
    // neighboring node without any color.
    DenseMap<Buffer *, int> bufferToColor;
    for (auto *buffer : buffers)
      bufferToColor[buffer] = -1; // color < 0 means uncolored
    if (buffers.size() > 0)
      bufferToColor[buffers[0]] = 0;
    SmallVector<bool> available(buffers.size());
    for (auto *buffer0 : buffers) {
      std::fill(available.begin(), available.end(), true);
      for (auto *buffer1 : graph.lookup(buffer0)) {
        auto color = bufferToColor[buffer1];
        if (color >= 0)
          available[color] = false;
      }
      auto it = std::find(available.begin(), available.end(), true);
      bufferToColor[buffer0] = std::distance(available.begin(), it);
    }
    // Perform allocation.
    // Example:
    //
    //   all alignments = 2
    //   color0: [0, 7), [0, 8), [0, 15) -> [0, 7), [0, 8), [0, 15)
    //   color1: [7, 9) -> [16, 16 + 9 - 7) -> [16, 18)
    //   color2: [8, 12) -> [18, 18 + 12 - 8) -> [18, 22)
    //
    for (auto *buffer0 : buffers) {
      uint64_t offset = 0;
      if (bufferToColor.lookup(buffer0) == 0) {
        buffer0->setOffsetAligned(offset);
      } else {
        for (auto *buffer1 : graph.lookup(buffer0))
          offset = std::max(offset, buffer1->offset + buffer1->size);
        buffer0->setOffsetAligned(offset);
      }
      info->allocatedSize =
          std::max(info->allocatedSize, buffer0->offset + buffer0->size);
    }
  };

  // Keep reducing conflicts until reach a fixed point.
  buildGraph();
  do {
    tryToAlloc();
    buildGraph();
  } while (!graph.empty());
}
