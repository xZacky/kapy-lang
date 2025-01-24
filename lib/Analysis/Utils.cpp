//===- Utils.cpp ------------------------------------------------*- C++ -*-===//
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

#include "kapy/Analysis/Utils.h"
#include "mlir/Analysis/DataFlow/ConstantPropagationAnalysis.h"
#include "mlir/Analysis/DataFlow/DeadCodeAnalysis.h"
#include "mlir/IR/Matchers.h"

using namespace mlir;
using namespace mlir::dataflow;

namespace {

/// A structure similar to SetVector but maintains a deque instead of a vector
/// to allow for efficient push_back and pop_front operations.
/// Using SetVector doesn't suffice our needs because it only pushes and pops
/// from back.
///
/// For example, if we have a queue like this:
///
///   0->4  1->2->3
///      ^--------'
///
/// where 4 depends on 3, once we pop 3, we found 4 is not ready, so we check 2
/// and push 3 back to the queue.
struct SetQueue {
  DenseSet<Operation *> set;
  std::deque<Operation *> queue;

  SetQueue() : set(), queue() {}

  bool push_back(Operation *op) {
    if (set.insert(op).second) {
      queue.push_back(op);
      return true;
    }
    return false;
  }

  Operation *pop_front() {
    auto *op = queue.front();
    queue.pop_front();
    set.erase(op);
    return op;
  }

  bool empty() { return queue.empty(); }
};

/// DFS post-order implementation that maintains a global state to work across
/// multiple invocations, to help implement topological sort on multi-root DAG.
/// We traverse all the operations but only add those appear in `ops` to the
/// final result.
struct DfsState {
  const SetVector<Operation *> &ops;
  DenseSet<Operation *> processedOps;
  SetVector<Operation *> sortedOps;

  DfsState(const SetVector<Operation *> &ops)
      : ops(ops), processedOps(), sortedOps() {}

  /// We mark each operation as ready if all its operands and parent operations
  /// are processed.
  /// Otherwise, we keep adding its operands to the SetQueue. We always want an
  /// operation to be scheduled after all its parents to handle correctly cases
  /// with SCF operations.
  void tryToMarkReady(Operation *op, SetQueue &list,
                      SmallVectorImpl<Operation *> &readyOps) {
    bool isReady = true;
    for (auto operand : op->getOperands()) {
      auto *defOp = operand.getDefiningOp();
      if (defOp && !processedOps.contains(defOp)) {
        list.push_back(defOp);
        isReady = false;
      }
    }
    auto *parentOp = op->getParentOp();
    while (parentOp) {
      if (!processedOps.contains(parentOp)) {
        list.push_back(parentOp);
        isReady = false;
      }
      parentOp = parentOp->getParentOp();
    }
    if (isReady)
      readyOps.push_back(op);
  }
};

} // namespace

static void postOrderDfs(Operation *rootOp, DfsState &state) {
  SetQueue list;
  list.push_back(rootOp);
  while (!list.empty()) {
    // Ready operations are ready to be processed, meaning that either their
    // operands are all processed or they have no operands.
    SmallVector<Operation *> readyOps;
    auto *op = list.pop_front();
    state.tryToMarkReady(op, list, readyOps);
    while (!readyOps.empty()) {
      auto *readyOp = readyOps.pop_back_val();
      // Process a ready operation.
      if (!state.processedOps.insert(readyOp).second)
        continue;
      // If we want to sort it, add it to sorted operations.
      if (state.ops.contains(readyOp))
        state.sortedOps.insert(readyOp);
      // Now it is processed. Try to mark all its users and child operations as
      // ready.
      for (auto result : readyOp->getResults())
        for (auto *useOp : result.getUsers())
          state.tryToMarkReady(useOp, list, readyOps);
      for (auto &region : readyOp->getRegions())
        for (auto &childOp : region.getOps())
          state.tryToMarkReady(&childOp, list, readyOps);
    }
  }
}

SetVector<Operation *>
kapy::multiRootTopoSort(const SetVector<Operation *> &ops) {
  if (ops.empty())
    return ops;
  // Run from each root with global state.
  DfsState state(ops);
  for (auto *op : ops)
    postOrderDfs(op, state);
  return state.sortedOps;
}

SetVector<Operation *> kapy::multiRootGetSlice(Operation *op,
                                               TransitiveFilter bwFilter,
                                               TransitiveFilter fwFilter) {
  SetVector<Operation *> slice;
  slice.insert(op);
  unsigned i = 0;
  SetVector<Operation *> bwSlice, fwSlice;
  while (i != slice.size()) {
    auto *opI = slice[i];

    bwSlice.clear();
    BackwardSliceOptions options;
    options.omitBlockArguments = true;
    options.filter = bwFilter;
    getBackwardSlice(opI, &bwSlice, options);
    slice.insert(bwSlice.begin(), bwSlice.end());

    fwSlice.clear();
    getForwardSlice(opI, &fwSlice, fwFilter);
    slice.insert(fwSlice.begin(), fwSlice.end());

    ++i;
  }
  return multiRootTopoSort(slice);
}

bool kapy::hasRestrictedPath(Operation *srcOp, Operation *dstOp,
                             const SetVector<Operation *> &slice,
                             function_ref<bool(Operation *)> filter) {
  SetVector<Operation *> list;
  DenseSet<Operation *> seen;
  list.insert(srcOp);
  while (!list.empty()) {
    auto *op = list.pop_back_val();
    for (auto result : op->getResults()) {
      for (auto *useOp : result.getUsers()) {
        if (!seen.insert(useOp).second)
          continue;
        if (useOp == dstOp)
          return true;
        if (slice.contains(useOp) && filter(useOp))
          list.insert(useOp);
      }
    }
  }
  return false;
}

namespace {

/// Copied from TestDeadCodeAnalysis.cpp, because some dead code analysis
/// interacts with constant propagation, but SparseConstantPropagation doesn't
/// seem to be sufficient.
class ConstantAnalysis : public DataFlowAnalysis {
public:
  using DataFlowAnalysis::DataFlowAnalysis;

  virtual LogicalResult initialize(Operation *srcOp) override {
    auto result = srcOp->walk([&](Operation *op) -> WalkResult {
      ProgramPoint point(op);
      if (failed(visit(point)))
        return WalkResult::interrupt();
      return WalkResult::advance();
    });
    return success(!result.wasInterrupted());
  }

  virtual LogicalResult visit(ProgramPoint point) override {
    auto *op = point.get<Operation *>();
    Attribute value;
    if (matchPattern(op, m_Constant(&value))) {
      auto *lattice = getOrCreate<Lattice<ConstantValue>>(op->getResult(0));
      propagateIfChanged(lattice,
                         lattice->join(ConstantValue(value, op->getDialect())));
      return success();
    }
    // Dead code analysis requires every operands has initialized ConstantValue
    // state before it is visited.
    // That's why we need to set all operands to unknown constants.
    setAllToUnknownConstants(op->getResults());
    for (auto &region : op->getRegions())
      for (auto &block : region.getBlocks())
        setAllToUnknownConstants(block.getArguments());
    return success();
  }

private:
  void setAllToUnknownConstants(ValueRange values) {
    ConstantValue unknown(nullptr, nullptr);
    for (auto value : values) {
      auto *lattice = getOrCreate<Lattice<ConstantValue>>(value);
      propagateIfChanged(lattice, lattice->join(unknown));
    }
  }
};

} // namespace

std::unique_ptr<DataFlowSolver> kapy::createDataFlowSolver() {
  auto solver = std::make_unique<DataFlowSolver>();
  solver->load<DeadCodeAnalysis>();
  solver->load<ConstantAnalysis>();
  return solver;
}
