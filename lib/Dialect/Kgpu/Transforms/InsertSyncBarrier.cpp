//===- InsertSyncBarrier.h --------------------------------------*- C++ -*-===//
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
#include "kapy/Dialect/Kgpu/IR/Kgpu.h"
#include "kapy/Dialect/Kgpu/Transforms/Passes.h"
#include <deque>
#include <set>

using namespace mlir;
using namespace mlir::kapy;

namespace {

class BlockInfo {
public:
  BlockInfo() = default;

  BlockInfo &join(const BlockInfo &other) {
    readIntervals.insert(other.readIntervals.begin(),
                         other.readIntervals.end());
    writeIntervals.insert(other.writeIntervals.begin(),
                          other.writeIntervals.end());
    return *this;
  }

  bool isIntersected(const BlockInfo &other) const {
    return /*RAW*/ isIntersected(writeIntervals, other.readIntervals) ||
           /*WAR*/ isIntersected(readIntervals, other.writeIntervals) ||
           /*WAW*/ isIntersected(writeIntervals, other.writeIntervals);
  }

  void sync() {
    readIntervals.clear();
    writeIntervals.clear();
  }

  bool operator==(const BlockInfo &other) const {
    return readIntervals == other.readIntervals &&
           writeIntervals == other.writeIntervals;
  }
  bool operator!=(const BlockInfo &other) const { return !(*this == other); }

private:
  std::set<Interval<uint64_t>> readIntervals;
  std::set<Interval<uint64_t>> writeIntervals;

  bool isIntersected(const std::set<Interval<uint64_t>> &lhsSet,
                     const std::set<Interval<uint64_t>> &rhsSet) const {
    for (const auto &lhs : lhsSet)
      for (const auto &rhs : rhsSet)
        if (lhs.intersects(rhs))
          return true;
    return false;
  }

  friend class BlockAnalysis;
};

class BlockAnalysis {
public:
  explicit BlockAnalysis(AllocInfo *allocInfo) : allocInfo(allocInfo) {
    builder = std::make_unique<OpBuilder>(allocInfo->getFunction());
  }

  /// Run this analysis on the function, insert a barrier if necessary.
  void run(DenseMap<FunctionOpInterface, BlockInfo> &funcToInfo) const;

private:
  AllocInfo *allocInfo;
  std::unique_ptr<OpBuilder> builder;

  /// Visit an operation and update the given BlockInfo.
  void visit(Operation *op, BlockInfo &infoToUpdate,
             DenseMap<FunctionOpInterface, BlockInfo> &funcToInfo) const;

  /// Collect the successors of the terminator.
  void visit(Operation *op, SmallVectorImpl<Block *> &successors) const;

  void insertBarrier(Operation *op, bool after = false) const;
};

class ModuleBlockAnalysis : public CallGraph<BlockInfo> {
public:
  ModuleBlockAnalysis(ModuleAllocAnalysis *allocAnalysis)
      : CallGraph<BlockInfo>(allocAnalysis->getModule()),
        allocAnalysis(allocAnalysis) {}

  void run() {
    walk<WalkOrder::PreOrder, WalkOrder::PostOrder>(
        [](CallOpInterface caller, FunctionOpInterface callee) {},
        [&](FunctionOpInterface funcOp) {
          auto *allocInfo = allocAnalysis->getData(funcOp);
          auto [it, inserted] = funcToData.try_emplace(funcOp, BlockInfo());
          if (inserted) {
            BlockAnalysis blockAnalysis(allocInfo);
            blockAnalysis.run(funcToData);
          }
        });
  }

private:
  ModuleAllocAnalysis *allocAnalysis;
};

void BlockAnalysis::run(
    DenseMap<FunctionOpInterface, BlockInfo> &funcToInfo) const {
  auto funcOp = allocInfo->getFunction();
  std::deque<Block *> list;
  // Collect all entry blocks in this function.
  funcOp.walk<WalkOrder::PreOrder>([&](Block *block) {
    for (auto &op : block->getOperations()) {
      // Check if the operation belongs to SCF dialect, if so, we need to throw
      // an error.
      if (isa<scf::SCFDialect>(op.getDialect()))
        llvm_unreachable("SCFDialect is not supported, please lower it to "
                         "ControlFlowDialect first");
    }
    if (block->isEntryBlock())
      list.push_back(block);
  });

  DenseMap<Block *, BlockInfo> oldBlockToInfo;
  DenseMap<Block *, BlockInfo> newBlockToInfo;
  // A fixed point algorithm.
  while (!list.empty()) {
    auto *block = list.front();
    list.pop_front();
    // Make a copy of the old info.
    auto curInfo = oldBlockToInfo[block];
    SmallVector<Block *> successors;
    // Visit all the operations in this block.
    for (auto &op : block->getOperations())
      if (op.hasTrait<OpTrait::IsTerminator>())
        visit(&op, successors);
      else
        visit(&op, curInfo, funcToInfo);
    if (newBlockToInfo.contains(block) && curInfo == newBlockToInfo[block]) {
      // If we have seen the block before and the there are no update for this
      // block, we skip it and its successors.
      continue;
    }
    // Update the current block.
    newBlockToInfo[block].join(curInfo);
    // Update the successors.
    for (auto *successor : successors) {
      oldBlockToInfo[successor].join(newBlockToInfo[block]);
      list.push_back(successor);
    }
  }
  // Update the final dangling buffers that haven't been synced.
  auto &newInfo = funcToInfo[funcOp];
  funcOp.walk<WalkOrder::PreOrder>([&](Block *block) {
    if (block->getParentOp() == funcOp && isa<ReturnOp>(block->getTerminator()))
      newInfo.join(newBlockToInfo[block]);
  });
}

void BlockAnalysis::visit(Operation *op,
                          SmallVectorImpl<Block *> &successors) const {
  if (auto branchOp = dyn_cast<BranchOpInterface>(op)) {
    auto *block = branchOp->getBlock();
    successors.append(block->getSuccessors().begin(),
                      block->getSuccessors().end());
    return;
  }
  // Otherwise, it could be a ReturnOp.
  if (op->hasTrait<OpTrait::ReturnLike>())
    return;
  llvm_unreachable("unknown terminator encountered");
}

void BlockAnalysis::insertBarrier(Operation *op, bool after) const {
  if (after)
    builder->setInsertionPointAfter(op);
  else
    builder->setInsertionPoint(op);
  builder->create<Barrier0Op>(op->getLoc());
}

void BlockAnalysis::visit(
    Operation *op, BlockInfo &infoToUpdate,
    DenseMap<FunctionOpInterface, BlockInfo> &funcToInfo) const {
  if (isa<Barrier0Op>(op)) {
    // If the current operation is a Barrier0Op, we sync previous reads and
    // writes.
    infoToUpdate.sync();
    return;
  }
  if (isa<CpAsyncWaitGroupOp>(op) && !isa<Barrier0Op>(op->getNextNode())) {
    insertBarrier(op, true);
    infoToUpdate.sync();
    return;
  }

  BlockInfo info;
  if (isa<CallOp>(op)) {
    // Inter-function dependencies.
    auto callOp = dyn_cast<CallOpInterface>(op);
    auto *callable = callOp.resolveCallable();
    if (auto funcOp = dyn_cast_if_present<FunctionOpInterface>(callable))
      info = funcToInfo.lookup(funcOp);
  } else {
    // Intra-function dependencies.
    if (auto effectOp = dyn_cast<MemoryEffectOpInterface>(op)) {
      // Explicit buffer read or write.
      SmallVector<SideEffects::EffectInstance<MemoryEffects::Effect>> effects;
      effectOp.getEffects(effects);
      for (auto &effect : effects) {
        if (auto value = effect.getValue()) {
          auto id = allocInfo->getBufferId(value);
          if (id != AllocInfo::INVALID_ID) {
            if (isa<MemoryEffects::Read>(effect.getEffect()))
              info.readIntervals.insert(allocInfo->getInterval(id));
            if (isa<MemoryEffects::Write>(effect.getEffect()))
              info.writeIntervals.insert(allocInfo->getInterval(id));
          }
        }
      }
    } else {
      // Virtual buffer read and write.
      auto id = allocInfo->getBufferId(op);
      if (id != AllocInfo::INVALID_ID) {
        info.readIntervals.insert(allocInfo->getInterval(id));
        info.writeIntervals.insert(allocInfo->getInterval(id));
      }
    }
  }
  if (infoToUpdate.isIntersected(info)) {
    insertBarrier(op);
    infoToUpdate.sync();
  }
  // Update the BlockInfo, even if barrier is inserted, we have to maintain the
  // current operation's read/write intervals.
  infoToUpdate.join(info);
}

#define GEN_PASS_DEF_KGPUINSERTSYNCBARRIER
#include "kapy/Dialect/Kgpu/Transforms/Passes.h.inc"

class KgpuInsertSyncBarrierPass
    : public impl::KgpuInsertSyncBarrierBase<KgpuInsertSyncBarrierPass> {
public:
  virtual void runOnOperation() override {
    auto module = getOperation();
    ModuleAllocAnalysis allocAnalysis(module);
    ModuleBlockAnalysis blockAnalysis(&allocAnalysis);
    blockAnalysis.run();
  }
};

} // namespace

std::unique_ptr<Pass> kapy::createKgpuInsertSyncBarrierPass() {
  return std::make_unique<KgpuInsertSyncBarrierPass>();
}
