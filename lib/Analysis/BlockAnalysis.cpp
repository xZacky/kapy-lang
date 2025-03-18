//===- BlockAnalysis.cpp ----------------------------------------*- C++ -*-===//
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

#include "kapy/Analysis/BlockAnalysis.h"
#include "kapy/Dialect/Kapy/IR/Kapy.h"
#include "mlir/Dialect/LLVMIR/NVVMDialect.h"
#include <deque>

using namespace mlir;
using namespace mlir::kapy;

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
        llvm_unreachable("scf dialect is not supported, please lower it to "
                         "control flow dialect first");
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
    auto info = oldBlockToInfo[block];
    SmallVector<Block *> successors;
    // Visit all the operations in this block.
    for (auto &op : block->getOperations())
      if (op.hasTrait<OpTrait::IsTerminator>())
        visit(&op, successors);
      else
        visit(&op, info, funcToInfo);
    if (newBlockToInfo.contains(block) && info == newBlockToInfo[block]) {
      // If we have seen the block before and the there are no update for this
      // block, we skip it and its successors.
      continue;
    }
    // Update the current block.
    newBlockToInfo[block].join(info);
    // Update the successors.
    for (auto *successor : successors) {
      oldBlockToInfo[successor].join(newBlockToInfo[block]);
      list.push_back(successor);
    }
  }
  // Update the final dangling buffers that haven't been synced.
  auto &info = funcToInfo[funcOp];
  funcOp.walk<WalkOrder::PreOrder>([&](Block *block) {
    if (block->getParentOp() == funcOp && isa<ReturnOp>(block->getTerminator()))
      info.join(newBlockToInfo[block]);
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
  builder->create<NVVM::Barrier0Op>(op->getLoc());
}

void BlockAnalysis::visit(
    Operation *op, BlockInfo &infoToUpdate,
    DenseMap<FunctionOpInterface, BlockInfo> &funcToInfo) const {
  if (isa<NVVM::Barrier0Op>(op)) {
    // If the current operation is a Barrier0Op, we sync previous memory reads
    // and writes.
    infoToUpdate.sync();
    return;
  }
  if (isa<CpAsyncWaitGroupOp>(op) &&
      !isa<NVVM::Barrier0Op>(op->getNextNode())) {
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
    if (auto effectOp = dyn_cast<MemoryEffectOpInterface>(op)) {
      // Intra-function dependencies. Explicit buffer read or write.
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
      // Inter-function dependencies. Virtual buffer read and write.
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
