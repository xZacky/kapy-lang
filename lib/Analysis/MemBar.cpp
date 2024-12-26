//===- MemBar.cpp -----------------------------------------------*- C++ -*-===//
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

#include "kapy/Analysis/MemBar.h"
#include "kapy/Analysis/Utils.h"
#include "kapy/Dialect/Kapy/IR/Kapy.h"
#include "kapy/Dialect/Kgpu/IR/Kgpu.h"

#include "mlir/Dialect/GPU/IR/GPUDialect.h"

using namespace mlir;
using namespace mlir::kapy;

void MemBarAnalysis::run(
    DenseMap<FunctionOpInterface, BlockInfo> &funcInfos) const {
  auto funcOp = cast<FunctionOpInterface>(allocation->getOperation());
  OpBuilder builder(funcOp.getContext());
  resolve(funcOp, funcInfos, builder);
}

void MemBarAnalysis::resolve(
    FunctionOpInterface funcOp,
    DenseMap<FunctionOpInterface, BlockInfo> &funcInfos,
    OpBuilder &builder) const {
  std::deque<Block *> list;
  funcOp.walk<WalkOrder::PreOrder>([&](Block *block) {
    for (auto &op : block->getOperations()) {
      // Check if the operation belongs to SCF dialect, if so, we need to throw
      // an error.
      if (fromDialect<scf::SCFDialect>(&op))
        llvm_unreachable("SCF dialect is not supported, please lower it to "
                         "ControlFlow dialect first");
    }
    if (block->isEntryBlock())
      list.push_back(block);
  });

  DenseMap<Block *, BlockInfo> succInfos;
  DenseMap<Block *, BlockInfo> thisInfos;
  // A fixed point algorithm.
  while (!list.empty()) {
    auto *block = list.front();
    list.pop_front();
    // Make a copy of the succInfo.
    auto copyInfo = succInfos[block];
    SmallVector<Block *> successors;
    for (auto &op : block->getOperations()) {
      if (op.hasTrait<OpTrait::IsTerminator>())
        visitTerminator(&op, successors);
      else
        update(&op, copyInfo, funcInfos, builder);
    }
    if (thisInfos.contains(block) && copyInfo == thisInfos[block]) {
      // If we have seen the block before and the copyInfo is the same as the
      // thisInfo, we skip it and its successors.
      continue;
    }
    // Update the current block.
    thisInfos[block].join(copyInfo);
    // Update the successors.
    for (auto *successor : successors) {
      succInfos[successor].join(thisInfos[block]);
      list.push_back(successor);
    }
  }
  // Update the final dangling buffers that haven't been synced.
  auto &funcInfo = funcInfos[funcOp];
  funcOp.walk<WalkOrder::PreOrder>([&](Block *block) {
    block->walk([&](ReturnOp returnOp) { funcInfo.join(thisInfos[block]); });
  });
}

void MemBarAnalysis::visitTerminator(
    Operation *op, SmallVectorImpl<Block *> &successors) const {
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

void MemBarAnalysis::insertBarrier(Operation *op, OpBuilder &builder) const {
  OpBuilder::InsertionGuard guard(builder);
  (void)builder.create<gpu::BarrierOp>(op->getLoc());
}

void MemBarAnalysis::update(Operation *op, BlockInfo &infoToUpdate,
                            DenseMap<FunctionOpInterface, BlockInfo> &funcInfos,
                            OpBuilder &builder) const {
  if (isa<gpu::BarrierOp>(op)) {
    // If the current operation is a BarrierOp, we sync previous reads and
    // writes.
    infoToUpdate.sync();
    return;
  }
  if (isa<AsyncWaitOp>(op) && !isa<gpu::BarrierOp>(op->getNextNode())) {
    // If the current operation is an AsyncWaitOp and the next operation is not
    // a BarrierOp we insert a BarrierOp and sync.
    builder.setInsertionPointAfter(op);
    insertBarrier(op, builder);
    infoToUpdate.sync();
    return;
  }

  BlockInfo info;
  if (isa<CallOp>(op)) {
    // Inter-function dependencies.
    auto callOp = dyn_cast<CallOpInterface>(op);
    auto *callable = callOp.resolveCallable();
    if (auto funcOp = dyn_cast_or_null<FunctionOpInterface>(callable))
      info = funcInfos.lookup(funcOp);
  } else {
    // Intra-function dependencies.
    if (auto effectOp = dyn_cast<MemoryEffectOpInterface>(op)) {
      // Explicit buffer read or write.
      SmallVector<SideEffects::EffectInstance<MemoryEffects::Effect>> effects;
      effectOp.getEffects(effects);
      for (auto &effect : effects) {
        if (auto value = effect.getValue()) {
          auto id = allocation->getBufferId(value);
          if (id != Allocation::InvalidId) {
            if (isa<MemoryEffects::Read>(effect.getEffect()))
              info.readIntervals.insert(allocation->getInterval(id));
            if (isa<MemoryEffects::Write>(effect.getEffect()))
              info.writeIntervals.insert(allocation->getInterval(id));
          }
        }
      }
    }
    // Scratch buffer read and write.
    auto id = allocation->getBufferId(op);
    if (id != Allocation::InvalidId) {
      info.readIntervals.insert(allocation->getInterval(id));
      info.writeIntervals.insert(allocation->getInterval(id));
    }
  }
  if (infoToUpdate.isIntersected(info)) {
    builder.setInsertionPoint(op);
    insertBarrier(op, builder);
    infoToUpdate.sync();
  }
  // Update the block info, even if barrier is inserted, we have to maintain the
  // current operation's read/write intervals.
  infoToUpdate.join(info);
}
