//===- TransformUtils.h -----------------------------------------*- C++ -*-===//
//
// This file implements functions for transformation.
//
//===----------------------------------------------------------------------===//

#include "kapy/Dialect/Kapy/Transforms/TransformUtils.h"
#include "kapy/Dialect/Kapy/IR/Kapy.h"

using namespace mlir;
using namespace mlir::kapy;

void kapy::propagateMemoryLayout(Value value, Attribute layout,
                                 DenseSet<Value> &seen) {
  if (!seen.insert(value).second)
    return;

  auto tensorType = cast<RankedTensorType>(value.getType());
  value.setType(cloneWithLayout(tensorType, layout));

  // Forward propagation.
  for (auto &use : value.getUses()) {
    auto *user = use.getOwner();
    auto index = use.getOperandNumber();
    if (auto forOp = dyn_cast<scf::ForOp>(user)) {
      auto iterArg = forOp.getTiedLoopRegionIterArg(&use);
      propagateMemoryLayout(iterArg, layout, seen);
      continue;
    }
    if (auto whileOp = dyn_cast<scf::WhileOp>(user)) {
      auto beforeArg = whileOp.getBeforeArguments()[index];
      propagateMemoryLayout(beforeArg, layout, seen);
      continue;
    }
    if (auto yieldOp = dyn_cast<scf::YieldOp>(user)) {
      auto *parentOp = yieldOp->getParentOp();
      if (auto forOp = dyn_cast<scf::ForOp>(parentOp)) {
        auto iterArg = forOp.getRegionIterArg(index);
        propagateMemoryLayout(iterArg, layout, seen);
        auto result = forOp->getResult(index);
        propagateMemoryLayout(result, layout, seen);
        continue;
      }
      if (auto whileOp = dyn_cast<scf::WhileOp>(parentOp)) {
        auto beforeArg = whileOp.getBeforeArguments()[index];
        propagateMemoryLayout(beforeArg, layout, seen);
        continue;
      }
      if (auto ifOp = dyn_cast<scf::IfOp>(parentOp)) {
        auto result = ifOp.getResult(index);
        propagateMemoryLayout(result, layout, seen);
        continue;
      }
    }
    if (auto conditionOp = dyn_cast<scf::ConditionOp>(user)) {
      auto whileOp = conditionOp.getParentOp();
      auto afterArg = whileOp.getAfterArguments()[index - 1];
      propagateMemoryLayout(afterArg, layout, seen);
      auto result = whileOp->getResult(index - 1);
      propagateMemoryLayout(result, layout, seen);
      continue;
    }
    if (auto selectOp = dyn_cast<arith::SelectOp>(user)) {
      propagateMemoryLayout(selectOp.getResult(), layout, seen);
      continue;
    }
    if (auto svGlobalOp = dyn_cast<SvGlobalOp>(user)) {
      propagateMemoryLayout(svGlobalOp.getResult(), layout, seen);
      continue;
    }
    if (auto svSharedOp = dyn_cast<SvSharedOp>(user)) {
      propagateMemoryLayout(svSharedOp.getResult(), layout, seen);
      continue;
    }
  }

  // Backward propagation.
  if (auto forOp = value.getDefiningOp<scf::ForOp>()) {
    auto *yieldOp = forOp.getBody()->getTerminator();
    auto index = cast<OpResult>(value).getResultNumber();
    auto yielded = yieldOp->getOperand(index);
    propagateMemoryLayout(yielded, layout, seen);
    return;
  }
  if (auto whileOp = value.getDefiningOp<scf::WhileOp>()) {
    auto *conditionOp = whileOp.getBeforeBody()->getTerminator();
    auto index = cast<OpResult>(value).getResultNumber();
    auto yielded = conditionOp->getOperand(index + 1);
    propagateMemoryLayout(yielded, layout, seen);
    return;
  }
  if (auto ifOp = value.getDefiningOp<scf::IfOp>()) {
    auto index = cast<OpResult>(value).getResultNumber();
    propagateMemoryLayout(ifOp.thenYield().getOperand(index), layout, seen);
    propagateMemoryLayout(ifOp.elseYield().getOperand(index), layout, seen);
    return;
  }
  if (auto selectOp = value.getDefiningOp<arith::SelectOp>()) {
    propagateMemoryLayout(selectOp.getTrueValue(), layout, seen);
    propagateMemoryLayout(selectOp.getFalseValue(), layout, seen);
    return;
  }
  if (auto svGlobalOp = value.getDefiningOp<SvGlobalOp>()) {
    propagateMemoryLayout(svGlobalOp.getSource(), layout, seen);
    return;
  }
  if (auto svSharedOp = value.getDefiningOp<SvSharedOp>()) {
    propagateMemoryLayout(svSharedOp.getSource(), layout, seen);
    return;
  }
  if (auto blockArg = dyn_cast<BlockArgument>(value)) {
    auto index = blockArg.getArgNumber();
    auto *block = blockArg.getOwner();
    auto *parentOp = block->getParentOp();
    if (auto forOp = dyn_cast<scf::ForOp>(parentOp)) {
      auto init = forOp.getTiedLoopInit(blockArg)->get();
      propagateMemoryLayout(init, layout, seen);
      auto yielded = forOp.getTiedLoopYieldedValue(blockArg)->get();
      propagateMemoryLayout(yielded, layout, seen);
      return;
    }
    if (auto whileOp = dyn_cast<scf::WhileOp>(parentOp)) {
      if (block == whileOp.getBeforeBody()) {
        auto *yieldOp = whileOp.getAfterBody()->getTerminator();
        auto init = whileOp->getOperand(index);
        propagateMemoryLayout(init, layout, seen);
        auto yielded = yieldOp->getOperand(index);
        propagateMemoryLayout(yielded, layout, seen);
        return;
      }
      if (block == whileOp.getAfterBody()) {
        auto *conditionOp = whileOp.getBeforeBody()->getTerminator();
        auto yielded = conditionOp->getOperand(index + 1);
        propagateMemoryLayout(yielded, layout, seen);
        return;
      }
    }
  }
}
