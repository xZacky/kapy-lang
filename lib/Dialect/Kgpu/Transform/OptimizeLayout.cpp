//===- OptimizeLayout.h -----------------------------------------*- C++ -*-===//
//
// This file implements the KgpuOptimizeLayoutPass.
//
//===----------------------------------------------------------------------===//

#include "kapy/Analysis/Layout.h"
#include "kapy/Dialect/Kapy/IR/Kapy.h"
#include "kapy/Dialect/Kgpu/IR/Kgpu.h"
#include "kapy/Dialect/Kgpu/Transform/Utils.h"

using namespace mlir;
using namespace mlir::kapy;

namespace {

/// The current algorithm works by analyzing the IR and doing a one-shot rewrite
/// based on the analysis. The algorithm is as follows:
/// 1. Find all the anchor operations. These are operations with several layouts
///    that we want to choose one of them and preserve.
/// 2. For each anchor operation, propagate its layouts to other operations, at
///    this stage a tensor may have multiple layouts associated with it.
/// 3. Resolve conflicts by deciding which one of the layouts each tensor should
///    keep, inserting changes to resolve conflicts.
/// 4. Rewrite the IR by walking the function in dominance order.
class LayoutPropagation {
public:
  LayoutPropagation(FuncOp funcOp) : funcOp(funcOp) {}

  /// Find the anchor operations and set their layouts.
  void initialize();

  /// Recursively propagate the layouts from anchor operations until we reach a
  /// fixed point.
  void propagate();

  /// Resolve cases where a tensor has multiple layouts associated to it.
  void resolve();

  /// Rewrite the IR for a function.
  void rewrite();

private:
  FuncOp funcOp;
  llvm::MapVector<Value, SetVector<Attribute>> valueLayouts;
  DenseMap<std::pair<Value, Attribute>, Value> valueMapping;
  SetVector<Operation *> opsToDelete;

  /// Propagate layouts from `value`, return all the changed values.
  SmallVector<Value> propagateToUses(Value value,
                                     const SetVector<Attribute> &layouts);
  SmallVector<Value> propagateToDefs(Value value,
                                     const SetVector<Attribute> &layouts);

  /// Propagate the layouts to all the values and fill out them with new layout
  /// into `changedValues`.
  void propagateToUses(Operation *op, ValueRange values,
                       const SetVector<Attribute> &layouts,
                       SmallVectorImpl<Value> &changedValues);
  void propagateToDefs(Operation *op, ValueRange values,
                       const SetVector<Attribute> &layouts,
                       SmallVectorImpl<Value> &changedValues);

  void rewrite(Region *region);
  Operation *rewrite(Operation *op);

  Operation *rewrite(OpBuilder &builder, Operation *op, Attribute newLayout);
  Operation *rewrite(scf::ForOp op);
  Operation *rewrite(scf::WhileOp op);
  Operation *rewrite(scf::IfOp op);
  void rewrite(scf::YieldOp op);
  void rewrite(scf::ConditionOp op);
  void rewrite(ReduceOp op);

  /// Map the original value to the rewritten one.
  void map(Value value, Value newValue);
  void replaceOrMap(ValueRange values, ValueRange newValues);

  /// Return the mapped value in the given layout. This will insert change if
  /// the new layout is different than the layout decided at resolve time.
  Value getWith(Value value, Attribute newLayout);
};

} // namespace

void LayoutPropagation::initialize() {
  auto numWarps = getNumWarps(funcOp->getParentOfType<ModuleOp>());
  auto addAnchor = [&](Value value, Operation *op = nullptr) {
    if (auto tensorType = dyn_cast<RankedTensorType>(value.getType())) {
      SetVector<Attribute> layouts;
      if (op)
        layouts = getCandidateLayouts(op, numWarps);
      layouts.insert(tensorType.getEncoding());
      valueLayouts.insert({value, layouts});
    }
  };

  // Consider function arguments as anchors. This makes it easier to write test.
  for (auto funcArg : funcOp.getArguments())
    addAnchor(funcArg);

  funcOp.walk([&](Operation *op) {
    if (isExpensiveMemoryRead(op) || isa<MatmulOp>(op))
      for (auto result : op->getResults())
        addAnchor(result, op);
    if (isExpensiveMemoryWrite(op))
      for (auto operand : op->getOperands())
        addAnchor(operand, op);
  });
}

void LayoutPropagation::propagate() {
  SmallVector<Value> list;
  // Forward propagation.
  for (auto value : llvm::make_first_range(valueLayouts))
    list.push_back(value);
  while (!list.empty()) {
    auto value = list.pop_back_val();
    const auto &layouts = valueLayouts[value];
    auto changedValues = propagateToUses(value, layouts);
    list.insert(list.end(), changedValues.begin(), changedValues.end());
  }
  // Backward propagation.
  for (auto value : llvm::make_first_range(valueLayouts)) {
    for (auto *useOp : value.getUsers()) {
      if (isExpensiveMemoryWrite(useOp)) {
        list.push_back(value);
        break;
      }
    }
  }
  while (!list.empty()) {
    auto value = list.pop_back_val();
    const auto &layouts = valueLayouts[value];
    auto changedValues = propagateToDefs(value, layouts);
    list.insert(list.end(), changedValues.begin(), changedValues.end());
  }
}

SmallVector<Value>
LayoutPropagation::propagateToUses(Value value,
                                   const SetVector<Attribute> &layouts) {
  SmallVector<Value> changedValues;
  for (auto &use : value.getUses()) {
    auto *useOp = use.getOwner();
    auto useIndex = use.getOperandNumber();
    if (auto forOp = dyn_cast<scf::ForOp>(useOp)) {
      auto iterArg = forOp.getTiedLoopRegionIterArg(&use);
      auto result = forOp.getTiedLoopResult(&use);
      propagateToUses(useOp, {iterArg, result}, layouts, changedValues);
      continue;
    }
    if (auto whileOp = dyn_cast<scf::WhileOp>(useOp)) {
      auto beforeArg = whileOp.getBeforeArguments()[useIndex];
      auto result = whileOp.getResult(useIndex);
      propagateToUses(useOp, {beforeArg, result}, layouts, changedValues);
      continue;
    }
    if (auto yieldOp = dyn_cast<scf::YieldOp>(useOp)) {
      auto *parentOp = yieldOp->getParentOp();
      SmallVector<Value> values;
      if (isa<scf::ForOp, scf::IfOp>(parentOp))
        values.push_back(parentOp->getResult(useIndex));
      if (auto forOp = dyn_cast<scf::ForOp>(parentOp))
        values.push_back(forOp.getRegionIterArg(useIndex));
      if (auto whileOp = dyn_cast<scf::WhileOp>(parentOp)) {
        values.push_back(whileOp.getBeforeArguments()[useIndex]);
        values.push_back(whileOp->getOperand(useIndex));
      }
      propagateToUses(useOp, values, layouts, changedValues);
      continue;
    }
    if (auto conditionOp = dyn_cast<scf::ConditionOp>(useOp)) {
      auto whileOp = cast<scf::WhileOp>(conditionOp->getParentOp());
      // Skip first operand as it is the condition.
      auto argIndex = useIndex - 1;
      auto afterArg = whileOp.getAfterArguments()[argIndex];
      auto result = whileOp->getResult(argIndex);
      propagateToUses(useOp, {afterArg, result}, layouts, changedValues);
      continue;
    }
    if (useOp->hasTrait<OpTrait::SameOperandsAndResultLayout>() ||
        useOp->hasTrait<OpTrait::Elementwise>() ||
        isa<ReduceOp, UnsqueezeOp, ChangeOp>(useOp)) {
      propagateToUses(useOp, useOp->getResults(), layouts, changedValues);
      continue;
    }
  }
  return changedValues;
}

void LayoutPropagation::propagateToUses(Operation *op, ValueRange values,
                                        const SetVector<Attribute> &layouts,
                                        SmallVectorImpl<Value> &changedValues) {
  for (auto value : values) {
    if (!isa<RankedTensorType>(value.getType()))
      continue;
    bool changed = false;
    for (auto layout : layouts) {
      Attribute newLayout;
      if (auto changeOp = dyn_cast<ChangeOp>(op)) {
        if (auto mmopdLayout =
                dyn_cast<MmOperandLayoutAttr>(changeOp.getType().getEncoding()))
          newLayout = mmopdLayout;
        // Try to remove the change by making the result layout same as operand.
        else
          newLayout = layout;
      } else {
        newLayout = inferUseLayout(op, layout);
      }
      if (newLayout)
        changed |= valueLayouts[value].insert(newLayout);
    }
    if (changed)
      changedValues.push_back(value);
  }
}

SmallVector<Value>
LayoutPropagation::propagateToDefs(Value value,
                                   const SetVector<Attribute> &layouts) {
  SmallVector<Value> changedValues;
  if (auto forOp = value.getDefiningOp<scf::ForOp>()) {
    auto *yieldOp = forOp.getBody()->getTerminator();
    auto result = cast<OpResult>(value);
    auto init = forOp.getTiedLoopInit(result)->get();
    auto operand = yieldOp->getOperand(result.getResultNumber());
    propagateToDefs(forOp, {init, operand}, layouts, changedValues);
    return changedValues;
  }
  if (auto whileOp = value.getDefiningOp<scf::WhileOp>()) {
    auto yieldOp = whileOp.getAfterBody()->getTerminator();
    auto result = cast<OpResult>(value);
    auto init = whileOp.getTiedLoopInit(result)->get();
    auto operand = yieldOp->getOperand(result.getResultNumber());
    propagateToDefs(whileOp, {init, operand}, layouts, changedValues);
    return changedValues;
  }
  if (auto ifOp = value.getDefiningOp<scf::IfOp>()) {
    auto index = cast<OpResult>(value).getResultNumber();
    auto thenOperand = ifOp.thenYield().getOperand(index);
    auto elseOperand = ifOp.elseYield().getOperand(index);
    propagateToDefs(ifOp, {thenOperand, elseOperand}, layouts, changedValues);
    return changedValues;
  }
  if (auto *defOp = value.getDefiningOp()) {
    if (defOp->hasTrait<OpTrait::SameOperandsAndResultLayout>() ||
        defOp->hasTrait<OpTrait::Elementwise>() ||
        isa<ReduceOp, UnsqueezeOp, ChangeOp>(defOp)) {
      propagateToDefs(defOp, defOp->getOperands(), layouts, changedValues);
      return changedValues;
    }
  }
  if (auto blockArg = dyn_cast<BlockArgument>(value)) {
    auto *region = blockArg.getOwner()->getParent();
    auto *parentOp = region->getParentOp();
    if (auto forOp = dyn_cast<scf::ForOp>(parentOp)) {
      auto *yieldOp = forOp.getBody()->getTerminator();
      auto init = forOp.getTiedLoopInit(blockArg)->get();
      auto operand = yieldOp->getOperand(blockArg.getArgNumber() - 1);
      propagateToDefs(forOp, {init, operand}, layouts, changedValues);
      return changedValues;
    }
    if (auto whileOp = dyn_cast<scf::WhileOp>(parentOp)) {
      if (region == &whileOp.getBefore()) {
        auto *yieldOp = whileOp.getAfterBody()->getTerminator();
        auto init = whileOp.getTiedLoopInit(blockArg)->get();
        auto operand = yieldOp->getOperand(blockArg.getArgNumber());
        propagateToDefs(whileOp, {init, operand}, layouts, changedValues);
        return changedValues;
      }
      if (region == &whileOp.getAfter()) {
        auto *conditionOp = whileOp.getBeforeBody()->getTerminator();
        auto operand = conditionOp->getOperand(blockArg.getArgNumber() + 1);
        propagateToDefs(whileOp, operand, layouts, changedValues);
        return changedValues;
      }
    }
  }
  return changedValues;
}

void LayoutPropagation::propagateToDefs(Operation *op, ValueRange values,
                                        const SetVector<Attribute> &layouts,
                                        SmallVectorImpl<Value> &changedValues) {
  for (auto value : values) {
    if (!isa<RankedTensorType>(value.getType()))
      continue;
    bool changed = false;
    for (auto layout : layouts) {
      Attribute newLayout;
      if (isFreeChangeOp(op))
        newLayout = cast<ChangeOp>(op).getOperand().getType().getEncoding();
      else if (isa<ChangeOp>(op))
        newLayout = layout;
      else
        newLayout = inferDefLayout(op, layout);
    }
    if (changed)
      changedValues.push_back(value);
  }
}
