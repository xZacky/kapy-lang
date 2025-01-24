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
/// 4. Update the IR by walking the function in dominance order.
class LayoutPropagation {
public:
  LayoutPropagation(FuncOp funcOp) : funcOp(funcOp) {}

  /// Find the anchor operations and set their layouts.
  void initialize();

  /// Recursively propagate the layouts from anchor operations until we reach a
  /// fixed point.
  void propagate();

  /// Resolve for each tensor by choosing one from all the layouts associated to
  /// it. We try to minimize the total cost by doing this.
  void resolve();

  /// Update the IR for a function.
  void update();

private:
  FuncOp funcOp;
  SmallVector<Operation *> anchorOps;
  DenseMap<Value, SetVector<Attribute>> valueToLayouts;
  DenseMap<OpOperand *, Value> operandMapping;

  struct LayoutResolver {
    // Layout choosed for each tensor.
    DenseMap<Value, Attribute> valueToLayout;
    // Layout for each operand that need to be changed.
    DenseMap<OpOperand *, Attribute> operandToLayout;
    // The minimum cost, we represent cost by three integers that corresponding
    // to global memory transactions, shared memory usage, and warp shuffles.
    // We always try to minimize global memory transactions first, then shared
    // memory usage, and warp shuffles last.
    std::array<int64_t, 3> minCost;
  };

  /// Propagate layouts from `value`, return all the changed values.
  SmallVector<Value> propagateToUsers(Value value,
                                      const SetVector<Attribute> &layouts);
  SmallVector<Value> propagateToDefOp(Value value,
                                      const SetVector<Attribute> &layouts);

  /// Propagate the layouts to all the values and fill out them with new layout
  /// into `changedValues`.
  void propagateToResults(Operation *op, ValueRange values,
                          const SetVector<Attribute> &layouts,
                          SmallVectorImpl<Value> &changedValues);
  void propagateToOperands(Operation *op, ValueRange values,
                           const SetVector<Attribute> &layouts,
                           SmallVectorImpl<Value> &changedValues);

  void computeCost(LayoutResolver &resolver);

  void update(Operation *op);
  void update(scf::ForOp op);
  void update(scf::WhileOp op);
  void update(scf::IfOp op);
  void updateOperands(Operation *op);
};

} // namespace

void LayoutPropagation::initialize() {
  auto numWarps = getNumWarps(funcOp->getParentOfType<ModuleOp>());
  funcOp.walk([&](Operation *op) {
    if (isExpensiveMemoryRead(op) || isExpensiveMemoryWrite(op)) {
      anchorOps.push_back(op);
      auto layouts = getCandidateLayouts(op, numWarps);
      for (auto operand : op->getOperands()) {
        if (auto tensorType = dyn_cast<RankedTensorType>(operand.getType())) {
          valueToLayouts[operand] = layouts;
          valueToLayouts[operand].insert(tensorType.getEncoding());
        }
      }
      for (auto result : op->getResults()) {
        if (auto tensorType = dyn_cast<RankedTensorType>(result.getType())) {
          valueToLayouts[result] = layouts;
          valueToLayouts[result].insert(tensorType.getEncoding());
        }
      }
    } else if (isa<MatmulOp>(op)) {
      anchorOps.push_back(op);
      auto layouts = getCandidateLayouts(op, numWarps);
      for (auto result : op->getResults()) {
        if (auto tensorType = dyn_cast<RankedTensorType>(result.getType())) {
          valueToLayouts[result] = layouts;
          valueToLayouts[result].insert(tensorType.getEncoding());
        }
      }
    }
  });
}

void LayoutPropagation::propagate() {
  SmallVector<Value> list;
  // Forward propagation.
  for (auto *op : anchorOps)
    if (isExpensiveMemoryRead(op) || isa<MatmulOp>(op))
      for (auto result : op->getResults())
        if (isa<RankedTensorType>(result.getType()))
          list.push_back(result);
  while (!list.empty()) {
    auto value = list.pop_back_val();
    const auto &layouts = valueToLayouts[value];
    auto changedValues = propagateToUsers(value, layouts);
    list.insert(list.end(), changedValues.begin(), changedValues.end());
  }
  // Backward propagation.
  for (auto *op : anchorOps)
    if (isExpensiveMemoryWrite(op))
      for (auto operand : op->getOperands())
        if (isa<RankedTensorType>(operand.getType()))
          list.push_back(operand);
  while (!list.empty()) {
    auto value = list.pop_back_val();
    const auto &layouts = valueToLayouts[value];
    auto changedValues = propagateToDefOp(value, layouts);
    list.insert(list.end(), changedValues.begin(), changedValues.end());
  }
}

void LayoutPropagation::resolve() {
  LayoutResolver resolver;
  for (auto *op : anchorOps) {
    // if (isExpensiveMemoryRead(op) || isExpensiveMemoryWrite(op)) {
    //   SmallVector<Value> values;
    //   for (auto operand : op->getOperands())
    //     if (isa<RankedTensorType>(operand.getType()))
    //       values.push_back(operand);
    //   for (auto result : op->getResults())
    //     if (isa<RankedTensorType>(result.getType()))
    //       values.push_back(result);
    //   SetVector<Attribute> layouts;
    //   for (auto value : values)
    //     if (valueToLayouts.contains(value))
    //       layouts.insert(valueToLayouts[value].begin(),
    //                      valueToLayouts[value].end());
    // }
  }
  computeCost(resolver);
}

void LayoutPropagation::update() {
  SmallVector<Region *> list;
  list.push_back(&funcOp.getBody());
  while (!list.empty()) {
    auto *region = list.pop_back_val();
    for (auto &op : region->getOps()) {
      bool updateResults = false;
      for (auto result : op.getResults()) {
        if (!valueToLayouts.contains(result))
          continue;
        const auto &layouts = valueToLayouts[result];
        assert(layouts.size() == 1);
        auto tensorType = cast<RankedTensorType>(result.getType());
        if (tensorType.getEncoding() == *layouts.begin())
          continue;
        updateResults = true;
      }
      if (updateResults)
        update(&op);
      else
        updateOperands(&op);
      for (auto &subRegion : op.getRegions())
        list.push_back(&subRegion);
    }
  }
}

SmallVector<Value>
LayoutPropagation::propagateToUsers(Value value,
                                    const SetVector<Attribute> &layouts) {
  SmallVector<Value> changedValues;
  for (auto &use : value.getUses()) {
    auto *useOp = use.getOwner();
    auto index = use.getOperandNumber();
    if (auto forOp = dyn_cast<scf::ForOp>(useOp)) {
      auto iterArg = forOp.getTiedLoopRegionIterArg(&use);
      propagateToResults(useOp, iterArg, layouts, changedValues);
      continue;
    }
    if (auto whileOp = dyn_cast<scf::WhileOp>(useOp)) {
      auto beforeArg = whileOp.getBeforeArguments()[index];
      propagateToResults(useOp, beforeArg, layouts, changedValues);
      continue;
    }
    if (auto yieldOp = dyn_cast<scf::YieldOp>(useOp)) {
      auto *parentOp = yieldOp->getParentOp();
      SmallVector<Value> values;
      if (isa<scf::ForOp, scf::IfOp>(parentOp))
        values.push_back(parentOp->getResult(index));
      if (auto forOp = dyn_cast<scf::ForOp>(parentOp))
        values.push_back(forOp.getRegionIterArg(index));
      if (auto whileOp = dyn_cast<scf::WhileOp>(parentOp))
        values.push_back(whileOp.getBeforeArguments()[index]);
      propagateToResults(useOp, values, layouts, changedValues);
      continue;
    }
    if (auto conditionOp = dyn_cast<scf::ConditionOp>(useOp)) {
      auto whileOp = conditionOp.getParentOp();
      auto afterArg = whileOp.getAfterArguments()[index - 1];
      auto result = whileOp->getResult(index - 1);
      propagateToResults(useOp, {afterArg, result}, layouts, changedValues);
      continue;
    }
    if (useOp->hasTrait<OpTrait::SameOperandsAndResultLayout>() ||
        useOp->hasTrait<OpTrait::Elementwise>() ||
        isa<ReduceOp, UnsqueezeOp, ChangeOp>(useOp)) {
      propagateToResults(useOp, useOp->getResults(), layouts, changedValues);
      continue;
    }
  }
  return changedValues;
}

void LayoutPropagation::propagateToResults(
    Operation *op, ValueRange values, const SetVector<Attribute> &layouts,
    SmallVectorImpl<Value> &changedValues) {
  for (auto value : values) {
    if (!isa<RankedTensorType>(value.getType()))
      continue;
    bool changed = false;
    for (auto layout : layouts) {
      Attribute newLayout;
      if (auto changeOp = dyn_cast<ChangeOp>(op)) {
        // Do not propagate through change to matmul operand layout.
        if (auto mmopdLayout =
                dyn_cast<MmOperandLayoutAttr>(changeOp.getType().getEncoding()))
          newLayout = mmopdLayout;
        // Try to remove the change by making the result layout same as operand.
        else
          newLayout = layout;
      } else {
        newLayout = inferResultLayout(op, layout);
      }
      if (newLayout)
        changed |= valueToLayouts[value].insert(newLayout);
    }
    if (changed)
      changedValues.push_back(value);
  }
}

SmallVector<Value>
LayoutPropagation::propagateToDefOp(Value value,
                                    const SetVector<Attribute> &layouts) {
  SmallVector<Value> changedValues;
  if (auto forOp = value.getDefiningOp<scf::ForOp>()) {
    auto *yieldOp = forOp.getBody()->getTerminator();
    auto result = cast<OpResult>(value);
    auto operand = yieldOp->getOperand(result.getResultNumber());
    propagateToOperands(forOp, operand, layouts, changedValues);
    return changedValues;
  }
  if (auto whileOp = value.getDefiningOp<scf::WhileOp>()) {
    auto conditionOp = whileOp.getBeforeBody()->getTerminator();
    auto result = cast<OpResult>(value);
    auto operand = conditionOp->getOperand(result.getResultNumber() + 1);
    propagateToOperands(whileOp, operand, layouts, changedValues);
    return changedValues;
  }
  if (auto ifOp = value.getDefiningOp<scf::IfOp>()) {
    auto index = cast<OpResult>(value).getResultNumber();
    auto thenOperand = ifOp.thenYield().getOperand(index);
    auto elseOperand = ifOp.elseYield().getOperand(index);
    propagateToOperands(ifOp, {thenOperand, elseOperand}, layouts,
                        changedValues);
    return changedValues;
  }
  if (auto *defOp = value.getDefiningOp()) {
    if (defOp->hasTrait<OpTrait::SameOperandsAndResultLayout>() ||
        defOp->hasTrait<OpTrait::Elementwise>() ||
        isa<ReduceOp, UnsqueezeOp, TransposeOp, ChangeOp>(defOp)) {
      propagateToOperands(defOp, defOp->getOperands(), layouts, changedValues);
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
      propagateToOperands(forOp, {init, operand}, layouts, changedValues);
      return changedValues;
    }
    if (auto whileOp = dyn_cast<scf::WhileOp>(parentOp)) {
      if (region == &whileOp.getBefore()) {
        auto *yieldOp = whileOp.getAfterBody()->getTerminator();
        auto init = whileOp.getTiedLoopInit(blockArg)->get();
        auto operand = yieldOp->getOperand(blockArg.getArgNumber());
        propagateToOperands(whileOp, {init, operand}, layouts, changedValues);
        return changedValues;
      }
      if (region == &whileOp.getAfter()) {
        auto *conditionOp = whileOp.getBeforeBody()->getTerminator();
        auto operand = conditionOp->getOperand(blockArg.getArgNumber() + 1);
        propagateToOperands(whileOp, operand, layouts, changedValues);
        return changedValues;
      }
    }
  }
  return changedValues;
}

void LayoutPropagation::propagateToOperands(
    Operation *op, ValueRange values, const SetVector<Attribute> &layouts,
    SmallVectorImpl<Value> &changedValues) {
  for (auto value : values) {
    if (!isa<RankedTensorType>(value.getType()))
      continue;
    bool changed = false;
    for (auto layout : layouts) {
      Attribute newLayout;
      // Try to remove the change by making the operand layout same as result.
      if (isa<ChangeOp>(op))
        newLayout = layout;
      else
        newLayout = inferOperandLayout(op, layout);
      if (newLayout)
        changed |= valueToLayouts[value].insert(newLayout);
    }
    if (changed)
      changedValues.push_back(value);
  }
}

void LayoutPropagation::computeCost(LayoutResolver &resolver) {
  SmallVector<Region *> list;
  list.push_back(&funcOp.getBody());
  while (!list.empty()) {
    auto *region = list.pop_back_val();
    for (auto &op : region->getOps()) {
      // if (isExpensiveMemoryRead(&op) || isExpensiveMemoryWrite(&op)) {
      //   auto result = op.getResult(0);
      //   resolver.valueToLayout[result] = *valueToLayouts[result].begin();
      //   for (auto operand : op.getOperands()) {
      //
      //   }
      // }
    }
  }
}

void LayoutPropagation::update(Operation *op) {
  if (auto forOp = dyn_cast<scf::ForOp>(op))
    return update(forOp);
  if (auto whileOp = dyn_cast<scf::WhileOp>(op))
    return update(whileOp);
  if (auto ifOp = dyn_cast<scf::IfOp>(op))
    return update(ifOp);

  updateOperands(op);
  for (auto result : op->getResults()) {
    if (auto tensorType = dyn_cast<RankedTensorType>(result.getType())) {
      auto resultLayout = tensorType.getEncoding();
      if (valueToLayouts.contains(result))
        resultLayout = *valueToLayouts[result].begin();
      result.setType(cloneWith(tensorType, resultLayout));
    }
  }
}

void LayoutPropagation::update(scf::ForOp op) {
  updateOperands(op);
  SmallVector<Type> resultTypes;
  for (auto result : op.getResults()) {
    if (auto tensorType = dyn_cast<RankedTensorType>(result.getType())) {
      auto resultLayout = tensorType.getEncoding();
      if (valueToLayouts.contains(result))
        resultLayout = *valueToLayouts[result].begin();
      resultTypes.push_back(cloneWith(tensorType, resultLayout));
    } else {
      resultTypes.push_back(result.getType());
    }
  }
  for (auto [index, iterArg] : llvm::enumerate(op.getRegionIterArgs()))
    iterArg.setType(resultTypes[index]);
  for (auto [index, result] : llvm::enumerate(op.getResults()))
    result.setType(resultTypes[index]);
}

void LayoutPropagation::update(scf::WhileOp op) {
  updateOperands(op);
  for (auto [index, beforeArg] : llvm::enumerate(op.getBeforeArguments()))
    beforeArg.setType(op.getOperand(index).getType());
  SmallVector<Type> resultTypes;
  for (auto result : op.getResults()) {
    if (auto tensorType = dyn_cast<RankedTensorType>(result.getType())) {
      auto resultLayout = tensorType.getEncoding();
      if (valueToLayouts.contains(result))
        resultLayout = *valueToLayouts[result].begin();
      resultTypes.push_back(cloneWith(tensorType, resultLayout));
    } else {
      resultTypes.push_back(result.getType());
    }
  }
  for (auto [index, afterArg] : llvm::enumerate(op.getAfterArguments()))
    afterArg.setType(resultTypes[index]);
  for (auto [index, result] : llvm::enumerate(op.getResults()))
    result.setType(resultTypes[index]);
}

void LayoutPropagation::update(scf::IfOp op) {
  for (auto result : op.getResults()) {
    if (auto tensorType = dyn_cast<RankedTensorType>(result.getType())) {
      auto resultLayout = tensorType.getEncoding();
      if (valueToLayouts.contains(result))
        resultLayout = *valueToLayouts[result].begin();
      result.setType(cloneWith(tensorType, resultLayout));
    }
  }
}

void LayoutPropagation::updateOperands(Operation *op) {
  for (auto &operand : op->getOpOperands())
    if (operandMapping.contains(&operand))
      op->setOperand(operand.getOperandNumber(), operandMapping[&operand]);
}
