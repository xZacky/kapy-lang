//===- OptimizeLayout.h -----------------------------------------*- C++ -*-===//
//
// This file implements the KgpuOptimizeLayoutPass.
//
//===----------------------------------------------------------------------===//

#include "kapy/Dialect/Kapy/IR/Kapy.h"
#include "kapy/Dialect/Kapy/Transforms/TransformUtils.h"
#include "kapy/Dialect/Kgpu/IR/Kgpu.h"
#include "kapy/Dialect/Kgpu/Transforms/Passes.h"
#include "kapy/Support/CommonUtils.h"
#include "kapy/Support/LayoutUtils.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "llvm/ADT/MapVector.h"

using namespace mlir;
using namespace mlir::kapy;

namespace {

/// The current algorithm works by analyzing the IR and doing a one-shot rewrite
/// based on the analysis. The algorithm is as follows:
/// 1. Find all the anchor operations. We will set several candidate layouts for
///    them and finally choose one of these layouts.
/// 2. For each anchor operation, propagate its layouts to other operations, now
///    each tensor may have multiple layouts associated with it.
/// 3. Resolve conflicts by deciding which one of the layouts each tensor should
///    keep, insert ChangeOp when necessary.
/// 4. Rewrite the IR by walking the function in dominance order.
///
/// This algorithm is similar to LayoutPropagation in Triton but different at:
/// 1. Triton only do forward propagation and this algorithm do both forward and
///    backward propagation.
/// 2. Triton depends on backward rematerialization to remove some layout change
///    and may duplicate computation. Currently this algorithm does not consider
///    duplicate computation, but try to reduce cost of layout change by using a
///    better layout.
/// 3. This algorithm will search in a larger space and select from combinations
///    according to a simple cost model (see CostInfo).
///
/// TODO: Enhance the cost model.
class LayoutOptimization {
public:
  explicit LayoutOptimization(FuncOp funcOp);

  /// Find the anchor operations and set their layouts.
  void initialize();

  /// Recursively propagate the layouts from anchor operations until we reach a
  /// fixed point. After that we get all the possible layouts for each tensor.
  void propagate();

  /// Resolve conflicts for each tensor. We will try all the combinations of the
  /// anchor layouts. For each combination, we propagate them to the remain part
  /// of the IR, and insert ChangeOp for each operand that need a new layout for
  /// its owner. We try to minimize the total cost by doing this.
  void resolve();

  /// Rewrite the IR for the function.
  void rewrite();

private:
  /// We represent cost by the number of shuffles in each basic block.
  struct CostInfo {
    llvm::MapVector<Block *, int64_t> blockToNumShfls;

    /// Only returns true if cost of `*this` is strictly less than `other`.
    /// That is, at least 1 block is less, and other blocks is less or equal.
    bool operator<(const CostInfo &other) const {
      bool lessThan = false;
      for (auto [block, numShfls] : blockToNumShfls) {
        if (!other.blockToNumShfls.contains(block) && numShfls != 0)
          return false;
        if (numShfls > other.blockToNumShfls.lookup(block))
          return false;
        if (numShfls < other.blockToNumShfls.lookup(block))
          lessThan = true;
      }
      return lessThan;
    }
  };

  struct ResolveState {
    // Layout selected for each tensor.
    DenseMap<Value, FragmentsLayoutAttr> valueToLayout;
    // Operands that need a new layout for its owner.
    llvm::MapVector<OpOperand *, FragmentsLayoutAttr> operandToLayout;
    // Shared memory to valid layouts (no bank conflict).
    DenseMap<Value, SetVector<SwizzlingLayoutAttr>> sharedToLayouts;
    // CostInfo of this state.
    CostInfo cost;
  };

  FuncOp funcOp;
  SmallVector<Operation *, 8> anchorOps;
  // Layouts of anchor operations' operands.
  DenseMap<OpOperand *, SetVector<FragmentsLayoutAttr>> operandToLayouts;
  // Layouts of anchor operations' results.
  DenseMap<OpResult, SetVector<FragmentsLayoutAttr>> resultToLayouts;
  // Layouts of values after propagation.
  llvm::MapVector<Value, SetVector<FragmentsLayoutAttr>> valueToLayouts;
  // Values for operands that need a new layout for its owner.
  DenseMap<OpOperand *, Value> operandMapping;
  // The minimum cost state.
  ResolveState minState;

  /// Propagate layouts from `value`, populate changed values.
  void propagateToUsers(Value value,
                        const SetVector<FragmentsLayoutAttr> &layouts,
                        SmallVectorImpl<Value> &changedValues);
  void propagateToDefOp(Value value,
                        const SetVector<FragmentsLayoutAttr> &layouts,
                        SmallVectorImpl<Value> &changedValues);
  void propagateToUseValues(Operation *op, ValueRange values,
                            const SetVector<FragmentsLayoutAttr> &layouts,
                            SmallVectorImpl<Value> &changedValues);
  void propagateToDefValues(Operation *op, ValueRange values,
                            const SetVector<FragmentsLayoutAttr> &layouts,
                            SmallVectorImpl<Value> &changedValues);

  /// Resolve conflicts for the remain part of the IR.
  /// Fails if we can not avoid bank conflict.
  LogicalResult resolve(ResolveState &curState);

  llvm::MapVector<Value, SetVector<FragmentsLayoutAttr>>
  collectValuesNeedChange(ResolveState &curState);

  void computeCost(ResolveState &curState);

  // Insert ChangeOp to places we need.
  void insertChangeOps();

  void rewrite(Operation *op);
  void rewrite(scf::ForOp op);
  void rewrite(scf::WhileOp op);
  void rewrite(scf::IfOp op);
  void rewrite(arith::ConstantOp op);
  void setOperands(Operation *op);
};

} // namespace

LayoutOptimization::LayoutOptimization(FuncOp funcOp) : funcOp(funcOp) {
  funcOp.walk([&](ChangeOp op) {
    auto *block = op->getBlock();
    ChangeOpHelper helper(op);
    minState.cost.blockToNumShfls[block] += helper.getNumShfls();
  });
  funcOp.walk([&](ReduceOp op) {
    auto *block = op->getBlock();
    ReduceOpHelper helper(op);
    minState.cost.blockToNumShfls[block] += helper.getNumShfls();
  });
}

void LayoutOptimization::initialize() {
  funcOp.walk([&](Operation *op) {
    // Currently anchor operations:
    // MatmulOp, LdGlobalOp, StGlobalOp, AtomicRMWOp
    if (isa<MatmulOp>(op)) {
      anchorOps.push_back(op);
      auto &lhs = op->getOpOperand(0);
      auto lhsType = cast<RankedTensorType>(lhs.get().getType());
      auto lhsLayout = getLayout<FragmentsLayoutAttr>(lhsType);
      operandToLayouts[&lhs].insert(lhsLayout);
      auto &rhs = op->getOpOperand(1);
      auto rhsType = cast<RankedTensorType>(rhs.get().getType());
      auto rhsLayout = getLayout<FragmentsLayoutAttr>(rhsType);
      operandToLayouts[&rhs].insert(rhsLayout);
      auto &acc = op->getOpOperand(2);
      auto accType = cast<RankedTensorType>(acc.get().getType());
      auto accLayout = getLayout<FragmentsLayoutAttr>(accType);
      operandToLayouts[&acc].insert(accLayout);
      auto result = op->getResult(0);
      resultToLayouts[result].insert(accLayout);
      return;
    }
    if (isa<LdMatrixOp, CpAsyncGlobalToSharedOp>(op))
      return;
    if (isGlobalMemoryRead(op) || isGlobalMemoryWrite(op)) {
      anchorOps.push_back(op);
      auto layouts = getCandidateLayouts(op);
      for (auto &operand : op->getOpOperands()) {
        auto tensorType = cast<RankedTensorType>(operand.get().getType());
        if (!inRegisterFile(tensorType))
          continue;
        operandToLayouts[&operand] = layouts;
      }
      // We assume that memory access operations have 0 or 1 result.
      if (op->getNumResults() == 1) {
        auto result = op->getResult(0);
        resultToLayouts[result] = std::move(layouts);
      }
      return;
    }
  });
}

void LayoutOptimization::propagate() {
  for (auto *op : anchorOps) {
    for (auto &operand : op->getOpOperands()) {
      if (operandToLayouts.contains(&operand)) {
        const auto &layouts = operandToLayouts[&operand];
        valueToLayouts[operand.get()].insert(layouts.begin(), layouts.end());
      }
    }
    if (op->getNumResults() == 1) {
      auto result = op->getResult(0);
      if (resultToLayouts.contains(result)) {
        const auto &layouts = resultToLayouts[result];
        valueToLayouts[result].insert(layouts.begin(), layouts.end());
      }
    }
  }

  SmallVector<Value> list;
  for (auto value : llvm::make_first_range(valueToLayouts))
    list.push_back(value);
  while (!list.empty()) {
    auto value = list.pop_back_val();
    SmallVector<Value> changedValues;
    propagateToUsers(value, valueToLayouts[value], changedValues);
    propagateToDefOp(value, valueToLayouts[value], changedValues);
    list.append(changedValues.begin(), changedValues.end());
  }
}

static bool alwaysLessThan(ArrayRef<int64_t> lhsArray,
                           ArrayRef<int64_t> rhsArray) {
  for (unsigned i = 0; i < lhsArray.size(); ++i)
    if (lhsArray[i] >= rhsArray[i])
      return false;
  return true;
}

static bool requiresSameOperandsLayout(Operation *op) {
  return op->hasTrait<OpTrait::SameOperandsLayout>() ||
         op->hasTrait<OpTrait::SameOperandsAndResultLayout>() ||
         op->hasTrait<OpTrait::SameTypeOperands>() ||
         op->hasTrait<OpTrait::SameOperandsAndResultType>() ||
         op->hasTrait<OpTrait::Elementwise>();
}

void LayoutOptimization::resolve() {
  // Now we regard these operations as anchors too:
  // 1. LdMatrixOp, LdSharedOp, ConstantOp, SplatOp, it is start point.
  // Now we regard these operands as anchors too:
  // 1. Operands of ForOp, WhileOp, ConditionOp.
  // 2. Operands of YieldOp with an IfOp parent.
  // 3. Operand of BroadcastOp, ExtUIOp, ExtSIOp, ExtFOp, FPToFPOp (up cast), it
  //    may increase cost of ChangeOp.
  // 4. Operands of operation requires multiple same layout operands.
  SmallVector<OpOperand *, 8> anchorOperands;
  funcOp.walk([&](Operation *op) {
    // Operands of ForOp, WhileOp, ConditionOp.
    if (isa<scf::ForOp, scf::WhileOp, scf::ConditionOp>(op))
      for (auto &operand : op->getOpOperands())
        if (valueToLayouts.contains(operand.get()))
          anchorOperands.push_back(&operand);
    // Operands of YieldOp with an IfOp parent.
    if (auto ifOp = dyn_cast<scf::IfOp>(op))
      for (auto &operand : ifOp.thenYield()->getOpOperands())
        if (valueToLayouts.contains(operand.get()))
          anchorOperands.push_back(&operand);
    // Operand of BroadcastOp, ExtUIOp, ExtSIOp, ExtFOp.
    if (isa<BroadcastOp, arith::ExtUIOp, arith::ExtSIOp, arith::ExtFOp>(op)) {
      auto &operand = op->getOpOperand(0);
      if (valueToLayouts.contains(operand.get()))
        anchorOperands.push_back(&operand);
    }
    // Operand of FPToFPOp (up cast).
    if (auto fptofpOp = dyn_cast<FPToFPOp>(op)) {
      auto &operand = op->getOpOperand(0);
      if (fptofpOp.isUpCast() && valueToLayouts.contains(operand.get()))
        anchorOperands.push_back(&operand);
    }
    // Operands of operation requires multiple same layout operands.
    if (op->getNumOperands() > 1 && requiresSameOperandsLayout(op)) {
      for (auto &operand : op->getOpOperands()) {
        if (valueToLayouts.contains(operand.get())) {
          anchorOperands.push_back(&operand);
          break;
        }
      }
    }
    // LdMatrixOp, LdSharedOp, ConstantOp and SplatOp.
    if (isa<LdMatrixOp, LdSharedOp, arith::ConstantOp, SplatOp>(op)) {
      auto result = op->getResult(0);
      if (valueToLayouts.contains(result))
        anchorOps.push_back(op);
    }
  });

  /// Get the combination domain of anchor layouts.
  auto numAnchors = anchorOps.size() + anchorOperands.size();
  SmallVector<unsigned, 16> combDomain(numAnchors);
  for (unsigned anchorId = 0; anchorId < numAnchors; ++anchorId) {
    if (anchorId < anchorOps.size()) {
      auto *op = anchorOps[anchorId];
      if (isa<LdMatrixOp>(op)) {
        auto result = op->getResult(0);
        auto resultType = cast<RankedTensorType>(result.getType());
        auto lhsLayout = getLayout<FragmentsLayoutAttr>(resultType);
        SmallVector<int64_t> lhsCosts;
        for (auto newLayout : valueToLayouts[result]) {
          auto oldType = cloneWithLayout(resultType, lhsLayout);
          auto newType = cloneWithLayout(resultType, newLayout);
          ChangeOpHelper helper(oldType, newType);
          lhsCosts.push_back(helper.getNumShfls());
        }
        auto rhsLayout = lhsLayout.transpose();
        SmallVector<int64_t> rhsCosts;
        for (auto newLayout : valueToLayouts[result]) {
          auto oldType = cloneWithLayout(resultType, rhsLayout);
          auto newType = cloneWithLayout(resultType, newLayout);
          ChangeOpHelper helper(oldType, newType);
          rhsCosts.push_back(helper.getNumShfls());
        }
        if (alwaysLessThan(lhsCosts, rhsCosts)) {
          resultToLayouts[result].insert(lhsLayout);
          combDomain[anchorId] = 1;
          continue;
        }
        if (alwaysLessThan(rhsCosts, lhsCosts)) {
          resultToLayouts[result].insert(rhsLayout);
          combDomain[anchorId] = 1;
          continue;
        }
        resultToLayouts[result].insert(lhsLayout);
        resultToLayouts[result].insert(rhsLayout);
        combDomain[anchorId] = 2;
        continue;
      }
      if (isa<MatmulOp>(op)) {
        combDomain[anchorId] = 1;
        continue;
      }
      if (op->getNumResults() == 1) {
        auto result = op->getResult(0);
        if (resultToLayouts.contains(result))
          combDomain[anchorId] = resultToLayouts[result].size();
        else
          combDomain[anchorId] = valueToLayouts[result].size();
      } else {
        for (auto &operand : op->getOpOperands()) {
          if (operandToLayouts.contains(&operand)) {
            combDomain[anchorId] = operandToLayouts[&operand].size();
            break;
          }
        }
      }
    } else {
      auto *operand = anchorOperands[anchorId - anchorOps.size()];
      combDomain[anchorId] = valueToLayouts[operand->get()].size();
    }
  }

  // Compute the number of combinations.
  auto numCombs = product(combDomain);
  // Try to resolve conflicts for each combination.
  // Find the minimum cost combination.
  for (unsigned combId = 0; combId < numCombs; ++combId) {
    ResolveState curState;
    auto selects = delinearize(combId, combDomain);
    for (unsigned anchorId = 0; anchorId < numAnchors; ++anchorId) {
      auto select = selects[anchorId];
      if (anchorId < anchorOps.size()) {
        auto *op = anchorOps[anchorId];
        if (isa<LdMatrixOp>(op)) {
          auto result = op->getResult(0);
          auto layout = resultToLayouts[result][select];
          curState.valueToLayout[result] = layout;
          continue;
        }
        if (isa<MatmulOp>(op)) {
          auto &lhs = op->getOpOperand(0);
          auto lhsLayout = operandToLayouts[&lhs][select];
          curState.operandToLayout[&lhs] = lhsLayout;
          auto &rhs = op->getOpOperand(1);
          auto rhsLayout = operandToLayouts[&rhs][select];
          curState.operandToLayout[&rhs] = rhsLayout;
          auto &acc = op->getOpOperand(2);
          auto accLayout = operandToLayouts[&acc][select];
          curState.operandToLayout[&acc] = accLayout;
          auto result = op->getResult(0);
          curState.valueToLayout[result] = accLayout;
          continue;
        }
        if (isa<LdSharedOp, arith::ConstantOp, SplatOp>(op)) {
          auto result = op->getResult(0);
          auto layout = valueToLayouts[result][select];
          curState.valueToLayout[result] = layout;
          continue;
        }
        if (isGlobalMemoryRead(op) || isGlobalMemoryWrite(op)) {
          for (auto &operand : op->getOpOperands()) {
            if (operandToLayouts.contains(&operand)) {
              auto layout = operandToLayouts[&operand][select];
              curState.operandToLayout[&operand] = layout;
            }
          }
          if (op->getNumResults() == 1) {
            auto result = op->getResult(0);
            if (resultToLayouts.contains(result)) {
              auto layout = resultToLayouts[result][select];
              curState.valueToLayout[result] = layout;
            }
          }
          continue;
        }
      } else {
        auto *operand = anchorOperands[anchorId - anchorOps.size()];
        auto *user = operand->getOwner();
        auto index = operand->getOperandNumber();
        auto layout = valueToLayouts[operand->get()][select];
        curState.operandToLayout[operand] = layout;
        if (auto forOp = dyn_cast<scf::ForOp>(user)) {
          auto iterArg = forOp.getTiedLoopRegionIterArg(operand);
          curState.valueToLayout[iterArg] = layout;
          auto *yielded = forOp.getTiedLoopYieldedValue(iterArg);
          curState.operandToLayout[yielded] = layout;
          auto result = forOp.getTiedLoopResult(operand);
          curState.valueToLayout[result] = layout;
          continue;
        }
        if (auto whileOp = dyn_cast<scf::WhileOp>(user)) {
          auto beforeArg = whileOp.getBeforeArguments()[index];
          curState.valueToLayout[beforeArg] = layout;
          auto *yieldOp = whileOp.getAfterBody()->getTerminator();
          auto &yielded = yieldOp->getOpOperand(index);
          curState.operandToLayout[&yielded] = layout;
          continue;
        }
        if (auto conditionOp = dyn_cast<scf::ConditionOp>(user)) {
          auto whileOp = conditionOp.getParentOp();
          auto afterArg = whileOp.getAfterArguments()[index - 1];
          curState.valueToLayout[afterArg] = layout;
          auto result = whileOp->getResult(index - 1);
          curState.valueToLayout[result] = layout;
          continue;
        }
        if (auto thenOp = dyn_cast<scf::YieldOp>(user)) {
          auto ifOp = cast<scf::IfOp>(thenOp->getParentOp());
          auto result = ifOp->getResult(index);
          curState.valueToLayout[result] = layout;
          auto elseOp = ifOp.elseYield();
          auto &yielded = elseOp->getOpOperand(index);
          curState.operandToLayout[&yielded] = layout;
          continue;
        }
        if (user->getNumOperands() > 1 && requiresSameOperandsLayout(user)) {
          for (auto &use : user->getOpOperands()) {
            if (operand == &use)
              continue;
            curState.operandToLayout[&use] = layout;
          }
          continue;
        }
      }
    }
    if (failed(resolve(curState)))
      continue;
    computeCost(curState);
    if (curState.cost < minState.cost)
      minState = std::move(curState);
  }

  operandToLayouts.clear();
  resultToLayouts.clear();
  valueToLayouts.clear();
  insertChangeOps();
}

void LayoutOptimization::rewrite() {
  funcOp.walk([&](Operation *op) {
    bool needRewrite = false;
    for (auto result : op->getResults()) {
      if (!minState.valueToLayout.contains(result))
        continue;
      needRewrite = true;
    }
    if (needRewrite)
      rewrite(op);
    else
      setOperands(op);
  });
  funcOp.walk([&](MkSharedOp op) {
    auto result = op.getResult();
    if (minState.sharedToLayouts.contains(result)) {
      DenseSet<Value> seen;
      propagateMemoryLayout(result, minState.sharedToLayouts[result][0], seen);
    }
  });
}

static bool willPassLayout(Operation *op) {
  return op->hasTrait<OpTrait::SameOperandsAndResultLayout>() ||
         op->hasTrait<OpTrait::SameOperandsAndResultType>() ||
         op->hasTrait<OpTrait::Elementwise>();
}

static FragmentsLayoutAttr inferLayout(Operation *op,
                                       FragmentsLayoutAttr layout) {
  if (willPassLayout(op))
    return layout;
  if (isa<scf::ForOp, scf::WhileOp, scf::IfOp, ChangeOp>(op))
    return layout;
  if (isa<TransposeOp>(op))
    return layout.transpose();
  return FragmentsLayoutAttr();
}

void LayoutOptimization::propagateToUsers(
    Value value, const SetVector<FragmentsLayoutAttr> &layouts,
    SmallVectorImpl<Value> &changedValues) {
  for (auto &use : value.getUses()) {
    auto *user = use.getOwner();
    auto index = use.getOperandNumber();
    if (auto forOp = dyn_cast<scf::ForOp>(user)) {
      auto iterArg = forOp.getTiedLoopRegionIterArg(&use);
      propagateToUseValues(user, iterArg, layouts, changedValues);
      continue;
    }
    if (auto whileOp = dyn_cast<scf::WhileOp>(user)) {
      auto beforeArg = whileOp.getBeforeArguments()[index];
      propagateToUseValues(user, beforeArg, layouts, changedValues);
      continue;
    }
    if (auto yieldOp = dyn_cast<scf::YieldOp>(user)) {
      auto *parentOp = yieldOp->getParentOp();
      if (auto forOp = dyn_cast<scf::ForOp>(parentOp)) {
        auto iterArg = forOp.getRegionIterArg(index);
        auto result = forOp->getResult(index);
        propagateToUseValues(forOp, {iterArg, result}, layouts, changedValues);
        continue;
      }
      if (auto whileOp = dyn_cast<scf::WhileOp>(parentOp)) {
        auto beforeArg = whileOp.getBeforeArguments()[index];
        propagateToUseValues(whileOp, beforeArg, layouts, changedValues);
        continue;
      }
      if (auto ifOp = dyn_cast<scf::IfOp>(parentOp)) {
        auto result = ifOp->getResult(index);
        propagateToUseValues(ifOp, result, layouts, changedValues);
        continue;
      }
    }
    if (auto conditionOp = dyn_cast<scf::ConditionOp>(user)) {
      auto whileOp = conditionOp.getParentOp();
      auto afterArg = whileOp.getAfterArguments()[index - 1];
      auto result = whileOp.getResult(index - 1);
      propagateToUseValues(whileOp, {afterArg, result}, layouts, changedValues);
      continue;
    }
    if (willPassLayout(user) || isa<TransposeOp, ChangeOp>(user)) {
      propagateToUseValues(user, user->getResults(), layouts, changedValues);
      continue;
    }
  }
}

void LayoutOptimization::propagateToUseValues(
    Operation *op, ValueRange values,
    const SetVector<FragmentsLayoutAttr> &layouts,
    SmallVectorImpl<Value> &changedValues) {
  for (auto value : values) {
    auto tensorType = dyn_cast<RankedTensorType>(value.getType());
    if (!tensorType || !inRegisterFile(tensorType))
      continue;
    bool changed = false;
    for (auto layout : layouts) {
      auto newLayout = inferLayout(op, layout);
      if (newLayout)
        changed |= valueToLayouts[value].insert(newLayout);
    }
    if (changed)
      changedValues.push_back(value);
  }
}

void LayoutOptimization::propagateToDefOp(
    Value value, const SetVector<FragmentsLayoutAttr> &layouts,
    SmallVectorImpl<Value> &changedValues) {
  if (auto forOp = value.getDefiningOp<scf::ForOp>()) {
    auto *yieldOp = forOp.getBody()->getTerminator();
    auto index = cast<OpResult>(value).getResultNumber();
    auto yielded = yieldOp->getOperand(index);
    propagateToDefValues(forOp, yielded, layouts, changedValues);
    return;
  }
  if (auto whileOp = value.getDefiningOp<scf::WhileOp>()) {
    auto *conditionOp = whileOp.getBeforeBody()->getTerminator();
    auto index = cast<OpResult>(value).getResultNumber();
    auto yielded = conditionOp->getOperand(index + 1);
    propagateToDefValues(whileOp, yielded, layouts, changedValues);
    return;
  }
  if (auto ifOp = value.getDefiningOp<scf::IfOp>()) {
    auto index = cast<OpResult>(value).getResultNumber();
    std::array<Value, 2> values;
    values[0] = ifOp.thenYield().getOperand(index);
    values[1] = ifOp.elseYield().getOperand(index);
    propagateToDefValues(ifOp, values, layouts, changedValues);
    return;
  }
  if (auto *defOp = value.getDefiningOp()) {
    if (willPassLayout(defOp) || isa<TransposeOp, ChangeOp>(defOp))
      propagateToDefValues(defOp, defOp->getOperands(), layouts, changedValues);
    return;
  }
  if (auto blockArg = dyn_cast<BlockArgument>(value)) {
    auto index = blockArg.getArgNumber();
    auto *block = blockArg.getOwner();
    auto *parentOp = block->getParentOp();
    if (auto forOp = dyn_cast<scf::ForOp>(parentOp)) {
      auto init = forOp.getTiedLoopInit(blockArg)->get();
      auto yielded = forOp.getTiedLoopYieldedValue(blockArg)->get();
      propagateToDefValues(forOp, {init, yielded}, layouts, changedValues);
      return;
    }
    if (auto whileOp = dyn_cast<scf::WhileOp>(parentOp)) {
      if (block == whileOp.getBeforeBody()) {
        auto *yieldOp = whileOp.getAfterBody()->getTerminator();
        auto init = whileOp->getOperand(index);
        auto yielded = yieldOp->getOperand(index);
        propagateToDefValues(whileOp, {init, yielded}, layouts, changedValues);
        return;
      }
      if (block == whileOp.getAfterBody()) {
        auto *conditionOp = whileOp.getBeforeBody()->getTerminator();
        auto yielded = conditionOp->getOperand(index + 1);
        propagateToDefValues(whileOp, yielded, layouts, changedValues);
        return;
      }
    }
  }
}

void LayoutOptimization::propagateToDefValues(
    Operation *op, ValueRange values,
    const SetVector<FragmentsLayoutAttr> &layouts,
    SmallVectorImpl<Value> &changedValues) {
  for (auto value : values) {
    auto tensorType = dyn_cast<RankedTensorType>(value.getType());
    if (!tensorType || !inRegisterFile(tensorType))
      continue;
    bool changed = false;
    for (auto layout : layouts) {
      auto newLayout = inferLayout(op, layout);
      if (newLayout)
        changed |= valueToLayouts[value].insert(newLayout);
    }
    if (changed)
      changedValues.push_back(value);
  }
}

/// Collect operations read or write this shared memory.
static void collectAccessOps(MkSharedOp op, SetVector<Operation *> &ops) {
  SmallVector<Operation *> list;
  DenseSet<Operation *> seen;
  list.push_back(op);
  while (!list.empty()) {
    auto *op = list.pop_back_val();
    for (auto *user : op->getUsers()) {
      if (!seen.insert(user).second)
        continue;
      if (isa<SvSharedOp>(user)) {
        list.push_back(user);
        continue;
      }
      if (isa<LdMatrixOp, CpAsyncGlobalToSharedOp>(user))
        continue;
      if (isSharedMemoryRead(user) || isSharedMemoryWrite(user)) {
        ops.insert(user);
        continue;
      }
    }
  }
}

static SetVector<SwizzlingLayoutAttr>
intersect(const SetVector<SwizzlingLayoutAttr> &lhsSet,
          const SetVector<SwizzlingLayoutAttr> &rhsSet) {
  SetVector<SwizzlingLayoutAttr> resultSet;
  for (auto lhs : lhsSet) {
    if (lhs.isDynamicParams()) {
      for (auto rhs : rhsSet)
        resultSet.insert(rhs);
    } else if (rhsSet.contains(lhs)) {
      resultSet.insert(lhs);
    }
  }
  return resultSet;
}

LogicalResult LayoutOptimization::resolve(ResolveState &curState) {
  funcOp.walk([&](Operation *op) {
    if ((willPassLayout(op) || isa<TransposeOp, ChangeOp>(op))) {
      assert(op->getNumOperands() >= 1 && op->getNumResults() == 1);
      auto &operand = op->getOpOperand(0);
      auto result = op->getResult(0);
      if (curState.valueToLayout.contains(result))
        return;
      auto tensorType = dyn_cast<RankedTensorType>(operand.get().getType());
      if (!tensorType || !inRegisterFile(tensorType))
        return;
      FragmentsLayoutAttr layout;
      if (curState.operandToLayout.contains(&operand))
        layout = curState.operandToLayout[&operand];
      else if (curState.valueToLayout.contains(operand.get()))
        layout = curState.valueToLayout[operand.get()];
      else
        layout = getLayout<FragmentsLayoutAttr>(tensorType);
      if (isa<TransposeOp>(op))
        layout = layout.transpose();
      curState.valueToLayout[result] = layout;
    }
    // Other operations should have been processed.
  });

  bool noBankConflict = true;
  funcOp.walk([&](MkSharedOp op) {
    auto shared = op.getResult();
    auto sharedType = shared.getType();
    auto sharedLayout = getLayout<SwizzlingLayoutAttr>(sharedType);
    curState.sharedToLayouts[shared].insert(sharedLayout);
    SetVector<Operation *> ops;
    collectAccessOps(op, ops);
    for (auto *op : ops) {
      if (isSharedMemoryRead(op)) {
        auto result = op->getResult(0);
        auto resultType = cast<RankedTensorType>(result.getType());
        auto resultLayout = getLayout<FragmentsLayoutAttr>(resultType);
        if (curState.valueToLayout.contains(result))
          resultLayout = curState.valueToLayout[result];
        resultType = cloneWithLayout(resultType, resultLayout);
        curState.sharedToLayouts[shared] = intersect(
            curState.sharedToLayouts[shared],
            getSwizzlingLayouts(sharedType, resultType, getAlignment(op)));
        continue;
      }
      if (isSharedMemoryWrite(op)) {
        auto source = op->getOperand(0);
        auto sourceType = cast<RankedTensorType>(source.getType());
        auto sourceLayout = getLayout<FragmentsLayoutAttr>(sourceType);
        if (curState.valueToLayout.contains(source))
          sourceLayout = curState.valueToLayout[source];
        sourceType = cloneWithLayout(sourceType, sourceLayout);
        curState.sharedToLayouts[shared] = intersect(
            curState.sharedToLayouts[shared],
            getSwizzlingLayouts(sharedType, sourceType, getAlignment(op)));
        continue;
      }
    }
    if (curState.sharedToLayouts[shared].empty())
      noBankConflict = false;
  });
  return success(noBankConflict);
}

llvm::MapVector<Value, SetVector<FragmentsLayoutAttr>>
LayoutOptimization::collectValuesNeedChange(ResolveState &curState) {
  llvm::MapVector<Value, SetVector<FragmentsLayoutAttr>> valuesNeedChange;
  for (auto [operand, layout] : curState.operandToLayout)
    valuesNeedChange[operand->get()].insert(layout);
  return valuesNeedChange;
}

void LayoutOptimization::computeCost(ResolveState &curState) {
  auto valuesNeedChange = collectValuesNeedChange(curState);
  for (const auto &it : valuesNeedChange) {
    auto value = it.first;
    auto sourceType = cast<RankedTensorType>(value.getType());
    if (curState.valueToLayout.contains(value)) {
      auto layout = curState.valueToLayout[value];
      sourceType = cloneWithLayout(sourceType, layout);
    }
    if (auto *defOp = value.getDefiningOp()) {
      if (isa<arith::ConstantOp, SplatOp>(defOp))
        continue;
      auto *block = defOp->getBlock();
      for (auto layout : it.second) {
        auto resultType = cloneWithLayout(sourceType, layout);
        ChangeOpHelper helper(sourceType, resultType);
        curState.cost.blockToNumShfls[block] += helper.getNumShfls();
      }
      continue;
    }
    if (auto blockArg = dyn_cast<BlockArgument>(value)) {
      auto *block = blockArg.getOwner();
      for (auto layout : valuesNeedChange[value]) {
        auto resultType = cloneWithLayout(sourceType, layout);
        ChangeOpHelper helper(sourceType, resultType);
        curState.cost.blockToNumShfls[block] += helper.getNumShfls();
      }
      continue;
    }
  }

  funcOp.walk([&](ReduceOp op) {
    auto *block = op->getBlock();
    auto source = op.getSource();
    auto sourceType = cast<RankedTensorType>(source.getType());
    if (curState.valueToLayout.contains(source)) {
      auto layout = curState.valueToLayout[source];
      sourceType = cloneWithLayout(sourceType, layout);
    }
    ReduceOpHelper helper(sourceType, op.getAxis());
    curState.cost.blockToNumShfls[block] += helper.getNumShfls();
  });
}

void LayoutOptimization::insertChangeOps() {
  OpBuilder builder(funcOp);
  auto valuesNeedChange = collectValuesNeedChange(minState);
  DenseMap<std::pair<Value, FragmentsLayoutAttr>, Value> valueMapping;
  for (const auto &it : valuesNeedChange) {
    auto value = it.first;
    builder.setInsertionPointAfterValue(value);
    auto tensorType = cast<RankedTensorType>(value.getType());
    for (auto layout : it.second) {
      tensorType = cloneWithLayout(tensorType, layout);
      valueMapping[{value, layout}] =
          builder.create<ChangeOp>(value.getLoc(), tensorType, value);
    }
  }
  for (auto [operand, layout] : minState.operandToLayout)
    operandMapping[operand] = valueMapping[{operand->get(), layout}];
}

void LayoutOptimization::rewrite(Operation *op) {
  if (auto forOp = dyn_cast<scf::ForOp>(op))
    return rewrite(forOp);
  if (auto whileOp = dyn_cast<scf::WhileOp>(op))
    return rewrite(whileOp);
  if (auto ifOp = dyn_cast<scf::IfOp>(op))
    return rewrite(ifOp);
  if (auto constantOp = dyn_cast<arith::ConstantOp>(op))
    return rewrite(constantOp);

  setOperands(op);
  for (auto result : op->getResults()) {
    auto resultType = dyn_cast<RankedTensorType>(result.getType());
    if (!resultType || !inRegisterFile(resultType))
      continue;
    auto layout = getLayout<FragmentsLayoutAttr>(resultType);
    if (minState.valueToLayout.contains(result))
      layout = minState.valueToLayout[result];
    resultType = cloneWithLayout(resultType, layout);
    result.setType(resultType);
  }
}

void LayoutOptimization::rewrite(scf::ForOp op) {
  setOperands(op);
  SmallVector<Type> resultTypes;
  for (auto result : op.getResults()) {
    auto resultType = dyn_cast<RankedTensorType>(result.getType());
    if (!resultType || !inRegisterFile(resultType)) {
      resultTypes.push_back(resultType);
      continue;
    }
    auto layout = getLayout<FragmentsLayoutAttr>(resultType);
    if (minState.valueToLayout.contains(result))
      layout = minState.valueToLayout[result];
    resultTypes.push_back(cloneWithLayout(resultType, layout));
  }
  for (auto [index, iterArg] : llvm::enumerate(op.getRegionIterArgs()))
    iterArg.setType(resultTypes[index]);
  for (auto [index, result] : llvm::enumerate(op.getResults()))
    result.setType(resultTypes[index]);
}

void LayoutOptimization::rewrite(scf::WhileOp op) {
  setOperands(op);
  for (auto [index, beforeArg] : llvm::enumerate(op.getBeforeArguments()))
    beforeArg.setType(op->getOperand(index).getType());
  SmallVector<Type> resultTypes;
  for (auto result : op.getResults()) {
    auto resultType = dyn_cast<RankedTensorType>(result.getType());
    if (!resultType || !inRegisterFile(resultType)) {
      resultTypes.push_back(resultType);
      continue;
    }
    auto layout = getLayout<FragmentsLayoutAttr>(resultType);
    if (minState.valueToLayout.contains(result))
      layout = minState.valueToLayout[result];
    resultTypes.push_back(cloneWithLayout(resultType, layout));
  }
  for (auto [index, afterArg] : llvm::enumerate(op.getAfterArguments()))
    afterArg.setType(resultTypes[index]);
  for (auto [index, result] : llvm::enumerate(op.getResults()))
    result.setType(resultTypes[index]);
}

void LayoutOptimization::rewrite(scf::IfOp op) {
  for (auto result : op.getResults()) {
    auto resultType = dyn_cast<RankedTensorType>(result.getType());
    if (!resultType || !inRegisterFile(resultType))
      continue;
    auto layout = getLayout<FragmentsLayoutAttr>(resultType);
    if (minState.valueToLayout.contains(result))
      layout = minState.valueToLayout[result];
    resultType = cloneWithLayout(resultType, layout);
    result.setType(resultType);
  }
}

void LayoutOptimization::rewrite(arith::ConstantOp op) {
  auto result = op.getResult();
  auto resultType = cast<RankedTensorType>(result.getType());
  auto splatAttr = cast<SplatElementsAttr>(op.getValue());
  resultType = cloneWithLayout(resultType, minState.valueToLayout[result]);
  op.setValueAttr(
      SplatElementsAttr::get(resultType, splatAttr.getSplatValue<Attribute>()));
  result.setType(resultType);
}

void LayoutOptimization::setOperands(Operation *op) {
  for (auto &operand : op->getOpOperands())
    if (operandMapping.contains(&operand))
      op->setOperand(operand.getOperandNumber(), operandMapping[&operand]);
}

namespace {

#define GEN_PASS_DEF_KGPUOPTIMIZELAYOUT
#include "kapy/Dialect/Kgpu/Transforms/Passes.h.inc"

class KgpuOptimizeLayoutPass
    : public impl::KgpuOptimizeLayoutBase<KgpuOptimizeLayoutPass> {
public:
  virtual void runOnOperation() override {
    auto module = getOperation();
    module.walk([](FuncOp funcOp) {
      LayoutOptimization optimization(funcOp);
      optimization.initialize();
      optimization.propagate();
      optimization.resolve();
      optimization.rewrite();
    });

    // Currently we assume all the functions are inlined.
    // TODO: Deal with CallOp and FuncOp.

    auto *context = &getContext();
    RewritePatternSet patterns(context);
    ChangeOp::getCanonicalizationPatterns(patterns, context);

    if (failed(applyPatternsAndFoldGreedily(module, std::move(patterns))))
      signalPassFailure();
  }
};

} // namespace

std::unique_ptr<Pass> kapy::createKgpuOptimizeLayoutPass() {
  return std::make_unique<KgpuOptimizeLayoutPass>();
}
