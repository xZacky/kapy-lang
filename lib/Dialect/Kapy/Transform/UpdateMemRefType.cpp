//===- UpdateMemRefType.cpp -------------------------------------*- C++ -*-===//
//
// This file implements the KapyUpdateMemRefTypePass.
//
//===----------------------------------------------------------------------===//

#include "kapy/Dialect/Kapy/IR/Kapy.h"
#include "kapy/Dialect/Kapy/Transform/Passes.h"

using namespace mlir;
using namespace mlir::kapy;

static void propagateNewType(Value value, Type newType, DenseSet<Value> &seen) {
  if (!seen.insert(value).second)
    return;

  value.setType(newType);
  for (auto &use : value.getUses()) {
    auto *useOp = use.getOwner();
    auto useIndex = use.getOperandNumber();
    if (auto forOp = dyn_cast<scf::ForOp>(useOp)) {
      auto iterArg = forOp.getTiedLoopRegionIterArg(&use);
      propagateNewType(iterArg, newType, seen);
      auto result = forOp.getTiedLoopResult(&use);
      propagateNewType(result, newType, seen);
      continue;
    }
    if (auto whileOp = dyn_cast<scf::WhileOp>(useOp)) {
      auto beforeArg = whileOp.getBeforeArguments()[useIndex];
      propagateNewType(beforeArg, newType, seen);
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
      for (auto value : values)
        propagateNewType(value, newType, seen);
    }
    if (auto conditionOp = dyn_cast<scf::ConditionOp>(useOp)) {
      auto whileOp = cast<scf::WhileOp>(conditionOp->getParentOp());
      // Skip argument 0 as it is the condition.
      auto argIndex = useIndex - 1;
      auto afterArg = whileOp.getAfterArguments()[argIndex];
      propagateNewType(afterArg, newType, seen);
      auto result = whileOp->getResult(argIndex);
      propagateNewType(result, newType, seen);
      continue;
    }
    if (auto moveOp = dyn_cast<MoveMemRefOp>(useOp)) {
      propagateNewType(moveOp.getResult(), newType, seen);
      continue;
    }
  }
}

namespace {

#define GEN_PASS_DEF_KAPYUPDATEMEMREFTYPE
#include "kapy/Dialect/Kapy/Transform/Passes.h.inc"

class KapyUpdateMemRefTypePass
    : public impl::KapyUpdateMemRefTypeBase<KapyUpdateMemRefTypePass> {
public:
  virtual void runOnOperation() override {
    auto module = getOperation();

    SmallVector<MakeMemRefOp> makeOps;
    module.walk([&](MakeMemRefOp getOp) { makeOps.push_back(getOp); });

    for (auto makeOp : makeOps) {
      auto memrefType = makeOp.getType();
      auto glmemLayout = cast<GlobalMemLayoutAttr>(memrefType.getEncoding());
      auto strides = llvm::to_vector<4>(glmemLayout.getStrides());
      bool modified = false;
      for (auto it : llvm::enumerate(makeOp.getStrides())) {
        if (!ShapedType::isDynamic(strides[it.index()]))
          continue;
        auto constantOp = it.value().getDefiningOp<arith::ConstantOp>();
        if (!constantOp)
          continue;
        auto intAttr = dyn_cast<IntegerAttr>(constantOp.getValue());
        if (!intAttr)
          continue;
        strides[it.index()] = intAttr.getInt();
        modified = true;
      }
      if (modified) {
        glmemLayout = GlobalMemLayoutAttr::get(makeOp.getContext(), strides);
        memrefType = cloneWith(memrefType, glmemLayout);
        DenseSet<Value> seen;
        propagateNewType(makeOp.getResult(), memrefType, seen);
      }
    }
  }
};

} // namespace

std::unique_ptr<Pass> kapy::createKapyUpdateMemRefTypePass() {
  return std::make_unique<KapyUpdateMemRefTypePass>();
}
