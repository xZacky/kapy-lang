//===- AnalyzeAlignment.cpp -------------------------------------*- C++ -*-===//
//
// This file implements the KapyAnalyzeAlignmentPass.
//
//===----------------------------------------------------------------------===//

#include "kapy/Analysis/Integer.h"
#include "kapy/Dialect/Kapy/IR/Kapy.h"
#include "kapy/Dialect/Kapy/Transform/Passes.h"

using namespace mlir;
using namespace mlir::kapy;

namespace {

#define GEN_PASS_DEF_KAPYANALYZEALIGNMENT
#include "kapy/Dialect/Kapy/Transform/Passes.h.inc"

class KapyAnalyzeAlignmentPass
    : public impl::KapyAnalyzeAlignmentBase<KapyAnalyzeAlignmentPass> {
public:
  virtual void runOnOperation() override {
    auto module = getOperation();
    ModuleIntegerInfoAnalysis analysis(module);
    auto i64Type = IntegerType::get(&getContext(), 64);
    module.walk([&](Operation *op) {
      auto effectOp = dyn_cast<MemoryEffectOpInterface>(op);
      if (!effectOp)
        return;
      SmallVector<SideEffects::EffectInstance<MemoryEffects::Effect>> effects;
      effectOp.getEffects(effects);
      for (auto &effect : effects) {
        if (isa<MemoryEffects::Read, MemoryEffects::Write>(
                effect.getEffect()) &&
            effect.getResource() == GlobalMemory::get()) {
          auto funcOp = op->getParentOfType<FunctionOpInterface>();
          auto *valueToInfo = analysis.getData(funcOp);
          auto value = effect.getValue();
          auto alignment = (*valueToInfo)[value].getDivisibility();
          op->setDiscardableAttr(alignmentAttrName,
                                 IntegerAttr::get(i64Type, alignment));
          // For operation has more than one memory effects, we assume that all
          // the effects are applied to the same memref, so we only need to set
          // alignment once.
          break;
        }
      }
    });
  }
};

} // namespace

std::unique_ptr<Pass> kapy::createKapyAnalyzeAlignmentPass() {
  return std::make_unique<KapyAnalyzeAlignmentPass>();
}
