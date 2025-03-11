//===- AnalyzeAlignment.cpp -------------------------------------*- C++ -*-===//
//
// This file implements the KapyAnalyzePass.
//
//===----------------------------------------------------------------------===//

#include "kapy/Analysis/AlignAnalysis.h"
#include "kapy/Analysis/AllocAnalysis.h"
#include "kapy/Dialect/Kapy/IR/Kapy.h"
#include "kapy/Dialect/Kapy/Transforms/Passes.h"
#include "kapy/Dialect/Kapy/Transforms/TransformUtils.h"

using namespace mlir;
using namespace mlir::kapy;

namespace {

#define GEN_PASS_DEF_KAPYANALYZE
#include "kapy/Dialect/Kapy/Transforms/Passes.h.inc"

class KapyAnalyzePass : public impl::KapyAnalyzeBase<KapyAnalyzePass> {
public:
  virtual void runOnOperation() override {
    setGlobalMemoryLayouts();

    auto module = getOperation();
    ModuleAlignAnalysis alignAnalysis(module);
    module.walk([&](Operation *op) {
      if (auto effectOp = dyn_cast<MemoryEffectOpInterface>(op))
        setAlignment(effectOp, alignAnalysis);
    });
    ModuleAllocAnalysis allocAnalysis(module);
    module.walk([&](FunctionOpInterface funcOp) {
      funcOp.walk([&](Operation *op) {
        setOffset(op, *allocAnalysis.getData(funcOp));
      });
    });
    auto size = allocAnalysis.getAllocatedSize();
    auto i64Type = IntegerType::get(&getContext(), 64);
    module->setAttr("kapy.size", IntegerAttr::get(i64Type, size));
  }

private:
  /// Set global memory layouts with known information.
  void setGlobalMemoryLayouts();

  /// Set alignment attribute for operations access global or shared memory.
  void setAlignment(MemoryEffectOpInterface op, ModuleAlignAnalysis &analysis);

  /// Set offset attribute of shared memory.
  void setOffset(Operation *op, const AllocInfo &info);
};

void KapyAnalyzePass::setGlobalMemoryLayouts() {
  auto module = getOperation();
  module.walk([](MkGlobalOp op) {
    SmallVector<int64_t, 2> strides;
    for (auto stride : {op.getStride0(), op.getStride1()}) {
      auto constantOp = stride.getDefiningOp<arith::ConstantOp>();
      if (!constantOp) {
        strides.push_back(ShapedType::kDynamic);
        continue;
      }
      auto intAttr = dyn_cast<IntegerAttr>(constantOp.getValue());
      if (!intAttr) {
        strides.push_back(ShapedType::kDynamic);
        continue;
      }
      strides.push_back(intAttr.getInt());
    }
    auto *context = op.getContext();
    auto layout = Strided2dLayoutAttr::get(context, strides[0], strides[1]);
    DenseSet<Value> seen;
    propagateMemoryLayout(op.getResult(), layout, seen);
  });
}

void KapyAnalyzePass::setAlignment(MemoryEffectOpInterface op,
                                   ModuleAlignAnalysis &analysis) {
  SmallVector<SideEffects::EffectInstance<MemoryEffects::Effect>> effects;
  op.getEffects(effects);
  int64_t alignment = 0;
  for (auto &effect : effects) {
    if (effect.getResource() != GlobalMemory::get() &&
        effect.getResource() != SharedMemory::get())
      continue;
    if (isa<MemoryEffects::Read, MemoryEffects::Write>(effect.getEffect())) {
      auto funcOp = op->getParentOfType<FunctionOpInterface>();
      auto info = analysis.getData(funcOp)->lookup(effect.getValue());
      if (alignment == 0)
        alignment = info.getAlignment();
      else
        alignment = std::min(alignment, info.getAlignment());
    }
  }
  if (alignment != 0) {
    auto i64Type = IntegerType::get(&getContext(), 64);
    op->setAttr("kapy.alignment", IntegerAttr::get(i64Type, alignment));
  }
}

void KapyAnalyzePass::setOffset(Operation *op, const AllocInfo &info) {
  auto id = info.getBufferId(op);
  int64_t offset = -1;
  if (id != AllocInfo::INVALID_ID) {
    offset = info.getOffset(id);
  } else if (op->getNumResults() == 1) {
    auto result = op->getResult(0);
    auto id = info.getBufferId(result);
    if (id != AllocInfo::INVALID_ID)
      offset = info.getOffset(id);
  }
  if (offset != -1) {
    auto i64Type = IntegerType::get(&getContext(), 64);
    op->setAttr("kapy.offset", IntegerAttr::get(i64Type, offset));
  }
}

} // namespace

std::unique_ptr<Pass> kapy::createKapyAnalyzePass() {
  return std::make_unique<KapyAnalyzePass>();
}
