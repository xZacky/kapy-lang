//===- AnalyzeAlignment.cpp -------------------------------------*- C++ -*-===//
//
// This file implements the KapyAnalyzeAlignmentPass.
//
//===----------------------------------------------------------------------===//

#include "kapy/Analysis/AlignAnalysis.h"
#include "kapy/Dialect/Kapy/IR/Kapy.h"
#include "kapy/Dialect/Kapy/Transforms/Passes.h"
#include "kapy/Dialect/Kapy/Transforms/TransformUtils.h"

using namespace mlir;
using namespace mlir::kapy;

namespace {

#define GEN_PASS_DEF_KAPYANALYZEALIGNMENT
#include "kapy/Dialect/Kapy/Transforms/Passes.h.inc"

class KapyAnalyzeAlignmentPass
    : public impl::KapyAnalyzeAlignmentBase<KapyAnalyzeAlignmentPass> {
public:
  virtual void runOnOperation() override {
    setGlobalMemoryLayouts();
    setSharedMemoryLayouts();
    auto module = getOperation();
    ModuleAlignAnalysis analysis(module);
    module.walk([&](Operation *op) {
      if (auto effectOp = dyn_cast<MemoryEffectOpInterface>(op))
        processEffectOp(effectOp, analysis);
    });
  }

private:
  /// Set global memory layouts with known information.
  void setGlobalMemoryLayouts();

  /// Set shared memory layouts with known information.
  void setSharedMemoryLayouts();

  void processEffectOp(MemoryEffectOpInterface op,
                       ModuleAlignAnalysis &analysis);
};

void KapyAnalyzeAlignmentPass::setGlobalMemoryLayouts() {
  auto module = getOperation();
  module.walk([](MkGlobalOp op) {
    SmallVector<int64_t, 2> strides;
    for (auto stride : {op.getStrideX(), op.getStrideY()}) {
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

void KapyAnalyzeAlignmentPass::setSharedMemoryLayouts() {
  auto module = getOperation();
  module.walk([](MkSharedOp op) {
    auto strideX = op.getStrideX();
    auto strideY = op.getStrideY();
    auto dynamic = ShapedType::kDynamic;
    auto *context = op.getContext();
    auto layout =
        SwizzlingLayoutAttr::get(context, strideX, strideY, dynamic, dynamic);
    DenseSet<Value> seen;
    propagateMemoryLayout(op.getResult(), layout, seen);
  });
}

void KapyAnalyzeAlignmentPass::processEffectOp(MemoryEffectOpInterface op,
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
    op->setAttr(alignmentAttrName, IntegerAttr::get(i64Type, alignment));
  }
}

} // namespace

std::unique_ptr<Pass> kapy::createKapyAnalyzeAlignmentPass() {
  return std::make_unique<KapyAnalyzeAlignmentPass>();
}
