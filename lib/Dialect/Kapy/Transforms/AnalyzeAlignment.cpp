//===- AnalyzeAlignment.cpp -------------------------------------*- C++ -*-===//
//
// This file implements the KapyAnalyzeAlignmentPass.
//
//===----------------------------------------------------------------------===//

#include "kapy/Analysis/AlignAnalysis.h"
#include "kapy/Dialect/Kapy/IR/Kapy.h"
#include "kapy/Dialect/Kapy/Transforms/Passes.h"
#include "kapy/Support/CommonUtils.h"

using namespace mlir;
using namespace mlir::kapy;

/// Implementation of euclidean algorithm.
template <typename T> static T gcdImpl(T a, T b, T &x, T &y) {
  if (a == 0) {
    x = 0;
    y = 1;
    return b;
  }
  T x1, y1; // to store results of recursive call
  T g = gcdImpl(b % a, a, x1, y1);
  // update `x` and `y` using results of recursive call
  x = y1 - (b / a) * x1;
  y = x1;
  return g;
}

/// Greatest common divisor.
template <typename T> static T gcd(T a, T b) {
  static_assert(std::is_integral_v<T>);
  if (a == 0)
    return b;
  if (b == 0)
    return a;
  T x, y;
  return gcdImpl(a, b, x, y);
}

/// Greatest power of two divisor. If `x == 0`, return the greatest power of two
/// for type `I`.
template <typename T> static T gpd(T x) {
  static_assert(std::is_integral_v<T>);
  if (x == 0)
    return static_cast<T>(1) << (sizeof(T) * 8 - 2);
  return x & (~(x - 1));
}

/// If `a * b` overflows, return greatest the power of two for type `I`.
template <typename T> static T mul(T a, T b) {
  static_assert(std::is_integral_v<T>);
  T g = gpd<T>(0);
  return a > g / b ? g : a * b;
}

namespace {

#define GEN_PASS_DEF_KAPYANALYZEALIGNMENT
#include "kapy/Dialect/Kapy/Transforms/Passes.h.inc"

class KapyAnalyzeAlignmentPass
    : public impl::KapyAnalyzeAlignmentBase<KapyAnalyzeAlignmentPass> {
public:
  virtual void runOnOperation() override {
    auto module = getOperation();
    ModuleAlignAnalysis analysis(module);
    module.walk([&](Operation *op) {
      if (auto mkGlobalOp = dyn_cast<MkGlobalOp>(op))
        return processMkGlobalOp(mkGlobalOp, analysis);
      if (auto svGlobalOp = dyn_cast<SvGlobalOp>(op))
        return processSvGlobalOp(svGlobalOp, analysis);
      if (auto effectOp = dyn_cast<MemoryEffectOpInterface>(op))
        return processEffectOp(effectOp);
    });
  }

private:
  void processMkGlobalOp(MkGlobalOp mkGlobalOp, ModuleAlignAnalysis &analysis);

  void processSvGlobalOp(SvGlobalOp svGlobalOp, ModuleAlignAnalysis &analysis);

  void processEffectOp(MemoryEffectOpInterface op);
};

void KapyAnalyzeAlignmentPass::processMkGlobalOp(
    MkGlobalOp mkGlobalOp, ModuleAlignAnalysis &analysis) {
  auto funcOp = mkGlobalOp->getParentOfType<FunctionOpInterface>();
  auto *valueToInfo = analysis.getData(funcOp);
  auto alignment = valueToInfo->lookup(mkGlobalOp.getAddress()).getAlignment();
  auto i64Type = IntegerType::get(&getContext(), 64);
  mkGlobalOp->setAttr(alignmentAttrName, IntegerAttr::get(i64Type, alignment));
}

void KapyAnalyzeAlignmentPass::processSvGlobalOp(
    SvGlobalOp svGlobalOp, ModuleAlignAnalysis &analysis) {
  auto funcOp = svGlobalOp->getParentOfType<FunctionOpInterface>();
  auto *valueToInfo = analysis.getData(funcOp);
  auto mkGlobalOp = svGlobalOp.getSource().getDefiningOp<MkGlobalOp>();
  auto alignment = getAlignment(mkGlobalOp);
  auto alignmentX =
      mul(valueToInfo->lookup(svGlobalOp.getStartX()).getAlignment(),
          valueToInfo->lookup(mkGlobalOp.getStrideX()).getAlignment());
  auto alignmentY =
      mul(valueToInfo->lookup(svGlobalOp.getStartY()).getAlignment(),
          valueToInfo->lookup(mkGlobalOp.getStrideY()).getAlignment());
  auto bitWidth = getIntOrFloatBitWidth(svGlobalOp.getType());
  alignmentX = mul(alignmentX, ceilDiv<int64_t>(bitWidth, 8));
  alignmentY = mul(alignmentY, ceilDiv<int64_t>(bitWidth, 8));
  alignment = gcd(alignment, gcd(alignmentX, alignmentY));
  auto i64Type = IntegerType::get(&getContext(), 64);
  svGlobalOp->setAttr(alignmentAttrName, IntegerAttr::get(i64Type, alignment));
}

void KapyAnalyzeAlignmentPass::processEffectOp(MemoryEffectOpInterface op) {
  SmallVector<SideEffects::EffectInstance<MemoryEffects::Effect>> effects;
  op.getEffects(effects);
  for (auto &effect : effects) {
    if (effect.getResource() == GlobalMemory::get() &&
        isa<MemoryEffects::Read, MemoryEffects::Write>(effect.getEffect())) {
      auto alignment = getAlignment(effect.getValue().getDefiningOp());
      auto i64Type = IntegerType::get(&getContext(), 64);
      op->setAttr(alignmentAttrName, IntegerAttr::get(i64Type, alignment));
      break;
    }
  }
}

} // namespace

std::unique_ptr<Pass> kapy::createKapyAnalyzeAlignmentPass() {
  return std::make_unique<KapyAnalyzeAlignmentPass>();
}
