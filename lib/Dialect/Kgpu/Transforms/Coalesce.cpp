//===- Coalesce.cpp ---------------------------------------------*- C++ -*-===//
//
// This file implements the KgpuCoalescePass.
//
//===----------------------------------------------------------------------===//

#include "kapy/Analysis/AnalysisUtils.h"
#include "kapy/Dialect/Kapy/IR/Kapy.h"
#include "kapy/Dialect/Kapy/Transforms/TransformUtils.h"
#include "kapy/Dialect/Kgpu/IR/Kgpu.h"
#include "kapy/Dialect/Kgpu/Transforms/Passes.h"
#include "kapy/Support/CommonUtils.h"
#include "kapy/Support/LayoutUtils.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

using namespace mlir;
using namespace mlir::kapy;

static int64_t getGlobalVecWidth(OpOperand *global) {
  auto alignment = getAlignment(global->getOwner());
  auto globalType = cast<RankedTensorType>(global->get().getType());
  auto bitWidth = getIntOrFloatBitWidth(globalType);
  auto vecWidth = std::min<int64_t>(alignment * 8 / bitWidth, 128 / bitWidth);
  auto numElems = globalType.getNumElements();
  vecWidth = std::min<int64_t>(vecWidth, ceilDiv(numElems, warpSize));
  vecWidth = std::max<int64_t>(vecWidth, 32 / bitWidth);
  return vecWidth;
}

static FragmentsLayoutAttr getGlobalAccessLayout(OpOperand *global) {
  auto globalType = cast<RankedTensorType>(global->get().getType());
  auto globalLayout = getLayout<Strided2dLayoutAttr>(globalType);
  // Initialize major axis as 2, that means no contiguous axis.
  unsigned j = 2;
  if (globalLayout.getStrideX() == 1)
    j = 0;
  if (globalLayout.getStrideY() == 1)
    j = 1;
  // Currently we assume that must have a contiguous axis.
  if (j == 2)
    llvm_unreachable("can not find a contiguous axis");
  auto vecWidth = getGlobalVecWidth(global);
  SmallVector<int64_t, 2> laneLoops{1, 1};
  laneLoops[j] = vecWidth;
  return getFragmentsLayout(laneLoops, globalType, j == 1);
}

static int64_t getSharedVecWidth(OpOperand *shared) {
  auto alignment = getAlignment(shared->getOwner());
  auto sharedType = cast<RankedTensorType>(shared->get().getType());
  auto bitWidth = getIntOrFloatBitWidth(sharedType);
  auto vecWidth = std::min<int64_t>(alignment * 8 / bitWidth, 128 / bitWidth);
  auto numElems = sharedType.getNumElements();
  vecWidth = std::min<int64_t>(vecWidth, ceilDiv(numElems, warpSize));
  vecWidth = std::max<int64_t>(vecWidth, 32 / bitWidth);
  return vecWidth;
}

static FragmentsLayoutAttr getSharedAccessLayout(OpOperand *shared) {
  auto sharedType = cast<RankedTensorType>(shared->get().getType());
  auto sharedLayout = getLayout<SwizzlingLayoutAttr>(sharedType);
  // Initialize major axis as 2, that means no contiguous axis.
  unsigned j = 2;
  if (sharedLayout.getStrideX() == 1)
    j = 0;
  if (sharedLayout.getStrideY() == 1)
    j = 1;
  // Currently we assume that must have a contiguous axis.
  if (j == 2)
    llvm_unreachable("can not find a contiguous axis");
  auto vecWidth = getSharedVecWidth(shared);
  SmallVector<int64_t, 2> laneLoops{1, 1};
  laneLoops[j] = vecWidth;
  return getFragmentsLayout(laneLoops, sharedType, j == 1);
}

namespace {

#define GEN_PASS_DEF_KGPUCOALESCE
#include "kapy/Dialect/Kgpu/Transforms/Passes.h.inc"

class KgpuCoalescePass : public impl::KgpuCoalesceBase<KgpuCoalescePass> {
public:
  virtual void runOnOperation() override {
    processGlobalAccessOps();
    processSharedAccessOps();

    if (failed(checkCpAsyncGlobalToSharedOps()))
      signalPassFailure();

    auto *context = &getContext();
    RewritePatternSet patterns(context);
    ChangeOp::getCanonicalizationPatterns(patterns, context);

    auto module = getOperation();
    if (failed(applyPatternsAndFoldGreedily(module, std::move(patterns))))
      signalPassFailure();
  }

private:
  /// Process global access operations.
  void processGlobalAccessOps();

  /// Process shared access operations and update corresponding memory layouts.
  void processSharedAccessOps();

  /// Check if CpAsyncGlobalToSharedOp's source layout and target layout are
  /// compatible.
  LogicalResult checkCpAsyncGlobalToSharedOps();

  /// Coalesce the given memory access operation.
  void coalesceOp(Operation *op, FragmentsLayoutAttr layout);
};

void KgpuCoalescePass::processGlobalAccessOps() {
  auto module = getOperation();
  module.walk([&](Operation *op) {
    auto effectOp = dyn_cast<MemoryEffectOpInterface>(op);
    if (!effectOp)
      return;
    SmallVector<SideEffects::EffectInstance<MemoryEffects::Effect>> effects;
    effectOp.getEffects(effects);
    for (auto &effect : effects) {
      if (effect.getResource() == GlobalMemory::get() &&
          isa<MemoryEffects::Read, MemoryEffects::Write>(effect.getEffect())) {
        auto *global = effect.getEffectValue<OpOperand *>();
        coalesceOp(op, getGlobalAccessLayout(global));
        break;
      }
    }
  });
}

void KgpuCoalescePass::processSharedAccessOps() {
  auto module = getOperation();
  module.walk([](LdMatrixOp op) {
    auto *context = op.getContext();
    auto source = op.getSource();
    auto sourceType = source.getType();
    auto sourceLayout = getLayout<SwizzlingLayoutAttr>(sourceType);
    if (sourceLayout.isColMajor()) {
      OpBuilder builder(op);
      auto loc = op.getLoc();
      auto loader = op.getLoader();
      auto loaderType = loader.getType();
      auto loaderLayout = getLayout<FragmentsLayoutAttr>(loaderType);
      loaderType = cloneWithLayout(loaderType, loaderLayout.transpose());
      op.setOperand(1, builder.create<ChangeOp>(loc, loaderType, loader));
    }
    DenseSet<Value> seen;
    propagateMemoryLayout(source, sourceLayout.setParams(4, 8), seen);
  });
  module.walk([&](Operation *op) {
    if (isa<LdMatrixOp, CpAsyncGlobalToSharedOp>(op))
      return;
    auto effectOp = dyn_cast<MemoryEffectOpInterface>(op);
    if (!effectOp)
      return;
    SmallVector<SideEffects::EffectInstance<MemoryEffects::Effect>> effects;
    effectOp.getEffects(effects);
    for (auto &effect : effects) {
      if (effect.getResource() == SharedMemory::get() &&
          isa<MemoryEffects::Read, MemoryEffects::Write>(effect.getEffect())) {
        auto *shared = effect.getEffectValue<OpOperand *>();
        coalesceOp(op, getSharedAccessLayout(shared));
        break;
      }
    }
  });
}

static LogicalResult checkImpl(CpAsyncGlobalToSharedOp op) {
  auto sourceType = op.getSource().getType();
  auto targetType = op.getTarget().getType();
  auto sourceLayout = getLayout<Strided2dLayoutAttr>(sourceType);
  auto targetLayout = getLayout<SwizzlingLayoutAttr>(targetType);
  if (sourceLayout.isRowMajor() != targetLayout.isRowMajor())
    return op->emitOpError("has incompatible source and target layout, "
                           "consider use another shared tensor for it");
  return success();
}

LogicalResult KgpuCoalescePass::checkCpAsyncGlobalToSharedOps() {
  bool noInvalid = true;
  auto module = getOperation();
  module.walk([&](CpAsyncGlobalToSharedOp op) {
    if (failed(checkImpl(op)))
      noInvalid = false;
  });
  return success(noInvalid);
}

void KgpuCoalescePass::coalesceOp(Operation *op, FragmentsLayoutAttr layout) {
  OpBuilder builder(op);
  auto *context = op->getContext();
  auto loc = op->getLoc();

  SmallVector<Value> operands;
  for (auto operand : op->getOperands()) {
    auto tensorType = dyn_cast<RankedTensorType>(operand.getType());
    if (!tensorType || !inRegisterFile(tensorType)) {
      operands.push_back(operand);
    } else {
      tensorType = cloneWithLayout(tensorType, layout);
      operands.push_back(builder.create<ChangeOp>(loc, tensorType, operand));
    }
  }

  SmallVector<Type> types;
  for (auto type : op->getResultTypes()) {
    auto tensorType = dyn_cast<RankedTensorType>(type);
    if (!tensorType || !inRegisterFile(tensorType)) {
      types.push_back(type);
    } else {
      tensorType = cloneWithLayout(tensorType, layout);
      types.push_back(tensorType);
    }
  }

  auto opName = op->getName().getIdentifier();
  auto *newOp = builder.create(loc, opName, operands, types, op->getAttrs());
  for (unsigned i = 0; i < op->getNumResults(); ++i) {
    auto oldResult = op->getResult(i);
    auto newResult = newOp->getResult(i);
    oldResult.replaceAllUsesWith(
        builder.create<ChangeOp>(loc, oldResult.getType(), newResult));
  }
  op->erase();
}

} // namespace

std::unique_ptr<Pass> kapy::createKgpuCoalescePass() {
  return std::make_unique<KgpuCoalescePass>();
}
