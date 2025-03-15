//===- CoalesceAccess.cpp ---------------------------------------*- C++ -*-===//
//
// This file implements the KgpuCoalesceAccessPass.
//
//===----------------------------------------------------------------------===//

#include "kapy/Dialect/Kapy/IR/Kapy.h"
#include "kapy/Dialect/Kapy/Transforms/TransformUtils.h"
#include "kapy/Dialect/Kgpu/IR/Kgpu.h"
#include "kapy/Dialect/Kgpu/Transforms/Passes.h"
#include "kapy/Support/CommonUtils.h"
#include "kapy/Support/LayoutUtils.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

using namespace mlir;
using namespace mlir::kapy;

static int64_t getGlobalSimdSize(OpOperand *global) {
  auto alignment = getAlignment(global->getOwner());
  auto globalType = cast<RankedTensorType>(global->get().getType());
  auto bitWidth = getIntOrFloatBitWidth(globalType);
  auto simdSize = std::min<int64_t>(alignment * 8 / bitWidth, 128 / bitWidth);
  auto numElems = globalType.getNumElements();
  simdSize = std::min<int64_t>(simdSize, ceilDiv(numElems, warpSize));
  simdSize = std::max<int64_t>(simdSize, 32 / bitWidth);
  return simdSize;
}

static FragmentsLayoutAttr getGlobalAccessLayout(OpOperand *global) {
  auto globalType = cast<RankedTensorType>(global->get().getType());
  auto globalLayout = getLayout<Strided2dLayoutAttr>(globalType);
  // Initialize major axis as 2, that means no contiguous axis.
  unsigned j = 2;
  if (globalLayout.getStride0() == 1)
    j = 0;
  if (globalLayout.getStride1() == 1)
    j = 1;
  // Currently we assume that must have a contiguous axis.
  if (j == 2)
    llvm_unreachable("can not find a contiguous axis");
  auto simdSize = getGlobalSimdSize(global);
  SmallVector<int64_t, 2> laneLoops{1, 1};
  laneLoops[j] = simdSize;
  return getFragmentsLayout(laneLoops, globalType, j == 1);
}

static int64_t getSharedSimdSize(OpOperand *shared) {
  auto alignment = getAlignment(shared->getOwner());
  auto sharedType = cast<RankedTensorType>(shared->get().getType());
  auto bitWidth = getIntOrFloatBitWidth(sharedType);
  auto simdSize = std::min<int64_t>(alignment * 8 / bitWidth, 128 / bitWidth);
  auto numElems = sharedType.getNumElements();
  simdSize = std::min<int64_t>(simdSize, ceilDiv(numElems, warpSize));
  simdSize = std::max<int64_t>(simdSize, 32 / bitWidth);
  return simdSize;
}

static FragmentsLayoutAttr getSharedAccessLayout(OpOperand *shared) {
  auto sharedType = cast<RankedTensorType>(shared->get().getType());
  auto sharedLayout = getLayout<SwizzlingLayoutAttr>(sharedType);
  // Initialize major axis as 2, that means no contiguous axis.
  unsigned j = 2;
  if (sharedLayout.getStride0() == 1)
    j = 0;
  if (sharedLayout.getStride1() == 1)
    j = 1;
  // Currently we assume that must have a contiguous axis.
  if (j == 2)
    llvm_unreachable("can not find a contiguous axis");
  auto simdSize = getSharedSimdSize(shared);
  SmallVector<int64_t, 2> laneLoops{1, 1};
  laneLoops[j] = simdSize;
  return getFragmentsLayout(laneLoops, sharedType, j == 1);
}

namespace {

#define GEN_PASS_DEF_KGPUCOALESCEACCESS
#include "kapy/Dialect/Kgpu/Transforms/Passes.h.inc"

class KgpuCoalesceAccessPass
    : public impl::KgpuCoalesceAccessBase<KgpuCoalesceAccessPass> {
public:
  virtual void runOnOperation() override {
    processGlobalAccessOps();
    processSharedAccessOps();

    if (failed(checkMemoryAccessOps()))
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

  LogicalResult checkMemoryAccessOps();

  /// Coalesce the given memory access operation.
  void coalesceOp(Operation *op, FragmentsLayoutAttr layout);
};

void KgpuCoalesceAccessPass::processGlobalAccessOps() {
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

void KgpuCoalesceAccessPass::processSharedAccessOps() {
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

static LogicalResult checkContiguity(CpAsyncGlobalToSharedOp op) {
  auto sourceType = op.getSource().getType();
  auto targetType = op.getTarget().getType();
  auto sourceLayout = getLayout<Strided2dLayoutAttr>(sourceType);
  auto targetLayout = getLayout<SwizzlingLayoutAttr>(targetType);
  if (sourceLayout.isRowMajor() != targetLayout.isRowMajor())
    return op->emitOpError("has incompatible source and target layout, "
                           "consider use another shared tensor for it");
  return success();
}

static LogicalResult checkAlignment(Operation *op) {
  auto alignment = getAlignment(op);
  if (alignment < 4)
    return op->emitOpError(
        "has memory access with alignment < 4, that is not supported");
  if (isa<LdMatrixOp>(op) && alignment < 16)
    return op->emitOpError(
        "has memory access with alignment < 16, that is not supported");
  return success();
}

LogicalResult KgpuCoalesceAccessPass::checkMemoryAccessOps() {
  bool noInvalid = true;
  auto module = getOperation();
  module.walk([&](Operation *op) {
    if (auto cpAsyncOp = dyn_cast<CpAsyncGlobalToSharedOp>(op)) {
      if (failed(checkContiguity(cpAsyncOp)))
        noInvalid = false;
    }
    if (isGlobalMemoryRead(op) || isGlobalMemoryWrite(op) ||
        isSharedMemoryRead(op) || isSharedMemoryWrite(op)) {
      if (failed(checkAlignment(op)))
        noInvalid = false;
    }
  });
  return success(noInvalid);
}

void KgpuCoalesceAccessPass::coalesceOp(Operation *op,
                                        FragmentsLayoutAttr layout) {
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

std::unique_ptr<Pass> kapy::createKgpuCoalesceAccessPass() {
  return std::make_unique<KgpuCoalesceAccessPass>();
}
