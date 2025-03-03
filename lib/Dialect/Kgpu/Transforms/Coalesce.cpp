//===- Coalesce.cpp ---------------------------------------------*- C++ -*-===//
//
// This file implements the KgpuCoalescePass.
//
//===----------------------------------------------------------------------===//

#include "kapy/Analysis/AnalysisUtils.h"
#include "kapy/Analysis/LayoutUtils.h"
#include "kapy/Dialect/Kapy/IR/Kapy.h"
#include "kapy/Dialect/Kgpu/IR/Kgpu.h"
#include "kapy/Dialect/Kgpu/Transforms/Passes.h"
#include "kapy/Dialect/Kgpu/Transforms/TransformUtils.h"
#include "kapy/Support/CommonUtils.h"

using namespace mlir;
using namespace mlir::kapy;

static int64_t getVecWidth(Operation *op, RankedTensorType globalType) {
  auto alignment = getAlignment(op);
  auto bitWidth = getIntOrFloatBitWidth(globalType);
  auto vecWidth = std::min<int64_t>(alignment * 8 / bitWidth, 128 / bitWidth);
  auto numElems = globalType.getNumElements();
  vecWidth = std::min<int64_t>(vecWidth, ceilDiv(numElems, warpSize));
  vecWidth = std::max<int64_t>(vecWidth, 1);
  return vecWidth;
}

static FragmentsLayoutAttr getGlobalAccessLayout(Operation *op,
                                                 RankedTensorType globalType) {
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
  auto vecWidth = getVecWidth(op, globalType);
  SmallVector<int64_t, 2> laneLoops{1, 1};
  laneLoops[j] = vecWidth;
  return getFragmentsLayout(laneLoops, globalType, j == 1);
}

static FragmentsLayoutAttr getSharedAccessLayout(RankedTensorType sharedType) {
  unsigned j = 1;
  if (hasLayout(sharedType)) {
    auto sharedLayout = getLayout<SwizzlingLayoutAttr>(sharedType);
    j = sharedLayout.getMajorAxis();
  }
  auto vecWidth = 128 / getIntOrFloatBitWidth(sharedType);
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
    setGlobalMemoryLayouts();
    processGlobalAccessOps();
    setSharedMemoryLayouts();
    processSharedAccessOps();
    if (failed(checkCpAsyncGlobalToSharedOps()))
      signalPassFailure();
  }

private:
  /// Set global memory layouts with known information.
  void setGlobalMemoryLayouts();

  /// Process global access operations.
  void processGlobalAccessOps();

  /// Set shared memory layouts by the CpAsyncGlobalToSharedOps.
  void setSharedMemoryLayouts();

  /// Process LdMatrixOps and update corresponding shared memory layouts.
  void processSharedAccessOps();

  /// Check if CpAsyncGlobalToSharedOp's source layout and target layout are
  /// compatible.
  LogicalResult checkCpAsyncGlobalToSharedOps();

  /// Coalesce the given memory access operation.
  void coalesceOp(Operation *op, FragmentsLayoutAttr layout);
};

void KgpuCoalescePass::setGlobalMemoryLayouts() {
  auto module = getOperation();
  module.walk([](MkGlobalOp op) {
    SmallVector<int64_t, 2> shape;
    for (auto size : {op.getSizeX(), op.getSizeY()}) {
      auto constantOp = size.getDefiningOp<arith::ConstantOp>();
      if (!constantOp) {
        shape.push_back(ShapedType::kDynamic);
        continue;
      }
      auto intAttr = dyn_cast<IntegerAttr>(constantOp.getValue());
      if (!intAttr) {
        shape.push_back(ShapedType::kDynamic);
        continue;
      }
      shape.push_back(intAttr.getInt());
    }
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
        auto globalType = cast<RankedTensorType>(effect.getValue().getType());
        coalesceOp(op, getGlobalAccessLayout(op, globalType));
        break;
      }
    }
  });
}

void KgpuCoalescePass::setSharedMemoryLayouts() {
  auto module = getOperation();
  module.walk([](MkSharedOp op) {
    if (hasLayout(op.getResult().getType()))
      return;
    auto slice = multiRootGetSlice(op);
    bool rowMajor = true;
    for (auto *op : slice) {
      if (auto cpAsyncOp = dyn_cast<CpAsyncGlobalToSharedOp>(op)) {
        auto type = cast<RankedTensorType>(cpAsyncOp.getSource().getType());
        auto layout = getLayout<Strided2dLayoutAttr>(type);
        rowMajor = layout.isRowMajor();
        break;
      }
    }
    auto *context = op.getContext();
    auto dynamic = ShapedType::kDynamic;
    auto layout = SwizzlingLayoutAttr::get(context, dynamic, dynamic, rowMajor);
    DenseSet<Value> seen;
    propagateMemoryLayout(op.getResult(), layout, seen);
  });
}

void KgpuCoalescePass::processSharedAccessOps() {
  auto module = getOperation();
  module.walk([](LdMatrixOp op) {
    auto *context = op.getContext();
    auto source = op.getSource();
    auto sourceType = source.getType();
    if (hasLayout(sourceType)) {
      if (getLayout<SwizzlingLayoutAttr>(sourceType).isColMajor()) {
        OpBuilder builder(op);
        auto loc = op.getLoc();
        auto loader = op.getLoader();
        auto loaderType = loader.getType();
        auto layout = getLayout<FragmentsLayoutAttr>(loaderType).transpose();
        loaderType = cloneWithLayout(loaderType, layout);
        op.setOperand(1, builder.create<ChangeOp>(loc, loaderType, loader));
        DenseSet<Value> seen;
        propagateMemoryLayout(
            source, SwizzlingLayoutAttr::get(context, 4, 8, 1, 0), seen);
      } else {
        DenseSet<Value> seen;
        propagateMemoryLayout(
            source, SwizzlingLayoutAttr::get(context, 4, 8, 0, 1), seen);
      }
    }
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
        auto sharedType = cast<RankedTensorType>(effect.getValue().getType());
        coalesceOp(op, getSharedAccessLayout(sharedType));
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
    return op->emitError("has incompatible source and target layout, "
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
