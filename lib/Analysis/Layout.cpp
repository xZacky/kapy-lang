//===- Layout.cpp -----------------------------------------------*- C++ -*-===//
//
// This file implements functions for optimizations of layouts.
//
//===----------------------------------------------------------------------===//

#include "kapy/Analysis/Layout.h"
#include "kapy/Analysis/Utils.h"
#include "kapy/Dialect/Kapy/IR/Kapy.h"
#include "kapy/Dialect/Kgpu/IR/Kgpu.h"

using namespace mlir;
using namespace kapy;

RegistersLayoutAttr kapy::getRegistersLayout(MLIRContext *context,
                                             ArrayRef<int64_t> loopsPerWarp,
                                             ArrayRef<int64_t> loopsPerLane,
                                             ArrayRef<int64_t> shape,
                                             int64_t numWarps) {
  auto rank = shape.size();
  assert(rank != 0);
  SmallVector<int64_t, 4> shapeOfLanes(rank);
  SmallVector<int64_t, 4> shapeOfWarps(rank);
  auto restThreads = numLanes * numWarps;
  auto restLanes = numLanes;
  auto restWarps = numWarps;
  for (unsigned i = rank - 1; i >= 1; --i) {
    auto numLoopsI = loopsPerLane[i] * loopsPerWarp[i];
    auto numThreadsI = std::clamp(restThreads, 1L, shape[i] / numLoopsI);
    shapeOfLanes[i] = std::clamp(numThreadsI, 1L, restLanes);
    shapeOfWarps[i] = std::clamp(numThreadsI / shapeOfLanes[i], 1L, restWarps);
    restThreads /= (shapeOfLanes[i] * shapeOfWarps[i]);
    restLanes /= shapeOfLanes[i];
    restWarps /= shapeOfWarps[i];
  }
  // Make the axis 0 to fill the rest lanes and warps.
  shapeOfLanes[0] = restLanes;
  shapeOfWarps[0] = restWarps;
  return RegistersLayoutAttr::get(context, shapeOfWarps, loopsPerWarp,
                                  shapeOfLanes, loopsPerLane);
}

RegistersLayoutAttr kapy::getRegistersLayout(MLIRContext *context,
                                             ArrayRef<int64_t> shape,
                                             int64_t numWarps) {
  SmallVector<int64_t, 4> loops(shape.size(), 1);
  return getRegistersLayout(context, loops, loops, shape, numWarps);
}

bool kapy::isNvidiaMmaToMmOperandShortcut(NvidiaMmaLayoutAttr nvmmaLayout,
                                          MmOperandLayoutAttr mmopdLayout) {
  if (nvmmaLayout != mmopdLayout.getParent())
    return false;
  if (nvmmaLayout.getRank() != 2 || mmopdLayout.getOperandIndex() != 0 ||
      mmopdLayout.getBitWidth() != 16)
    return false;
  if (nvmmaLayout.getShapeOfWarpsRef()[1] != 1)
    return false;
  return true;
}

NvidiaMmaLayoutAttr kapy::getNvidiaMmaLayout(MatmulOp matmulOp,
                                             int64_t numWarps) {
  auto rank = matmulOp.getType().getRank();
  auto *context = matmulOp.getContext();
  SmallVector<int64_t, 4> shapeOfWarps(rank, 1);
  SmallVector<int64_t, 4> loopsPerWarp(rank, 1);
  loopsPerWarp[rank - 2] = 2;

  // Early exit for batched matmul case.
  if (rank == 3) {
    // TODO: This may cause waste of warps, we should consider the tensor shape.
    shapeOfWarps[0] = numWarps;
    return NvidiaMmaLayoutAttr::get(context, shapeOfWarps, loopsPerWarp);
  }

  TransitiveFilter inSameRegion = [matmulOp](Operation *op) {
    return matmulOp->getParentRegion() == op->getParentRegion();
  };
  auto slice = multiRootGetSlice(matmulOp, inSameRegion, inSameRegion);

  bool hasChainedMatmulOps = false;
  for (auto *op : slice) {
    if (isa<MatmulOp>(op) && op != matmulOp.getOperation()) {
      auto type = cast<MatmulOp>(op).getType();
      if (type.getRank() != rank)
        continue;
      if (auto nvmmaLayout =
              dyn_cast_or_null<NvidiaMmaLayoutAttr>(type.getEncoding()))
        return nvmmaLayout;
      hasChainedMatmulOps = true;
    }
  }
  if (hasChainedMatmulOps) {
    // TODO: This may cause waste of warps, we should consider the tensor shape.
    shapeOfWarps[0] = numWarps;
    return NvidiaMmaLayoutAttr::get(context, shapeOfWarps, loopsPerWarp);
  }

  auto shape = matmulOp.getType().getShape();
  while (shapeOfWarps[0] * shapeOfWarps[1] < numWarps) {
    if (shape[0] / (shapeOfWarps[0] * 16) >=
        shape[1] / (shapeOfWarps[1] * 16)) {
      if (shapeOfWarps[0] * 16 < shape[0]) {
        shapeOfWarps[0] *= 2;
        continue;
      }
    }
    shapeOfWarps[1] *= 2;
  }
  return NvidiaMmaLayoutAttr::get(context, shapeOfWarps, loopsPerWarp);
}
