//===- Layout.cpp -----------------------------------------------*- C++ -*-===//
//
// This file implements functions for optimizations of layouts.
//
//===----------------------------------------------------------------------===//

#include "kapy/Analysis/Layout.h"
#include "kapy/Analysis/Utils.h"
#include "kapy/Dialect/Kapy/IR/Kapy.h"
#include "kapy/Dialect/Kapy/IR/Utils.h"
#include "kapy/Dialect/Kgpu/IR/Kgpu.h"
#include "kapy/Dialect/Kgpu/IR/Utils.h"

using namespace mlir;
using namespace kapy;

FragmentsLayoutAttr kapy::getFragmentsLayout(MLIRContext *context,
                                             ArrayRef<int64_t> laneLoops,
                                             ArrayRef<int64_t> shape,
                                             int64_t numWarps,
                                             bool needTranspose) {
  auto rank = shape.size();
  assert(rank == 1 || rank == 2);
  SmallVector<int64_t, 2> shapeOfLanes(rank);
  SmallVector<int64_t, 2> shapeOfWarps(rank);
  SmallVector<int64_t, 2> warpLoops(rank, 1);
  if (rank == 1) {
    shapeOfLanes[0] = numLanes;
    shapeOfWarps[0] = numWarps;
    return FragmentsLayoutAttr::get(context, shapeOfWarps, warpLoops,
                                    shapeOfLanes, laneLoops);
  }
  auto numThreads = numLanes * numWarps;
  auto restLanes = numLanes;
  auto restWarps = numWarps;
  unsigned majorAxis = needTranspose ? 0 : 1;
  unsigned minorAxis = needTranspose ? 1 : 0;
  auto numLoopsMajor = warpLoops[majorAxis] * laneLoops[majorAxis];
  auto numThreadsMajor =
      std::clamp<int64_t>(numThreads, 1, shape[majorAxis] / numLoopsMajor);
  auto numLanesMajor = std::clamp<int64_t>(numThreadsMajor, 1, numLanes);
  auto numWarpsMajor =
      std::clamp<int64_t>(numThreadsMajor / numLanesMajor, 1, numWarps);
  shapeOfLanes[majorAxis] = numLanesMajor;
  shapeOfLanes[minorAxis] = numLanes / numLanesMajor;
  shapeOfWarps[majorAxis] = numWarpsMajor;
  shapeOfWarps[minorAxis] = numWarps / numWarpsMajor;
  return FragmentsLayoutAttr::get(context, shapeOfWarps, warpLoops,
                                  shapeOfLanes, laneLoops, majorAxis);
}

FragmentsLayoutAttr kapy::getFragmentsLayout(MLIRContext *context,
                                             ArrayRef<int64_t> shape,
                                             int64_t numWarps) {
  SmallVector<int64_t, 2> laneLoops(shape.size(), 1);
  return getFragmentsLayout(context, laneLoops, shape, numWarps);
}

NvidiaMmaLayoutAttr kapy::getNvidiaMmaLayout(MatmulOp matmulOp,
                                             int64_t numWarps) {
  TransitiveFilter inSameRegion = [matmulOp](Operation *op) {
    return matmulOp->getParentRegion() == op->getParentRegion();
  };
  auto slice = multiRootGetSlice(matmulOp, inSameRegion, inSameRegion);

  bool hasChainedMatmulOps = false;
  for (auto *op : slice) {
    if (isa<MatmulOp>(op) && op != matmulOp.getOperation()) {
      if (auto nvmmaLayout = dyn_cast_or_null<NvidiaMmaLayoutAttr>(
              cast<MatmulOp>(op).getType().getEncoding()))
        return nvmmaLayout;
      hasChainedMatmulOps = true;
    }
  }

  auto *context = matmulOp.getContext();
  SmallVector<int64_t, 2> shapeOfWarps{1, 1};
  SmallVector<int64_t, 2> warpLoops{2, 1};
  if (hasChainedMatmulOps) {
    // TODO: This may cause waste of warps, we should consider the tensor shape.
    shapeOfWarps[0] = numWarps;
    return NvidiaMmaLayoutAttr::get(context, shapeOfWarps, warpLoops);
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
  return NvidiaMmaLayoutAttr::get(context, shapeOfWarps, warpLoops);
}

SharedMemLayoutAttr kapy::getSharedMemLayout(MLIRContext *context,
                                             MmOperandLayoutAttr mmopdLayout,
                                             ArrayRef<int64_t> shape,
                                             bool needTranspose) {
  auto shmemShape = llvm::to_vector<2>(shape);
  if (needTranspose)
    shmemShape = transpose(shape);
  auto strides = computeStrides(shmemShape);
  if (needTranspose)
    strides = transpose(strides);
  auto bitWidth = mmopdLayout.getBitWidth();
  auto nvmmaLayout = dyn_cast<NvidiaMmaLayoutAttr>(mmopdLayout.getParent());
  if (!nvmmaLayout)
    return SharedMemLayoutAttr::get(context, strides, bitWidth, 1);
  auto perPhase = std::max<unsigned>(1024 / (shmemShape[1] * bitWidth), 1);
  if (mmopdLayout.getOperandIndex() == 0) {
    auto maxPhase = (needTranspose ? 128 / bitWidth : 8) / perPhase;
    return SharedMemLayoutAttr::get(context, strides, bitWidth, maxPhase);
  } else {
    auto maxPhase = (needTranspose ? 8 : 128 / bitWidth) / perPhase;
    return SharedMemLayoutAttr::get(context, strides, bitWidth, maxPhase);
  }
}

static SetVector<Attribute> getCandidateLayouts1d(MLIRContext *context,
                                                  RankedTensorType type,
                                                  int64_t numWarps) {
  auto shape = type.getShape();
  auto fragsLayout = cast<FragmentsLayoutAttr>(type.getEncoding());
  assert(fragsLayout.getRank() == 1);

  auto numElems = product(shape);
  auto shapeOfWarps = fragsLayout.getShapeOfWarpsRef();
  auto warpLoops = fragsLayout.getWarpLoops();
  auto shapeOfLanes = fragsLayout.getShapeOfLanesRef();
  auto laneLoops = fragsLayout.getLaneLoopsRef();

  auto vecWidth = laneLoops[0];
  auto restLoop =
      std::max<int64_t>(numElems / (numWarps * numLanes * vecWidth), 1);

  SetVector<Attribute> layouts;
  for (int64_t warpLoop = 1; warpLoop <= restLoop; warpLoop *= 2) {
    warpLoops[0] = warpLoop;
    layouts.insert(FragmentsLayoutAttr::get(context, shapeOfWarps, warpLoops,
                                            shapeOfLanes, laneLoops));
  }
  return layouts;
}

static SetVector<Attribute> getCandidateLayouts2d(MLIRContext *context,
                                                  RankedTensorType type,
                                                  int64_t numWarps) {
  auto shape = type.getShape();
  auto fragsLayout = cast<FragmentsLayoutAttr>(type.getEncoding());
  assert(fragsLayout.getRank() == 2);

  auto numElems = product(shape);
  auto shapeOfWarps = fragsLayout.getShapeOfWarps();
  auto warpLoops = fragsLayout.getWarpLoops();
  auto shapeOfLanes = fragsLayout.getShapeOfLanes();
  auto laneLoops = fragsLayout.getLaneLoops();

  unsigned majorAxis = 1;
  int64_t vecWidth = 1;
  int64_t maxLanesMajor = numLanes;
  int64_t maxThreadsMajor = numLanes * numWarps;
  for (unsigned i = 0; i < 2; ++i) {
    if (laneLoops[i] > 1) {
      majorAxis = i;
      vecWidth = laneLoops[i];
      maxLanesMajor = shapeOfLanes[i];
      maxThreadsMajor = shapeOfWarps[i] * shapeOfLanes[i];
      break;
    }
  }
  assert(vecWidth > 1);

  auto restLoops =
      std::max<int64_t>(numElems / (numWarps * numLanes * vecWidth), 1);
  // Possibile combinations of `(warpLoopMinor, warpLoopMajor, laneLoopMinor)`.
  SmallVector<std::array<int64_t, 3>> loopsCombs;
  for (int64_t warpLoopMinor = 1; warpLoopMinor <= restLoops;
       warpLoopMinor *= 2) {
    for (int64_t warpLoopMajor = 1; warpLoopMajor <= restLoops;
         warpLoopMajor *= 2) {
      if (warpLoopMinor * warpLoopMajor > restLoops)
        break;
      for (int64_t laneLoopMinor = 1; laneLoopMinor <= restLoops;
           laneLoopMinor *= 2) {
        if (warpLoopMinor * warpLoopMajor * laneLoopMinor > restLoops)
          break;
        loopsCombs.push_back({warpLoopMinor, warpLoopMajor, laneLoopMinor});
      }
    }
  }

  unsigned minorAxis = majorAxis == 1 ? 0 : 1;
  auto bitWidth = getIntOrFloatBitWidth(type);
  auto minLanesMajor = 1024 / (vecWidth * bitWidth);

  SetVector<Attribute> layouts;
  for (int64_t numLanesMajor = minLanesMajor; numLanesMajor <= maxLanesMajor;
       numLanesMajor *= 2) {
    auto numLanesMinor = numLanes / numLanesMajor;
    shapeOfLanes[majorAxis] = numLanesMajor;
    shapeOfLanes[minorAxis] = numLanesMinor;
    for (int64_t numWarpsMajor = 1;
         numWarpsMajor * numLanesMajor <= maxThreadsMajor; numWarpsMajor *= 2) {
      if (numWarpsMajor * numLanesMajor * vecWidth > shape[majorAxis])
        break;
      auto numWarpsMinor = numWarps / numWarpsMajor;
      shapeOfWarps[majorAxis] = numWarpsMajor;
      shapeOfWarps[minorAxis] = numWarpsMinor;
      for (auto [warpLoopMinor, warpLoopMajor, laneLoopMinor] : loopsCombs) {
        if (numWarpsMajor * warpLoopMajor * numLanesMajor * vecWidth >
            shape[majorAxis])
          continue;
        if (numWarpsMinor * warpLoopMinor * numLanesMinor * laneLoopMinor >
            shape[minorAxis])
          continue;
        warpLoops[minorAxis] = warpLoopMinor;
        warpLoops[majorAxis] = warpLoopMajor;
        laneLoops[minorAxis] = laneLoopMinor;
        layouts.insert(FragmentsLayoutAttr::get(context, shapeOfWarps,
                                                warpLoops, shapeOfLanes,
                                                laneLoops, majorAxis));
      }
    }
  }
  return layouts;
}

static SetVector<Attribute> getCandidateLayouts(LoadOp loadOp,
                                                int64_t numWarps) {
  auto resultType = cast<RankedTensorType>(loadOp.getType());
  if (resultType.getRank() == 1)
    return getCandidateLayouts1d(loadOp.getContext(), resultType, numWarps);
  if (resultType.getRank() == 2)
    return getCandidateLayouts2d(loadOp.getContext(), resultType, numWarps);
  llvm_unreachable("invalid tensor rank");
}

static SetVector<Attribute> getCandidateLayouts(StoreOp storeOp,
                                                int64_t numWarps) {
  auto valueType = cast<RankedTensorType>(storeOp.getValue().getType());
  if (valueType.getRank() == 1)
    return getCandidateLayouts1d(storeOp.getContext(), valueType, numWarps);
  if (valueType.getRank() == 2)
    return getCandidateLayouts2d(storeOp.getContext(), valueType, numWarps);
  llvm_unreachable("invalid tensor rank");
}

static SetVector<Attribute> getCandidateLayouts(AtomicRMWOp rmwOp,
                                                int64_t numWarps) {
  auto valueType = cast<RankedTensorType>(rmwOp.getValue().getType());
  if (valueType.getRank() == 1)
    return getCandidateLayouts1d(rmwOp.getContext(), valueType, numWarps);
  if (valueType.getRank() == 2)
    return getCandidateLayouts2d(rmwOp.getContext(), valueType, numWarps);
  llvm_unreachable("invalid tensor rank");
}

static SetVector<Attribute> getCandidateLayouts(AtomicCASOp casOp,
                                                int64_t numWarps) {
  auto valueType = cast<RankedTensorType>(casOp.getValue().getType());
  if (valueType.getRank() == 1)
    return getCandidateLayouts1d(casOp.getContext(), valueType, numWarps);
  if (valueType.getRank() == 2)
    return getCandidateLayouts2d(casOp.getContext(), valueType, numWarps);
  llvm_unreachable("invalid tensor rank");
}

static SetVector<Attribute> getCandidateLayouts(MatmulOp matmulOp,
                                                int64_t numWarps) {
  auto resultType = matmulOp.getType();
  if (isa<FragmentsLayoutAttr>(resultType.getEncoding())) {
    auto layouts =
        getCandidateLayouts2d(matmulOp.getContext(), resultType, numWarps);
    for (auto layout : layouts) {
      auto laneLoops = cast<FragmentsLayoutAttr>(layout).getLaneLoopsRef();
      if (laneLoops[0] != laneLoops[1])
        layouts.remove(layout);
    }
    return layouts;
  }
  auto shape = resultType.getShape();
  auto numElems = product(shape);
  auto nvmmaLayout = cast<NvidiaMmaLayoutAttr>(resultType.getEncoding());
  auto shapeOfWarps = nvmmaLayout.getShapeOfWarps();
  auto warpLoops = nvmmaLayout.getWarpLoops();
  auto restLoop = std::max<int64_t>(numElems / (numWarps * 64), 1);
  SetVector<Attribute> layouts;
  for (int64_t numWarpsMinor = 1; numWarpsMinor <= numWarps;
       numWarpsMinor *= 2) {
    auto numWarpsMajor = numWarps / numWarpsMinor;
    shapeOfWarps[0] = numWarpsMinor;
    shapeOfWarps[1] = numWarpsMajor;
    // Warp loop minor is at least 2.
    for (int64_t warpLoopMinor = 2; warpLoopMinor <= restLoop;
         warpLoopMinor *= 2) {
      for (int64_t warpLoopMajor = 1; warpLoopMajor <= restLoop;
           warpLoopMajor *= 2) {
        if (warpLoopMinor * warpLoopMajor > restLoop)
          break;
        if (numWarpsMinor * warpLoopMinor * 8 > shape[0])
          break;
        if (numWarpsMajor * warpLoopMajor * 8 > shape[1])
          break;
        warpLoops[0] = warpLoopMinor;
        warpLoops[1] = warpLoopMajor;
        layouts.insert(NvidiaMmaLayoutAttr::get(matmulOp.getContext(),
                                                shapeOfWarps, warpLoops));
      }
    }
  }
  return layouts;
}

SetVector<Attribute> kapy::getCandidateLayouts(Operation *op,
                                               int64_t numWarps) {
  if (auto loadOp = dyn_cast<LoadOp>(op))
    return ::getCandidateLayouts(loadOp, numWarps);
  if (auto storeOp = dyn_cast<StoreOp>(op))
    return ::getCandidateLayouts(storeOp, numWarps);
  if (auto rmwOp = dyn_cast<AtomicRMWOp>(op))
    return ::getCandidateLayouts(rmwOp, numWarps);
  if (auto casOp = dyn_cast<AtomicCASOp>(op))
    return ::getCandidateLayouts(casOp, numWarps);
  if (auto matmulOp = dyn_cast<MatmulOp>(op))
    return ::getCandidateLayouts(matmulOp, numWarps);
  llvm_unreachable("unsupported operation");
}
