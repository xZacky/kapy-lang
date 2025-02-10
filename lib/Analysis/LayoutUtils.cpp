//===- LayoutUtils.cpp ------------------------------------------*- C++ -*-===//
//
// This file implements functions for layout optimization.
//
//===----------------------------------------------------------------------===//

#include "kapy/Analysis/LayoutUtils.h"
#include "kapy/Analysis/AnalysisUtils.h"
#include "kapy/Dialect/Kapy/IR/Kapy.h"
#include "kapy/Dialect/Kgpu/IR/Kgpu.h"

using namespace mlir;
using namespace mlir::kapy;

FragmentsLayoutAttr kapy::getFragmentsLayout(ArrayRef<int64_t> laneLoops,
                                             RankedTensorType tensorType,
                                             int64_t numWarps, bool rowMajor) {
  auto shape = tensorType.getShape();
  SmallVector<int64_t, 2> laneArray(2);
  SmallVector<int64_t, 2> warpArray(2);
  SmallVector<int64_t, 2> warpLoops{1, 1};
  unsigned i = rowMajor ? 0 : 1;
  unsigned j = rowMajor ? 1 : 0;
  auto maxLanesJ =
      std::clamp<int64_t>(numWarps * numLanes, 1, shape[j] / laneLoops[j]);
  auto numLanesJ = std::clamp<int64_t>(maxLanesJ, 1, numLanes);
  auto numWarpsJ = std::clamp<int64_t>(maxLanesJ / numLanesJ, 1, numWarps);
  laneArray[i] = numLanes / numLanesJ;
  laneArray[j] = numLanesJ;
  warpArray[i] = numWarps / numWarpsJ;
  warpArray[j] = numWarpsJ;
  return FragmentsLayoutAttr::get(tensorType.getContext(), warpArray, warpLoops,
                                  laneArray, laneLoops, i, j);
}

FragmentsLayoutAttr kapy::getFragmentsLayout(RankedTensorType tensorType,
                                             int64_t numWarps,
                                             bool isRowMajor) {
  return getFragmentsLayout({1, 1}, tensorType, numWarps, isRowMajor);
}

FragmentsLayoutAttr kapy::getFragmentsLayout(MatmulOp matmulOp,
                                             int64_t numWarps) {
  auto implWay = matmulOp.getMatmulImplWay();
  auto resultType = matmulOp.getType();
  auto shape = resultType.getShape();
  auto numElements = resultType.getNumElements();
  SmallVector<int64_t, 2> laneLoops{1, 1};
  if (implWay == MatmulImplWay::FMA) {
    if (shape[0] >= 32 && shape[1] >= 32 &&
        numElements / (numWarps * numLanes) >= 16) {
      laneLoops[0] = 4;
      laneLoops[1] = 4;
    } else {
      laneLoops[0] = 2;
      laneLoops[1] = 2;
    }
    return getFragmentsLayout(laneLoops, resultType, numWarps);
  }

  SmallVector<int64_t, 2> warpArray{1, 1};
  SmallVector<int64_t, 2> warpLoops{2, 1};
  SmallVector<int64_t, 2> laneArray{8, 4};
  laneLoops = {1, 2};
  while (warpArray[0] * warpArray[1] < numWarps) {
    if (shape[0] / (warpArray[0] * 16) >= shape[1] / (warpArray[1] * 16)) {
      if (warpArray[0] * 16 < shape[0]) {
        warpArray[0] *= 2;
        continue;
      }
    }
    warpArray[1] *= 2;
  }
  return FragmentsLayoutAttr::get(matmulOp.getContext(), warpArray, warpLoops,
                                  laneArray, laneLoops);
}

static SetVector<Attribute> getCandidateLayoutsImpl(RankedTensorType tensorType,
                                                    int64_t numWarps) {
  auto shape = tensorType.getShape();
  auto numElements = tensorType.getNumElements();
  auto layout = cast<FragmentsLayoutAttr>(tensorType.getEncoding());

  auto warpArray = layout.getWarpArray();
  auto warpLoops = layout.getWarpLoops();
  auto laneArray = layout.getLaneArray();
  auto laneLoops = layout.getLaneLoops();

  auto i = layout.getMinorAxis();
  auto j = layout.getMajorAxis();

  auto laneLoopJ = laneLoops[j];
  auto maxLanesJ = laneArray[j];
  auto maxWarpsJ = warpArray[j];

  auto restLoop =
      std::max<int64_t>(numElements / (numWarps * numLanes * laneLoopJ), 1);
  // Possible combinations of `(warpLoopI, warpLoopJ, laneLoopI)`.
  SmallVector<std::array<int64_t, 3>, 8> loopCombs;
  for (int64_t warpLoopI = 1; warpLoopI <= restLoop; warpLoopI *= 2) {
    for (int64_t warpLoopJ = 1; warpLoopJ <= restLoop; warpLoopJ *= 2) {
      if (warpLoopI * warpLoopJ > restLoop)
        break;
      for (int64_t laneLoopI = 1; laneLoopI <= restLoop; laneLoopI *= 2) {
        if (warpLoopI * warpLoopJ * laneLoopI > restLoop)
          break;
        loopCombs.push_back({warpLoopI, warpLoopJ, laneLoopI});
      }
    }
  }

  // We try to make each warp access global memory that is at least 128 bytes
  // contiguous if possible.
  // TODO: This may not be a necessary condition.
  auto minLanesJ = 1024 / (laneLoopJ * getIntOrFloatBitWidth(tensorType));

  SetVector<Attribute> layouts;
  for (int64_t numLanesJ = minLanesJ; numLanesJ <= maxLanesJ; numLanesJ *= 2) {
    auto numLanesI = numLanes / numLanesJ;
    laneArray[i] = numLanesI;
    laneArray[j] = numLanesJ;
    for (int64_t numWarpsJ = 1; numWarpsJ <= maxWarpsJ; numWarpsJ *= 2) {
      if (numWarpsJ * numLanesJ * laneLoopJ > shape[j])
        break;
      auto numWarpsI = numWarps / numWarpsJ;
      warpArray[i] = numWarpsI;
      warpArray[j] = numWarpsJ;
      for (auto [warpLoopI, warpLoopJ, laneLoopI] : loopCombs) {
        if (numWarpsI * warpLoopI * numLanesI * laneLoopI > shape[i])
          continue;
        if (numWarpsJ * warpLoopJ * numLanesJ * laneLoopJ > shape[j])
          continue;
        warpLoops[i] = warpLoopI;
        warpLoops[j] = warpLoopJ;
        laneLoops[i] = laneLoopI;
        layouts.insert(FragmentsLayoutAttr::get(tensorType.getContext(),
                                                warpArray, warpLoops, laneArray,
                                                laneLoops, i, j));
      }
    }
  }
  return layouts;
}

static SetVector<Attribute> getCandidateLayoutsImpl(MatmulOp matmulOp,
                                                    int64_t numWarps) {
  auto implWay = matmulOp.getMatmulImplWay();
  auto resultType = matmulOp.getType();

  if (implWay == MatmulImplWay::FMA) {
    auto layouts = getCandidateLayoutsImpl(resultType, numWarps);
    for (auto layout : layouts) {
      auto laneLoops = cast<FragmentsLayoutAttr>(layout).getLaneLoops();
      if (laneLoops[0] != laneLoops[1])
        layouts.remove(layout);
    }
    return layouts;
  }

  auto shape = resultType.getShape();
  auto numElements = resultType.getNumElements();
  auto layout = cast<FragmentsLayoutAttr>(resultType.getEncoding());

  auto warpArray = layout.getWarpArray();
  auto warpLoops = layout.getWarpLoops();
  auto laneArray = layout.getLaneArray();
  auto laneLoops = layout.getLaneLoops();

  auto restLoop = std::max<int64_t>(numElements / (numWarps * 64), 1);

  SetVector<Attribute> layouts;
  for (int64_t numWarps0 = 1; numWarps0 <= numWarps; numWarps0 *= 2) {
    auto numWarps1 = numWarps / numWarps0;
    warpArray[0] = numWarps0;
    warpArray[1] = numWarps1;
    // Warp loop on axis 0 is at least 2.
    for (int64_t warpLoop0 = 2; warpLoop0 <= restLoop; warpLoop0 *= 2) {
      for (int64_t warpLoop1 = 1; warpLoop1 <= restLoop; warpLoop1 *= 2) {
        if (warpLoop0 * warpLoop1 > restLoop)
          break;
        if (numWarps0 * warpLoop0 * 8 > shape[0])
          break;
        if (numWarps1 * warpLoop1 * 8 > shape[1])
          break;
        warpLoops[0] = warpLoop0;
        warpLoops[1] = warpLoop1;
        layouts.insert(FragmentsLayoutAttr::get(
            matmulOp.getContext(), warpArray, warpLoops, laneArray, laneLoops));
      }
    }
  }
  return layouts;
}

SetVector<Attribute> kapy::getCandidateLayouts(Operation *op,
                                               int64_t numWarps) {
  if (isGlobalMemoryRead(op))
    for (auto result : op->getResults())
      if (auto tensorType = dyn_cast<RankedTensorType>(result.getType()))
        return getCandidateLayoutsImpl(tensorType, numWarps);

  if (isGlobalMemoryWrite(op))
    for (auto operand : op->getOperands())
      if (auto tensorType = dyn_cast<RankedTensorType>(operand.getType()))
        return getCandidateLayoutsImpl(tensorType, numWarps);

  if (auto matmulOp = dyn_cast<MatmulOp>(op))
    return getCandidateLayoutsImpl(matmulOp, numWarps);

  llvm_unreachable("unsupported operation");
}

/*
SharedMemLayoutAttr kapy::getSharedMemLayout(RankedTensorType sourceType,
                                             RankedTensorType resultType) {
  auto tensorShape = sourceType.getShape();

  auto sourceLayout = getLayoutAsFragmentsLayout(sourceType);
  auto i = sourceLayout.getMinorAxis();
  auto j = sourceLayout.getMajorAxis();

  if (auto mmopdLayout =
          dyn_cast<MmOperandLayoutAttr>(resultType.getEncoding())) {
    auto *context = resultType.getContext();
    SmallVector<int64_t, 2> strides;
    if (i == 0)
      strides = computeStrides(tensorShape);
    else
      strides = transpose(computeStrides(transpose(tensorShape)));

    /// Do not swizzle if parent is not nvidia mma layout.
    if (!isa<NvidiaMmaLayoutAttr>(mmopdLayout.getParent()))
      return SharedMemLayoutAttr::get(context, strides, 1, 1);

    // We use "ldmatrix...x4" instructions for tensor core operands.
    auto bitWidth = mmopdLayout.getBitWidth();
    auto unitSize = 128 / bitWidth;
    // In each phase, 8 lanes access 32 banks and each bank provides 4 bytes.
    // Compute how many row/columns per phase.
    auto perPhase = std::max<unsigned>(1024 / (tensorShape[j] * bitWidth), 1);
    // In each phase, 8 lanes hold `[8, unitSize]` elements, so we compute the
    // maximum phase by this to avoid bank conflicts.
    if (mmopdLayout.getOperandIndex() == 0) {
      auto maxPhase = (i == 0 ? 8 : unitSize) / perPhase;
      return SharedMemLayoutAttr::get(context, strides, unitSize, maxPhase);
    } else {
      auto maxPhase = (i == 0 ? unitSize : 8) / perPhase;
      return SharedMemLayoutAttr::get(context, strides, unitSize, maxPhase);
    }
  }
}
*/
