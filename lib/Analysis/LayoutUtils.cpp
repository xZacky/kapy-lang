//===- LayoutUtils.cpp ------------------------------------------*- C++ -*-===//
//
// This file implements functions for layout optimization.
//
//===----------------------------------------------------------------------===//

#include "kapy/Analysis/LayoutUtils.h"
#include "kapy/Analysis/AnalysisUtils.h"
#include "kapy/Dialect/Kapy/IR/Kapy.h"
#include "kapy/Dialect/Kgpu/IR/Kgpu.h"
#include "llvm/ADT/MapVector.h"

using namespace mlir;
using namespace mlir::kapy;

FragmentsLayoutAttr kapy::getFragmentsLayout(ArrayRef<int64_t> laneLoops,
                                             RankedTensorType tensorType,
                                             bool rowMajor) {
  unsigned i = rowMajor ? 0 : 1;
  unsigned j = rowMajor ? 1 : 0;
  auto shape = tensorType.getShape();
  SmallVector<int64_t, 2> laneArray(2);
  laneArray[j] = std::clamp<int64_t>(warpSize, 1, shape[j] / laneLoops[j]);
  laneArray[i] = warpSize / laneArray[j];
  auto *context = tensorType.getContext();
  return FragmentsLayoutAttr::get(context, laneArray, laneLoops, i, j);
}

FragmentsLayoutAttr kapy::getFragmentsLayout(RankedTensorType tensorType,
                                             bool rowMajor) {
  return getFragmentsLayout({1, 1}, tensorType, rowMajor);
}

std::array<FragmentsLayoutAttr, 3> kapy::getDefaultLayouts(MatmulOp op) {
  auto *context = op.getContext();
  switch (op.getMatmulImplWay()) {
  case (MatmulImplWay::MMA_M16N8K8_F16):
  case (MatmulImplWay::MMA_M16N8K16_F16): {
    auto lhsLayout = FragmentsLayoutAttr::get(context, {8, 4}, {1, 2}, 0, 1);
    auto rhsLayout = FragmentsLayoutAttr::get(context, {4, 8}, {2, 1}, 1, 0);
    auto accLayout = FragmentsLayoutAttr::get(context, {8, 4}, {1, 2}, 0, 1);
    return {lhsLayout, rhsLayout, accLayout};
  }

  case (MatmulImplWay::MMA_M16N8K8_TF32): {
    auto lhsLayout = FragmentsLayoutAttr::get(context, {8, 4}, {1, 1}, 0, 1);
    auto rhsLayout = FragmentsLayoutAttr::get(context, {4, 8}, {1, 1}, 1, 0);
    auto accLayout = FragmentsLayoutAttr::get(context, {8, 4}, {1, 2}, 0, 1);
    return {lhsLayout, rhsLayout, accLayout};
  }

  case (MatmulImplWay::MMA_M16N8K16_F8): {
    auto lhsLayout = FragmentsLayoutAttr::get(context, {8, 4}, {1, 4}, 0, 1);
    auto rhsLayout = FragmentsLayoutAttr::get(context, {4, 8}, {4, 1}, 1, 0);
    auto accLayout = FragmentsLayoutAttr::get(context, {8, 4}, {1, 2}, 0, 1);
    return {lhsLayout, rhsLayout, accLayout};
  }
  }
  llvm_unreachable("unsupported matmul implement way");
}

std::array<FragmentsLayoutAttr, 2> kapy::getDefaultLayouts(LdMatrixOp op) {
  auto sourceType = op.getSource().getType();
  auto bitWidth = getIntOrFloatBitWidth(sourceType);
  auto vecWidth = 128 / bitWidth;
  auto pacWidth = 32 / bitWidth;
  auto *context = op.getContext();
  auto loaderLayout =
      FragmentsLayoutAttr::get(context, {16, 2}, {1, vecWidth}, 1, 0);
  auto resultLayout =
      FragmentsLayoutAttr::get(context, {8, 4}, {1, pacWidth}, 0, 1);
  return {loaderLayout, resultLayout};
}

SetVector<FragmentsLayoutAttr> kapy::getCandidateLayouts(Operation *op) {
  if (isGlobalRead(op) || isGlobalWrite(op)) {
    RankedTensorType tensorType;
    if (isGlobalRead(op))
      tensorType = cast<RankedTensorType>(op->getResult(0).getType());
    else
      tensorType = cast<RankedTensorType>(op->getOperand(0).getType());
    auto *context = tensorType.getContext();
    auto numElems = tensorType.getNumElements();
    auto shape = tensorType.getShape();
    auto layout = getLayout<FragmentsLayoutAttr>(tensorType);

    auto laneArray = layout.getLaneArray();
    auto laneLoops = layout.getLaneLoops();

    unsigned i = layout.getMinorAxis();
    unsigned j = layout.getMajorAxis();

    auto vecWidth = laneLoops[j];
    auto restLoop = std::max<int64_t>(numElems / (warpSize * vecWidth), 1);

    SetVector<FragmentsLayoutAttr> layouts;
    layouts.insert(layout);

    for (int64_t laneLoopI = 1; laneLoopI <= restLoop; laneLoopI *= 2) {
      auto laneLoopJ = vecWidth;
      if (laneArray[i] * laneLoopI > shape[i])
        break;
      if (laneArray[j] * laneLoopJ > shape[j])
        break;
      laneLoops[i] = laneLoopI;
      laneLoops[j] = laneLoopJ;
      layouts.insert(
          FragmentsLayoutAttr::get(context, laneArray, laneLoops, i, j));
    }

    i = layout.getMajorAxis();
    j = layout.getMinorAxis();

    for (int64_t laneLoopJ = 1; laneLoopJ <= restLoop; laneLoopJ *= 2) {
      auto laneLoopI = vecWidth;
      if (laneArray[i] * laneLoopI > shape[i])
        break;
      if (laneArray[j] * laneLoopJ > shape[j])
        break;
      laneLoops[i] = laneLoopI;
      laneLoops[j] = laneLoopJ;
      layouts.insert(
          FragmentsLayoutAttr::get(context, laneArray, laneLoops, i, j));
    }

    return layouts;
  }
  llvm_unreachable("unsupported operation");
}

static bool isNoBankConflict(SwizzlingLayoutAttr sharedLayout,
                             RankedTensorType tensorType) {
  auto bankParam = sharedLayout.getBankParam();
  auto lineParam = sharedLayout.getLineParam();

  auto shape = tensorType.getShape();
  auto tensorLayout = getLayout<FragmentsLayoutAttr>(tensorType);
  auto option = FragmentsLayoutAttr::MapOption::TO_TENSOR;
  auto map = tensorLayout.getAffineMap(shape, option);

  auto bitWidth = getIntOrFloatBitWidth(tensorType);
  auto vecWidth = bankParam * 32 / bitWidth;

  int64_t ivStep = 1;
  if (sharedLayout.isRowMajor() && tensorLayout.isColMajor())
    ivStep = tensorLayout.getLoopSpace(shape)[0];
  if (sharedLayout.isColMajor() && tensorLayout.isRowMajor())
    ivStep = tensorLayout.getLoopSpace(shape)[1];

  llvm::MapVector<int64_t, DenseSet<int64_t>> bankIdToLineIds;
  for (int64_t laneId = 0; laneId < warpSize / bankParam; ++laneId) {
    for (int64_t loopIv = 0; loopIv < vecWidth * ivStep; loopIv += ivStep) {
      auto indices = map.compose({laneId, loopIv});
      int64_t bitOffset = bitWidth;
      if (sharedLayout.isRowMajor())
        bitOffset *= indices[0] * shape[1] + indices[1];
      else
        bitOffset *= indices[0] + indices[1] * shape[0];
      auto bankId = bitOffset % 1024 / 32;
      auto lineId = bitOffset / 1024;
      bankId = (bankId / bankParam ^ lineId % lineParam) * bankParam +
               (bankId % bankParam);
      bankIdToLineIds[bankId].insert(lineId);
    }
  }
  for (const auto &it : bankIdToLineIds)
    if (it.second.size() > 1)
      return false;
  return true;
}

SetVector<SwizzlingLayoutAttr>
kapy::getSwizzlingLayouts(RankedTensorType tensorType) {
  auto *context = tensorType.getContext();
  SetVector<SwizzlingLayoutAttr> layouts;
  for (int64_t bankParam : {4, 2, 1}) {
    for (int64_t lineParam = 1; lineParam <= 32 / bankParam; lineParam *= 2) {
      auto layout = SwizzlingLayoutAttr::get(context, bankParam, lineParam);
      if (isNoBankConflict(layout, tensorType))
        layouts.insert(layout);
      layout = SwizzlingLayoutAttr::get(context, bankParam, lineParam, false);
      if (isNoBankConflict(layout, tensorType))
        layouts.insert(layout);
    }
  }
  return layouts;
}
