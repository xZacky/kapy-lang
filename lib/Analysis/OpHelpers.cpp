//===- OpHelpers.h ----------------------------------------------*- C++ -*-===//
//
// This file implements layout analysis for operations using shuffle.
//
//===----------------------------------------------------------------------===//

#include "kapy/Analysis/OpHelpers.h"
#include "kapy/Dialect/Kapy/IR/Kapy.h"
#include "kapy/Dialect/Kgpu/IR/Kgpu.h"
#include "kapy/Support/CommonUtils.h"
#include "llvm/ADT/MapVector.h"

using namespace mlir;
using namespace mlir::kapy;

ReduceOpHelper::ReduceOpHelper(ReduceOp reduceOp) {
  this->sourceType = reduceOp.getSource().getType();
  this->axis = reduceOp.getAxis();
}

ReduceOpHelper::ReduceOpHelper(RankedTensorType sourceType, unsigned axis) {
  this->sourceType = sourceType;
  this->axis = axis;
}

int64_t ReduceOpHelper::getNumShfls() const {
  auto shape = sourceType.getShape();
  auto layout = getLayout<FragmentsLayoutAttr>(sourceType);
  auto option = FragmentsLayoutAttr::MapOption::FROM_VALUES;
  auto map = layout.getAffineMap(shape, option);
  DenseSet<int64_t> laneIdSet;
  for (int64_t index = 0; index < shape[axis]; ++index) {
    int64_t laneId = 0;
    if (axis == 0)
      laneId = map.compose({index, 0})[0];
    else
      laneId = map.compose({0, index})[0];
    laneIdSet.insert(laneId);
  }
  return log2(laneIdSet.size());
}

int64_t ReduceOpHelper::getLaneOffset() const {
  auto layout = getLayout<FragmentsLayoutAttr>(sourceType);
  if (layout.getMajorAxis() == axis)
    return 1;
  else
    return layout.getLaneArray()[axis == 1 ? 0 : 1];
}

ChangeOpHelper::ChangeOpHelper(ChangeOp changeOp) {
  this->sourceType = changeOp.getSource().getType();
  this->resultType = changeOp.getType();
}

ChangeOpHelper::ChangeOpHelper(RankedTensorType sourceType,
                               RankedTensorType resultType) {
  this->sourceType = sourceType;
  this->resultType = resultType;
}

int64_t ChangeOpHelper::getNumShfls() const {
  auto layout = getLayout<FragmentsLayoutAttr>(resultType);
  auto loopSize = product(layout.getLoopSpace(resultType.getShape()));
  auto bitWidth = getIntOrFloatBitWidth(resultType);
  auto map = getShflIdxMap();
  llvm::MapVector<int64_t, int64_t> laneIdToNumValues;
  for (int64_t loopIv = 0; loopIv < loopSize; ++loopIv) {
    auto laneId = map.compose({0, loopIv})[0];
    if (laneIdToNumValues.contains(laneId))
      laneIdToNumValues[laneId] += 1;
    else
      laneIdToNumValues[laneId] = 1;
  }
  int64_t numShfls = 0;
  for (auto [laneId, numValues] : laneIdToNumValues) {
    if (laneId == 0)
      continue;
    // We can exchange 32 bits in each shuffle.
    numShfls += ceilDiv<int64_t>(numValues * bitWidth, 32);
  }
  return numShfls;
}

AffineMap ChangeOpHelper::getShflIdxMap() const {
  auto shape = resultType.getShape();
  auto sourceLayout = getLayout<FragmentsLayoutAttr>(sourceType);
  auto resultLayout = getLayout<FragmentsLayoutAttr>(resultType);
  auto sourceOption = FragmentsLayoutAttr::MapOption::FROM_VALUES;
  auto resultOption = FragmentsLayoutAttr::MapOption::TO_VALUES;
  auto sourceMap = sourceLayout.getAffineMap(shape, sourceOption);
  auto resultMap = resultLayout.getAffineMap(shape, resultOption);
  return sourceMap.compose(resultMap);
}
