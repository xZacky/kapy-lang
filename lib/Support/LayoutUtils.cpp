//===- LayoutUtils.cpp ------------------------------------------*- C++ -*-===//
//
// This file implements functions about layout.
//
//===----------------------------------------------------------------------===//

#include "kapy/Support/LayoutUtils.h"
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
  auto map = layout.getAffineMap(shape, 3);
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
  auto map = getShflIdxMap();
  int64_t numShfls = 0;
  for (int64_t loopIv = 0; loopIv < loopSize; ++loopIv) {
    auto laneId = map.compose({0, loopIv})[0];
    if (laneId != 0)
      numShfls += 1;
  }
  return numShfls;
}

AffineMap ChangeOpHelper::getShflIdxMap() const {
  auto shape = resultType.getShape();
  auto sourceLayout = getLayout<FragmentsLayoutAttr>(sourceType);
  auto resultLayout = getLayout<FragmentsLayoutAttr>(resultType);
  auto oldMap = sourceLayout.getAffineMap(shape, 3);
  auto newMap = resultLayout.getAffineMap(shape, 2);
  return oldMap.compose(newMap);
}

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

std::array<FragmentsLayoutAttr, 2> kapy::getDefaultLayouts(LdMatrixOp op) {
  auto *context = op.getContext();
  auto bitWidth = getIntOrFloatBitWidth(op.getSource().getType());
  auto simdSize = 128 / bitWidth;
  auto packSize = 32 / bitWidth;
  auto loaderLayout =
      FragmentsLayoutAttr::get(context, {16, 2}, {1, simdSize}, 1, 0);
  auto resultLayout =
      FragmentsLayoutAttr::get(context, {8, 4}, {1, packSize}, 0, 1);
  return {loaderLayout, resultLayout};
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

SetVector<FragmentsLayoutAttr> kapy::getCandidateLayouts(Operation *op) {
  if (isGlobalMemoryRead(op) || isGlobalMemoryWrite(op)) {
    RankedTensorType tensorType;
    if (isGlobalMemoryRead(op))
      tensorType = cast<RankedTensorType>(op->getResult(0).getType());
    else
      tensorType = cast<RankedTensorType>(op->getOperand(0).getType());
    auto layout = getLayout<FragmentsLayoutAttr>(tensorType);
    SetVector<FragmentsLayoutAttr> layouts;
    layouts.insert(layout);
    layouts.insert(layout.exchangeAxes());
    return layouts;
  }
  llvm_unreachable("unsupported operation");
}

static bool isNoBankConflict(RankedTensorType sharedType,
                             RankedTensorType tensorType, int64_t alignment) {
  auto sharedLayout = getLayout<SwizzlingLayoutAttr>(sharedType);
  auto stride0 = sharedLayout.getStride0();
  auto stride1 = sharedLayout.getStride1();
  auto bankParam = sharedLayout.getBankParam();
  auto lineParam = sharedLayout.getLineParam();

  auto shape = tensorType.getShape();
  auto tensorLayout = getLayout<FragmentsLayoutAttr>(tensorType);
  auto map = tensorLayout.getAffineMap(shape, 1);

  auto bitWidth = getIntOrFloatBitWidth(tensorType);
  auto simdSize = tensorLayout.getLaneLoops()[stride0 == 1 ? 0 : 1];
  simdSize = std::min(simdSize, bankParam * 32 / bitWidth);

  int64_t ivStep = 1;
  if (sharedLayout.isRowMajor() && tensorLayout.isColMajor())
    ivStep = tensorLayout.getLoopSpace(shape)[0];
  if (sharedLayout.isColMajor() && tensorLayout.isRowMajor())
    ivStep = tensorLayout.getLoopSpace(shape)[1];

  llvm::MapVector<int64_t, DenseSet<int64_t>> bankIdToLineIds;
  for (int64_t laneId = 0; laneId < warpSize / bankParam; ++laneId) {
    for (int64_t loopIv = 0; loopIv < simdSize * ivStep; loopIv += ivStep) {
      auto bitOffset = alignment % 128 * 8;
      auto indices = map.compose({laneId, loopIv});
      if (sharedLayout.isRowMajor())
        bitOffset += (indices[0] * stride1 + indices[1]) * bitWidth;
      else
        bitOffset += (indices[0] + indices[1] * stride0) * bitWidth;
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
kapy::getSwizzlingLayouts(RankedTensorType sharedType,
                          RankedTensorType tensorType, int64_t alignment) {
  auto layout = getLayout<SwizzlingLayoutAttr>(sharedType);
  SetVector<SwizzlingLayoutAttr> layouts;
  for (int64_t bankParam : {4, 2, 1}) {
    for (int64_t lineParam = 1; lineParam <= 32 / bankParam; lineParam *= 2) {
      layout = layout.setParams(bankParam, lineParam);
      sharedType = cloneWithLayout(sharedType, layout);
      if (isNoBankConflict(sharedType, tensorType, alignment))
        layouts.insert(layout);
    }
  }
  return layouts;
}
