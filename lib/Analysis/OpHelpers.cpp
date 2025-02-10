//===- OpHelpers.h ----------------------------------------------*- C++ -*-===//
//
// This file implements layout analysis for operations that need shared memory
// as scratch.
//
//===----------------------------------------------------------------------===//

#include "kapy/Analysis/OpHelpers.h"
#include "kapy/Dialect/Kapy/IR/Kapy.h"
#include "kapy/Dialect/Kgpu/IR/Kgpu.h"
#include "kapy/Support/CommonUtils.h"
#include "mlir/Analysis/FlatLinearValueConstraints.h"
#include "mlir/Analysis/Presburger/IntegerRelation.h"

using namespace mlir;
using namespace mlir::kapy;
using namespace mlir::presburger;

static constexpr char laneSyncCostAttrName[] = "kgpu.lane_sync_cost";
static constexpr char warpSyncCostAttrName[] = "kgpu.warp_sync_cost";

static std::optional<int64_t> getLaneSyncCostFromAttr(Operation *op) {
  if (!op->hasAttr(laneSyncCostAttrName))
    return std::nullopt;
  return cast<IntegerAttr>(op->getAttr(laneSyncCostAttrName)).getInt();
}

static std::optional<int64_t> getWarpSyncCostFromAttr(Operation *op) {
  if (!op->hasAttr(warpSyncCostAttrName))
    return std::nullopt;
  return cast<IntegerAttr>(op->getAttr(warpSyncCostAttrName)).getInt();
}

static LogicalResult getRelationFromMap(AffineMap map, IntegerRelation &rel) {
  // Get flattened affine expressions.
  std::vector<SmallVector<int64_t, 8>> flatExprs;
  FlatLinearConstraints set;
  if (failed(getFlattenedAffineExprs(map, &flatExprs, &set)))
    return failure();

  auto oldNumDims = set.getNumDimVars();
  auto oldNumCols = set.getNumCols();
  auto numRangeVars = map.getNumResults();
  auto numDomainVars = map.getNumDims();

  // Add range as the new expressions.
  set.appendDimVar(numRangeVars);
  // Set space for the set.
  set.setSpace(PresburgerSpace::getRelationSpace(numDomainVars, numRangeVars,
                                                 set.getNumSymbolVars(),
                                                 set.getNumLocalVars()));

  // Add equalities between domain and range.
  SmallVector<int64_t, 8> eq(set.getNumCols());
  for (unsigned i = 0; i < map.getNumResults(); ++i) {
    // Zero fill.
    std::fill(eq.begin(), eq.end(), 0);
    // Fill equality.
    for (unsigned j = 0; j < oldNumDims; ++j)
      eq[j] = flatExprs[i][j];
    for (unsigned j = oldNumDims; j < oldNumCols; ++j)
      eq[j + numRangeVars] = flatExprs[i][j];
    // Set this dimension to -1 to equate lhs and rhs and add equality.
    eq[numDomainVars + i] = -1;
    set.addEquality(eq);
  }

  rel = set;
  return success();
}

ReduceOpHelper::ReduceOpHelper(ReduceOp reduceOp) {
  this->sourceType = reduceOp.getSource().getType();
  this->axis = reduceOp.getAxis();
  auto maybeLaneSyncCost = getLaneSyncCostFromAttr(reduceOp);
  auto maybeWarpSyncCost = getWarpSyncCostFromAttr(reduceOp);
  if (maybeLaneSyncCost.has_value() && maybeWarpSyncCost.has_value()) {
    laneSyncCost = maybeLaneSyncCost.value();
    warpSyncCost = maybeWarpSyncCost.value();
  } else {
    runRelationAnalysis();
    auto i64Type = IntegerType::get(reduceOp.getContext(), 64);
    reduceOp->setAttr(laneSyncCostAttrName,
                      IntegerAttr::get(i64Type, laneSyncCost));
    reduceOp->setAttr(warpSyncCostAttrName,
                      IntegerAttr::get(i64Type, warpSyncCost));
  }
}

ReduceOpHelper::ReduceOpHelper(RankedTensorType sourceType, unsigned axis) {
  this->sourceType = sourceType;
  this->axis = axis;
  runRelationAnalysis();
}

int64_t ReduceOpHelper::getMinShflBflyOffset() const {
  auto sourceLayout = cast<FragmentsLayoutAttr>(sourceType.getEncoding());
  if (sourceLayout.isRowMajor() == (axis == 1))
    return 1;
  return sourceLayout.getLaneArray()[axis];
}

SmallVector<int64_t, 2> ReduceOpHelper::getScratchShape() const {
  if (warpSyncCost == 0)
    return {};
  auto scraNumRows = sourceType.getShape()[axis == 1 ? 0 : 1];
  auto scraNumCols = getScratchNumCols();
  return {scraNumRows, scraNumCols};
}

uint64_t ReduceOpHelper::getScratchSizeInBytes() const {
  if (warpSyncCost == 0)
    return 0;
  auto bitWidth = getIntOrFloatBitWidth(sourceType);
  return product(getScratchShape()) * ceilDiv<unsigned>(bitWidth, 8);
}

void ReduceOpHelper::runRelationAnalysis() {
  auto shape = sourceType.getShape();
  auto sourceLayout = cast<FragmentsLayoutAttr>(sourceType.getEncoding());

  auto sourceMap = sourceLayout.getAffineMap(shape, true);
  IntegerRelation reduceRel(PresburgerSpace::getRelationSpace(3, 2));
  if (failed(getRelationFromMap(sourceMap, reduceRel)))
    llvm_unreachable("failed to get relation from map");

  auto numWarps = product(sourceLayout.getWarpArray());
  auto loopSize = product(sourceLayout.getLoopSpace(shape));

  // Add bounds for warp id, lane id, loop iv.
  reduceRel.addBound(BoundType::LB, 0, 0);
  reduceRel.addBound(BoundType::UB, 0, numWarps - 1);
  reduceRel.addBound(BoundType::LB, 1, 0);
  reduceRel.addBound(BoundType::UB, 1, numLanes - 1);
  reduceRel.addBound(BoundType::LB, 2, 0);
  reduceRel.addBound(BoundType::UB, 2, loopSize - 1);

  // Add bounds for a reduction slice.
  for (unsigned i = 3; i < 5; ++i) {
    if (i - 3 != axis) {
      reduceRel.addBound(BoundType::EQ, i, 0);
    } else {
      reduceRel.addBound(BoundType::LB, i, 0);
      reduceRel.addBound(BoundType::UB, i, shape[axis] - 1);
    }
  }

  bool laneSynchronous = true;
  bool warpSynchronous = true;
  reduceRel.simplify();
  auto domainSet = reduceRel.getDomainSet();
  // Check if any other lane in this set, if so, not lane synchronous.
  for (int64_t laneId = 1; laneId < numLanes; ++laneId) {
    if (domainSet.containsPointNoLocal({0, laneId, 0, 0})) {
      laneSynchronous = false;
      break;
    }
  }
  // Check if any other warp in this set, if so, not warp synchronous.
  for (int64_t warpId = 1; warpId < numWarps; ++warpId) {
    if (domainSet.containsPointNoLocal({warpId, 0, 0, 0})) {
      warpSynchronous = false;
      break;
    }
  }
  if (!laneSynchronous) {
    auto laneArray = sourceLayout.getLaneArray();
    laneSyncCost = log2(laneArray[axis]);
  }
  if (!warpSynchronous) {
    unsigned r = axis == 1 ? 0 : 1;
    auto loopSpace = sourceLayout.getLoopSpace(shape);
    auto numLoadsR = std::min(loopSpace[r], shape[r]);
    auto vectorWidth = 128 / getIntOrFloatBitWidth(sourceType);
    auto numLoadsC = ceilDiv<int64_t>(getScratchNumCols(), vectorWidth);
    warpSyncCost = 1 + numLoadsR * numLoadsC;
  }
}

int64_t ReduceOpHelper::getScratchNumCols() const {
  auto shape = sourceType.getShape();
  auto sourceLayout = cast<FragmentsLayoutAttr>(sourceType.getEncoding());

  auto numWarpsC = sourceLayout.getWarpArray()[axis];
  auto warpLoopC = sourceLayout.getWarpLoops()[axis];
  auto numLanesC = sourceLayout.getLaneArray()[axis];
  auto laneLoopC = sourceLayout.getLaneLoops()[axis];

  auto scraNumCols = numWarpsC;
  if (numWarpsC * warpLoopC * numLanesC * laneLoopC > shape[axis])
    scraNumCols = shape[axis] / (warpLoopC * numLanesC * laneLoopC);
  return scraNumCols;
}

ChangeOpHelper::ChangeOpHelper(ChangeOp changeOp) {
  this->sourceType = changeOp.getSource().getType();
  this->resultType = changeOp.getType();
  auto maybeLaneSyncCost = getLaneSyncCostFromAttr(changeOp);
  auto maybeWarpSyncCost = getWarpSyncCostFromAttr(changeOp);
  if (maybeLaneSyncCost.has_value() && maybeWarpSyncCost.has_value()) {
    laneSyncCost = maybeLaneSyncCost.value();
    warpSyncCost = maybeWarpSyncCost.value();
  } else {
    runRelationAnalysis();
    auto i64Type = IntegerType::get(changeOp.getContext(), 64);
    changeOp->setAttr(laneSyncCostAttrName,
                      IntegerAttr::get(i64Type, laneSyncCost));
    changeOp->setAttr(warpSyncCostAttrName,
                      IntegerAttr::get(i64Type, warpSyncCost));
  }
}

ChangeOpHelper::ChangeOpHelper(RankedTensorType sourceType,
                               RankedTensorType resultType) {
  this->sourceType = sourceType;
  this->resultType = resultType;
  runRelationAnalysis();
}

bool ChangeOpHelper::useTransposedScratch() const {
  auto sourceLayout = cast<FragmentsLayoutAttr>(sourceType.getEncoding());
  auto resultLayout = cast<FragmentsLayoutAttr>(resultType.getEncoding());
  return std::min(sourceLayout.getLaneLoops()[0],
                  resultLayout.getLaneLoops()[0]) >
         std::min(sourceLayout.getLaneLoops()[1],
                  resultLayout.getLaneLoops()[1]);
}

SmallVector<int64_t, 2> ChangeOpHelper::getScratchShape() const {
  if (warpSyncCost == 0)
    return {};
  return llvm::to_vector<2>(sourceType.getShape());
}

uint64_t ChangeOpHelper::getScratchSizeInBytes() const {
  if (warpSyncCost == 0)
    return 0;
  auto bitWidth = getIntOrFloatBitWidth(sourceType);
  return product(getScratchShape()) * ceilDiv<unsigned>(bitWidth, 8);
}

void ChangeOpHelper::runRelationAnalysis() {
  auto sourceShape = sourceType.getShape();
  auto sourceLayout = cast<FragmentsLayoutAttr>(sourceType.getEncoding());
  auto resultLayout = cast<FragmentsLayoutAttr>(resultType.getEncoding());

  auto sourceMap = sourceLayout.getAffineMap(sourceShape, true);
  auto resultMap = resultLayout.getAffineMap(sourceShape, true);
  IntegerRelation sourceRel(PresburgerSpace::getRelationSpace(3, 2));
  IntegerRelation resultRel(PresburgerSpace::getRelationSpace(3, 2));
  if (failed(getRelationFromMap(sourceMap, sourceRel)) ||
      failed(getRelationFromMap(resultMap, resultRel)))
    llvm_unreachable("failed to get relation from map");

  auto numWarps = product(sourceLayout.getWarpArray());
  auto loopSize = product(sourceLayout.getLoopSpace(sourceShape));

  // Add bounds for warp id, lane id, loop iv.
  sourceRel.addBound(BoundType::LB, 0, 0);
  sourceRel.addBound(BoundType::UB, 0, numWarps - 1);
  sourceRel.addBound(BoundType::LB, 1, 0);
  sourceRel.addBound(BoundType::UB, 1, numLanes - 1);
  sourceRel.addBound(BoundType::LB, 2, 0);
  sourceRel.addBound(BoundType::UB, 2, loopSize - 1);

  loopSize = product(resultLayout.getLoopSpace(sourceShape));

  // Add bounds for warp id, lane id, loop iv.
  resultRel.addBound(BoundType::LB, 0, 0);
  resultRel.addBound(BoundType::UB, 0, numWarps - 1);
  resultRel.addBound(BoundType::LB, 1, 0);
  resultRel.addBound(BoundType::UB, 1, numLanes - 1);
  resultRel.addBound(BoundType::LB, 2, 0);
  resultRel.addBound(BoundType::UB, 2, loopSize - 1);

  // Add bounds for tensor shape.
  sourceRel.addBound(BoundType::LB, 3, 0);
  sourceRel.addBound(BoundType::UB, 3, sourceShape[0] - 1);
  sourceRel.addBound(BoundType::LB, 4, 0);
  sourceRel.addBound(BoundType::UB, 4, sourceShape[1] - 1);
  resultRel.addBound(BoundType::LB, 3, 0);
  resultRel.addBound(BoundType::UB, 3, sourceShape[0] - 1);
  resultRel.addBound(BoundType::LB, 4, 0);
  resultRel.addBound(BoundType::UB, 4, sourceShape[1] - 1);

  resultRel.inverse();
  sourceRel.compose(resultRel);

  bool warpSynchronous = true;
  // Add bound for warp 0.
  sourceRel.addBound(BoundType::EQ, 0, 0);
  sourceRel.simplify();
  auto resultSet = sourceRel.getRangeSet();
  // Check if any other warp in this set, if so, not warp synchronous.
  // TODO: This is expensive to prove it is warp synchronous.
  for (int64_t warpId = 1; warpId < numWarps; ++warpId) {
    if (!warpSynchronous)
      break;
    for (int64_t laneId = 0; laneId < numLanes; ++laneId) {
      if (!warpSynchronous)
        break;
      for (int64_t loopIv = 0; loopIv < loopSize; ++loopIv) {
        if (!warpSynchronous)
          break;
        if (resultSet.containsPointNoLocal({warpId, laneId, loopIv, 0}))
          warpSynchronous = false;
      }
    }
  }
  if (warpSynchronous) {
    // Add bound for lane 0.
    sourceRel.addBound(BoundType::EQ, 1, 0);
    sourceRel.simplify();
    resultSet = sourceRel.getRangeSet();
    // Compute the number of elements in other lane, for each of them we need to
    // perform a shuffle.
    for (int64_t laneId = 1; laneId < numLanes; ++laneId)
      for (int64_t loopIv = 0; loopIv < loopSize; ++loopIv)
        if (resultSet.containsPointNoLocal({0, laneId, loopIv, 0}))
          ++laneSyncCost;
  } else {
    // TODO: Compute this.
    warpSyncCost = 1;
  }
}

int64_t ChangeOpHelper::getScratchNumCols() const {
  auto sourceLayout = cast<FragmentsLayoutAttr>(sourceType.getEncoding());
  auto resultLayout = cast<FragmentsLayoutAttr>(resultType.getEncoding());
  bool transposed = useTransposedScratch();
  auto bitWidth = getIntOrFloatBitWidth(sourceType);
  // The number of columns for each memory transaction.
  int64_t memtNumCols = 1;

  auto laneArray = sourceLayout.getLaneArray();
  auto laneLoops = sourceLayout.getLaneLoops();
  if (transposed) {
    auto stNumBits = std::min<int64_t>(laneLoops[0] * bitWidth, 128);
    memtNumCols = std::min(laneArray[0], 1024 / stNumBits) * laneLoops[0];
  } else {
    auto stNumBits = std::min<int64_t>(laneLoops[1] * bitWidth, 128);
    memtNumCols = std::min(laneArray[1], 1024 / stNumBits) * laneLoops[1];
  }

  laneArray = resultLayout.getLaneArray();
  laneLoops = resultLayout.getLaneLoops();
  if (transposed) {
    auto ldNumBits = std::min<int64_t>(laneLoops[0] * bitWidth, 128);
    memtNumCols = std::max(
        memtNumCols, std::min(laneArray[0], 1024 / ldNumBits) * laneLoops[0]);
  } else {
    auto ldNumBits = std::min<int64_t>(laneLoops[1] * bitWidth, 128);
    memtNumCols = std::max(
        memtNumCols, std::min(laneArray[1], 1024 / ldNumBits) * laneLoops[1]);
  }

  // TODO: We can avoid bank conflicts by swizzling instead of padding.
  if (transposed)
    return sourceType.getShape()[0] + memtNumCols;
  else
    return sourceType.getShape()[1] + memtNumCols;
}
