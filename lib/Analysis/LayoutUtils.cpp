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
  case (MatmulImplWay::FMA): {
    auto accType = op.getAcc().getType();
    auto accLayout = getFragmentsLayout(accType);
    auto lhsShape = op.getLhs().getType().getShape();
    auto rhsShape = op.getRhs().getType().getShape();
    auto laneArray = accLayout.getLaneArray();
    auto lhsLayout =
        FragmentsLayoutAttr::get(context, laneArray, {1, lhsShape[1]}, 0, 1);
    auto rhsLayout =
        FragmentsLayoutAttr::get(context, laneArray, {rhsShape[0], 1}, 1, 0);
    return {lhsLayout, rhsLayout, accLayout};
  }

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
  if (hasLayout(sourceType) &&
      getLayout<SwizzlingLayoutAttr>(sourceType).isColMajor())
    loaderLayout = loaderLayout.transpose();
  return {loaderLayout, resultLayout};
}

/// Get candidate layouts for expensive global or shared memory access.
static SetVector<FragmentsLayoutAttr>
getCandidateLayoutsImpl(RankedTensorType tensorType, bool isGlobalAccess) {
  auto shape = tensorType.getShape();
  auto numElems = tensorType.getNumElements();
  auto *context = tensorType.getContext();
  auto layout = getLayout<FragmentsLayoutAttr>(tensorType);

  auto laneArray = layout.getLaneArray();
  auto laneLoops = layout.getLaneLoops();

  unsigned i = 0;
  unsigned j = 1;

  // Currently always use 128 bits instruction for shared memory access.
  auto vecWidth = isGlobalAccess ? laneLoops[layout.getMajorAxis()]
                                 : 128 / getIntOrFloatBitWidth(tensorType);
  auto restLoop = std::max<int64_t>(numElems / (warpSize * vecWidth), 1);

  SetVector<FragmentsLayoutAttr> layouts;
  layouts.insert(layout);

  for (int64_t numLanesI = 1; numLanesI <= warpSize; numLanesI *= 2) {
    auto numLanesJ = warpSize / numLanesI;
    laneArray[i] = numLanesI;
    laneArray[j] = numLanesJ;
    for (int64_t laneLoopI = 1; laneLoopI <= restLoop; laneLoopI *= 2) {
      auto laneLoopJ = vecWidth;
      if (numLanesI * laneLoopI > shape[i])
        break;
      if (numLanesJ * laneLoopJ > shape[j])
        break;
      laneLoops[i] = laneLoopI;
      laneLoops[j] = laneLoopJ;
      layouts.insert(
          FragmentsLayoutAttr::get(context, laneArray, laneLoops, i, j));
    }
  }

  i = 1;
  j = 0;

  for (int64_t numLanesI = 1; numLanesI <= warpSize; numLanesI *= 2) {
    auto numLanesJ = warpSize / numLanesI;
    laneArray[i] = numLanesI;
    laneArray[j] = numLanesJ;
    for (int64_t laneLoopI = 1; laneLoopI <= restLoop; laneLoopI *= 2) {
      auto laneLoopJ = vecWidth;
      if (numLanesI * laneLoopI > shape[i])
        break;
      if (numLanesJ * laneLoopJ > shape[j])
        break;
      laneLoops[i] = laneLoopI;
      laneLoops[j] = laneLoopJ;
      layouts.insert(
          FragmentsLayoutAttr::get(context, laneArray, laneLoops, i, j));
    }
  }

  return layouts;
}

SetVector<FragmentsLayoutAttr> kapy::getCandidateLayouts(Operation *op) {
  if (auto ldMatrixOp = dyn_cast<LdMatrixOp>(op)) {
    SetVector<FragmentsLayoutAttr> layouts;
    auto layout = getDefaultLayouts(ldMatrixOp)[1];
    layouts.insert(layout);
    layouts.insert(layout.transpose());
    return layouts;
  }
  if (isExpensiveGlobalRead(op) || isExpensiveGlobalWrite(op)) {
    RankedTensorType globalType;
    RankedTensorType tensorType;
    if (isExpensiveGlobalRead(op)) {
      globalType = cast<RankedTensorType>(op->getOperand(0).getType());
      tensorType = cast<RankedTensorType>(op->getResult(0).getType());
    } else {
      globalType = cast<RankedTensorType>(op->getOperand(0).getType());
      tensorType = cast<RankedTensorType>(op->getOperand(3).getType());
    }
    auto alignment = getAlignment(op);
    auto layouts = getCandidateLayoutsImpl(tensorType, true);
    for (auto layout : layouts) {
      tensorType = cloneWithLayout(tensorType, layout);
      if (!isCoalescedGlobalAccess(globalType, tensorType, alignment))
        layouts.remove(layout);
    }
    return layouts;
  }
  if (isExpensiveSharedRead(op) || isExpensiveSharedWrite(op)) {
    RankedTensorType sharedType;
    RankedTensorType tensorType;
    if (isExpensiveSharedRead(op)) {
      sharedType = cast<RankedTensorType>(op->getOperand(0).getType());
      tensorType = cast<RankedTensorType>(op->getResult(0).getType());
    } else {
      sharedType = cast<RankedTensorType>(op->getOperand(0).getType());
      tensorType = cast<RankedTensorType>(op->getOperand(3).getType());
    }
    auto layouts = getCandidateLayoutsImpl(tensorType, false);
    for (auto layout : layouts) {
      tensorType = cloneWithLayout(tensorType, layout);
      if (hasLayout(sharedType)) {
        if (!is0ConflictSharedAccess(sharedType, tensorType))
          layouts.remove(layout);
      } else {
        // For shared memory tensor have not set a layout, we try both row major
        // and column major swizzling layout.
        auto *context = sharedType.getContext();
        bool is0Conflict = false;
        sharedType = cloneWithLayout(sharedType,
                                     SwizzlingLayoutAttr::get(context, 0, 1));
        is0Conflict |= is0ConflictSharedAccess(sharedType, tensorType);
        sharedType = cloneWithLayout(sharedType,
                                     SwizzlingLayoutAttr::get(context, 1, 0));
        is0Conflict |= is0ConflictSharedAccess(sharedType, tensorType);
        if (!is0Conflict)
          layouts.remove(layout);
      }
    }
    return layouts;
  }
  llvm_unreachable("unsupported operation");
}

bool kapy::isCoalescedGlobalAccess(RankedTensorType globalType,
                                   RankedTensorType tensorType,
                                   int64_t alignment) {
  auto globalLayout = getLayout<Strided2dLayoutAttr>(globalType);
  auto tensorLayout = getLayout<FragmentsLayoutAttr>(tensorType);
  auto globalMap = globalLayout.getAffineMap();
  auto option = FragmentsLayoutAttr::MapOption::TO_TENSOR;
  auto tensorMap = tensorLayout.getAffineMap(tensorType.getShape(), option);
  auto accessMap = globalMap.compose(tensorMap);

  auto bitWidth = getIntOrFloatBitWidth(globalType);
  // Currently we assume that `alignment >= 16` so always 128 bits.
  // TODO: Handle `alignment < 16`.
  auto vecWidth = 128 / bitWidth;

  int64_t ivStep = 1;
  if (globalLayout.isRowMajor() && tensorLayout.isColMajor())
    ivStep = tensorLayout.getLoopSpace(tensorType.getShape())[0];
  if (globalLayout.isColMajor() && tensorLayout.isRowMajor())
    ivStep = tensorLayout.getLoopSpace(tensorType.getShape())[1];

  if (accessMap.getNumSymbols() == 0) {
    DenseSet<int64_t> cacheLineSet;
    for (int64_t laneId = 0; laneId < warpSize; ++laneId) {
      for (int64_t loopIv = 0; loopIv < vecWidth * ivStep; loopIv += ivStep) {
        auto bitOffset = accessMap.compose({laneId, loopIv})[0] * bitWidth;
        cacheLineSet.insert((bitOffset + alignment * 8) / 1024);
      }
    }
    if (alignment >= 128)
      return cacheLineSet.size() <= 4;
    else
      return cacheLineSet.size() <= 5;
  } else {
    DenseSet<AffineExpr> cacheLineSet;
    auto *context = accessMap.getContext();
    for (int64_t laneId = 0; laneId < warpSize; ++laneId) {
      for (int64_t loopIv = 0; loopIv < vecWidth * ivStep; loopIv += ivStep) {
        SmallVector<AffineExpr, 2> inputs;
        inputs.push_back(getAffineConstantExpr(laneId, context));
        inputs.push_back(getAffineConstantExpr(loopIv, context));
        auto inputsMap = AffineMap::get(0, 0, inputs, context);
        auto offsetMap = accessMap.compose(inputsMap);
        auto bitOffset = offsetMap.getResult(0) * bitWidth;
        cacheLineSet.insert((bitOffset + alignment * 8).floorDiv(1024));
      }
    }
    if (alignment >= 128)
      return cacheLineSet.size() <= 4;
    else
      return cacheLineSet.size() <= 5;
  }
}

bool kapy::is0ConflictSharedAccess(RankedTensorType sharedType,
                                   RankedTensorType tensorType) {
  auto sharedLayout = getLayout<SwizzlingLayoutAttr>(sharedType);
  auto tensorLayout = getLayout<FragmentsLayoutAttr>(tensorType);
  auto option = FragmentsLayoutAttr::MapOption::TO_TENSOR;
  auto tensorMap = tensorLayout.getAffineMap(tensorType.getShape(), option);

  auto bitWidth = getIntOrFloatBitWidth(sharedType);
  auto vecWidth = 128 / bitWidth;

  int64_t ivStep = 1;
  if (sharedLayout.isRowMajor() && tensorLayout.isColMajor())
    ivStep = tensorLayout.getLoopSpace(tensorType.getShape())[0];
  if (sharedLayout.isColMajor() && tensorLayout.isRowMajor())
    ivStep = tensorLayout.getLoopSpace(tensorType.getShape())[1];

  llvm::MapVector<int64_t, DenseSet<int64_t>> bankIdToLineIds;
  for (int64_t laneId = 0; laneId < quadSize; ++laneId) {
    for (int64_t loopIv = 0; loopIv < vecWidth * ivStep; loopIv += ivStep) {
      auto indices = tensorMap.compose({laneId, loopIv});
      int64_t bitOffset = bitWidth;
      if (sharedLayout.isRowMajor())
        bitOffset *= indices[0] * sharedType.getShape()[1] + indices[1];
      else
        bitOffset *= indices[0] + indices[1] * sharedType.getShape()[0];
      auto bankId = bitOffset % 1024 / 32;
      auto lineId = bitOffset / 1024;
      bankId = (bankId / 4 ^ lineId % 8) * 4 + bankId % 4;
      bankIdToLineIds[bankId].insert(lineId);
    }
  }
  for (const auto &it : bankIdToLineIds)
    if (it.second.size() > 1)
      return false;
  return true;
}
