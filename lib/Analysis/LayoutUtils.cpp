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

std::array<FragmentsLayoutAttr, 3> kapy::getOperandLayouts(MatmulOp matmulOp) {
  auto implWay = matmulOp.getMatmulImplWay();
  auto accType = matmulOp.getAcc().getType();
  auto lhsShape = matmulOp.getLhs().getType().getShape();
  auto rhsShape = matmulOp.getRhs().getType().getShape();
  auto accShape = accType.getShape();
  auto numElems = accType.getNumElements();
  auto *context = matmulOp.getContext();
  SmallVector<int64_t, 2> laneLoops{2, 2};
  if (implWay == MatmulImplWay::FMA) {
    if (accShape[0] >= 32 && accShape[1] >= 32 && numElems / warpSize >= 16)
      laneLoops = {4, 4};
    auto accLayout = getFragmentsLayout(laneLoops, accType);
    auto laneArray = accLayout.getLaneArray();
    auto lhsLayout = FragmentsLayoutAttr::get(
        context, laneArray, {laneLoops[0], lhsShape[1]}, 0, 1);
    auto rhsLayout = FragmentsLayoutAttr::get(
        context, laneArray, {rhsShape[0], laneLoops[1]}, 1, 0);
    return {lhsLayout, rhsLayout, accLayout};
  }
  auto accLayout = FragmentsLayoutAttr::get(context, {8, 4}, {1, 2}, 0, 1);
  if (implWay == MatmulImplWay::MMA_M16N8K8_F16 ||
      implWay == MatmulImplWay::MMA_M16N8K16_F16) {
    auto lhsLayout = FragmentsLayoutAttr::get(context, {8, 4}, {1, 2}, 0, 1);
    auto rhsLayout = FragmentsLayoutAttr::get(context, {4, 8}, {2, 1}, 1, 0);
    return {lhsLayout, rhsLayout, accLayout};
  }
  if (implWay == MatmulImplWay::MMA_M16N8K8_TF32) {
    auto lhsLayout = FragmentsLayoutAttr::get(context, {8, 4}, {1, 1}, 0, 1);
    auto rhsLayout = FragmentsLayoutAttr::get(context, {4, 8}, {1, 1}, 1, 0);
    return {lhsLayout, rhsLayout, accLayout};
  }
  if (implWay == MatmulImplWay::MMA_M16N8K16_F8) {
    auto lhsLayout = FragmentsLayoutAttr::get(context, {8, 4}, {1, 4}, 0, 1);
    auto rhsLayout = FragmentsLayoutAttr::get(context, {4, 8}, {4, 1}, 1, 0);
    return {lhsLayout, rhsLayout, accLayout};
  }
  llvm_unreachable("unsupported matmul implement way");
}

static SetVector<FragmentsLayoutAttr>
getCandidateLayoutsImpl(RankedTensorType tensorType) {
  auto shape = tensorType.getShape();
  auto numElems = tensorType.getNumElements();
  auto layout = cast<FragmentsLayoutAttr>(tensorType.getEncoding());

  auto laneArray = layout.getLaneArray();
  auto laneLoops = layout.getLaneLoops();

  unsigned i = 0;
  unsigned j = 1;

  auto vecWidth = laneLoops[layout.getMajorAxis()];
  auto restLoop = std::max<int64_t>(numElems / (warpSize * vecWidth), 1);

  auto *context = tensorType.getContext();
  SetVector<FragmentsLayoutAttr> layouts;
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
  if (auto matmulOp = dyn_cast<MatmulOp>(op)) {
    if (matmulOp.getMatmulImplWay() == MatmulImplWay::FMA) {
      auto layouts = getCandidateLayoutsImpl(matmulOp.getType());
      for (auto layout : layouts) {
        auto laneLoops = layout.getLaneLoops();
        if (laneLoops[0] != laneLoops[1])
          layouts.remove(layout);
      }
      return layouts;
    }
    return {};
  }
  if (isExpensiveGlobalRead(op) || isExpensiveGlobalWrite(op)) {
    auto alignment = getAlignment(op);
    GlobalMemRefType globalType;
    RankedTensorType tensorType;
    if (isExpensiveGlobalRead(op)) {
      globalType = cast<GlobalMemRefType>(op->getOperand(0).getType());
      tensorType = cast<RankedTensorType>(op->getResult(0).getType());
    } else {
      globalType = cast<GlobalMemRefType>(op->getOperand(0).getType());
      tensorType = cast<RankedTensorType>(op->getOperand(3).getType());
    }
    auto layouts = getCandidateLayoutsImpl(tensorType);
    for (auto layout : layouts) {
      tensorType = RankedTensorType::get(tensorType.getShape(),
                                         tensorType.getElementType(), layout);
      if (!isCoalescedGlobalAccess(globalType, tensorType, alignment))
        layouts.remove(layout);
    }
    return layouts;
  }
  if (isExpensiveSharedRead(op) || isExpensiveSharedWrite(op)) {
    SharedMemRefType sharedType;
    RankedTensorType tensorType;
    if (isExpensiveSharedRead(op)) {
      sharedType = cast<SharedMemRefType>(op->getOperand(0).getType());
      tensorType = cast<RankedTensorType>(op->getResult(0).getType());
    } else {
      sharedType = cast<SharedMemRefType>(op->getOperand(0).getType());
      tensorType = cast<RankedTensorType>(op->getOperand(3).getType());
    }
    auto layouts = getCandidateLayoutsImpl(tensorType);
    for (auto layout : layouts) {
      tensorType = RankedTensorType::get(tensorType.getShape(),
                                         tensorType.getElementType(), layout);
      if (!is0ConflictSharedAccess(sharedType, tensorType))
        layouts.remove(layout);
    }
    return layouts;
  }
  llvm_unreachable("unsupported operation");
}

bool kapy::isCoalescedGlobalAccess(GlobalMemRefType globalType,
                                   RankedTensorType tensorType,
                                   int64_t alignment) {
  auto globalLayout = cast<Strided2dLayoutAttr>(globalType.getEncoding());
  auto tensorLayout = cast<FragmentsLayoutAttr>(tensorType.getEncoding());
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

bool kapy::is0ConflictSharedAccess(SharedMemRefType sharedType,
                                   RankedTensorType tensorType) {
  auto tensorLayout = cast<FragmentsLayoutAttr>(tensorType.getEncoding());
  auto option = FragmentsLayoutAttr::MapOption::TO_TENSOR;
  auto tensorMap = tensorLayout.getAffineMap(tensorType.getShape(), option);
  auto bitWidth = getIntOrFloatBitWidth(sharedType);
  auto vecWidth = 128 / bitWidth;
  if (auto sharedLayout =
          dyn_cast<Strided2dLayoutAttr>(sharedType.getEncoding())) {
    auto sharedMap = sharedLayout.getAffineMap();
    auto accessMap = sharedMap.compose(tensorMap);

    int64_t ivStep = 1;
    if (sharedLayout.isRowMajor() && tensorLayout.isColMajor())
      ivStep = tensorLayout.getLoopSpace(tensorType.getShape())[0];
    if (sharedLayout.isColMajor() && tensorLayout.isRowMajor())
      ivStep = tensorLayout.getLoopSpace(tensorType.getShape())[1];

    llvm::MapVector<int64_t, DenseSet<int64_t>> bankIdToLineIds;
    for (int64_t laneId = 0; laneId < quadSize; ++laneId) {
      for (int64_t loopIv = 0; loopIv < vecWidth * ivStep; loopIv += ivStep) {
        auto bitOffset = accessMap.compose({laneId, loopIv})[0] * bitWidth;
        auto bankId = bitOffset % 1024 / 32;
        auto lineId = bitOffset / 1024;
        bankIdToLineIds[bankId].insert(lineId);
      }
    }
    for (const auto &it : bankIdToLineIds)
      if (it.second.size() > 1)
        return false;
    return true;
  }
  if (auto sharedLayout =
          dyn_cast<Swizzled4LayoutAttr>(sharedType.getEncoding())) {
    int64_t ivStep = 1;
    if (sharedLayout.isRowMajor() && tensorLayout.isColMajor())
      ivStep = tensorLayout.getLoopSpace(tensorType.getShape())[0];
    if (sharedLayout.isColMajor() && tensorLayout.isRowMajor())
      ivStep = tensorLayout.getLoopSpace(tensorType.getShape())[1];

    llvm::MapVector<int64_t, DenseSet<int64_t>> bankIdToLineIds;
    for (int64_t laneId = 0; laneId < quadSize; ++laneId) {
      for (int64_t loopIv = 0; loopIv < vecWidth * ivStep; loopIv += ivStep) {
        auto indices = tensorMap.compose({laneId, loopIv});
        auto bitOffset = static_cast<int64_t>(bitWidth);
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
  llvm_unreachable("unsupported layout");
}
