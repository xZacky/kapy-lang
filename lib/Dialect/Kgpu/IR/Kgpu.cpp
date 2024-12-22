//===- Kgpu.cpp -------------------------------------------------*- C++ -*-===//
//
// This file implements the functions in the kgpu dialect.
//
//===----------------------------------------------------------------------===//

#include "kapy/Dialect/Kgpu/IR/Kgpu.h"
#include "kapy/Dialect/Kapy/IR/Utils.h"
#include "kapy/Dialect/Kgpu/IR/Utils.h"

#include "mlir/IR/DialectImplementation.h"
#include "mlir/IR/OpImplementation.h"
#include "llvm/ADT/TypeSwitch.h"

using namespace mlir;
using namespace mlir::kapy;

#include "kapy/Dialect/Kgpu/IR/Dialect.cpp.inc"

#define GET_ATTRDEF_CLASSES
#include "kapy/Dialect/Kgpu/IR/Attrs.cpp.inc"

#define GET_TYPEDEF_CLASSES
#include "kapy/Dialect/Kgpu/IR/Types.cpp.inc"

#define GET_OP_CLASSES
#include "kapy/Dialect/Kgpu/IR/Ops.cpp.inc"

namespace {
class KgpuOpAsmInterface : public OpAsmDialectInterface {
public:
  using OpAsmDialectInterface::OpAsmDialectInterface;

  virtual AliasResult getAlias(Attribute attr,
                               llvm::raw_ostream &os) const override {
    if (isa<RegistersLayoutAttr>(attr)) {
      os << "regs";
      return AliasResult::FinalAlias;
    }
    if (isa<SharedMemLayoutAttr>(attr)) {
      os << "smem";
      return AliasResult::FinalAlias;
    }
    if (isa<NvidiaMmaLayoutAttr>(attr)) {
      os << "nvmma";
      return AliasResult::FinalAlias;
    }
    if (isa<DotOpLoadLayoutAttr>(attr)) {
      os << "dotld";
      return AliasResult::FinalAlias;
    }
    return OpAsmDialectInterface::getAlias(attr, os);
  }
};

class KgpuLayoutInterface : public KapyLayoutInterface {
public:
  using KapyLayoutInterface::KapyLayoutInterface;

  virtual FailureOr<Attribute>
  inferReduceOpLayout(Value operand, int axis,
                      std::optional<Location> loc) const override {
    auto operandType = cast<RankedTensorType>(operand.getType());
    auto operandLayout =
        dyn_cast<RegistersLayoutAttr>(operandType.getEncoding());
    if (!operandLayout)
      return emitOptionalError(loc, "operand must have registers layout");
    return RegistersLayoutAttr::get(operandLayout.getContext(),
                                    operandLayout.getMap().dropResult(axis));
  }

  virtual FailureOr<Attribute>
  inferUnsqueezeOpLayout(Value operand, int axis,
                         std::optional<Location> loc) const override {
    auto operandType = cast<RankedTensorType>(operand.getType());
    auto operandLayout =
        dyn_cast<RegistersLayoutAttr>(operandType.getEncoding());
    if (!operandLayout)
      return emitOptionalError(loc, "operand must have registers layout");
    auto *context = operandLayout.getContext();
    auto zeroExpr = getAffineConstantExpr(0, context);
    return RegistersLayoutAttr::get(
        context, operandLayout.getMap().insertResult(zeroExpr, axis));
  }

  virtual FailureOr<Attribute>
  inferBroadcastOpLayout(Value operand, ArrayRef<int64_t> shape,
                         std::optional<Location> loc) const override {
    auto operandType = cast<RankedTensorType>(operand.getType());
    auto operandLayout =
        dyn_cast<RegistersLayoutAttr>(operandType.getEncoding());
    if (!operandLayout)
      return emitOptionalError(loc, "operand must have registers layout");
    SmallVector<int, 4> broadcastAxes;
    SmallVector<int64_t, 4> broadcastSizes;
    for (int i = 0; i < shape.size(); ++i) {
      if (operandType.getShape()[i] == 1 && shape[i] != 1) {
        broadcastAxes.push_back(i);
        broadcastSizes.push_back(shape[i]);
      }
    }
    auto numWarps = KgpuDialect::getNumWarps(getModule(operand));
    auto maxElemId =
        operandLayout.getMaxElementId(operandType.getShape(), numWarps);
    auto *context = operandLayout.getContext();
    auto elemIdExpr = getAffineDimExpr(0, context);
    auto laneIdExpr = getAffineDimExpr(1, context);
    auto warpIdExpr = getAffineDimExpr(2, context);
    auto inputMap = AffineMap::get(
        3, 0, {elemIdExpr % maxElemId, laneIdExpr, warpIdExpr}, context);
    auto exprs = llvm::to_vector<4>(
        operandLayout.getMap().compose(inputMap).getResults());
    auto broadcastExprs =
        delinearize(elemIdExpr.floorDiv(maxElemId), broadcastSizes);
    for (auto it : llvm::enumerate(broadcastAxes))
      exprs[it.value()] = broadcastExprs[it.index()];
    return RegistersLayoutAttr::get(context,
                                    AffineMap::get(3, 0, exprs, context));
  }

  virtual FailureOr<Attribute>
  inferPermuteOpLayout(Value operand, ArrayRef<int32_t> order,
                       std::optional<Location> loc) const override {
    auto operandType = cast<RankedTensorType>(operand.getType());
    auto operandLayout =
        dyn_cast<RegistersLayoutAttr>(operandType.getEncoding());
    if (!operandLayout)
      return emitOptionalError(loc, "operand must have registers layout");
    auto exprs = permute(operandLayout.getMap().getResults(), order);
    auto *context = operandLayout.getContext();
    return RegistersLayoutAttr::get(context,
                                    AffineMap::get(3, 0, exprs, context));
  }

  virtual FailureOr<Attribute>
  inferReshapeOpLayout(Value operand, ArrayRef<int64_t> shape,
                       std::optional<Location> loc) const override {
    auto operandType = cast<RankedTensorType>(operand.getType());
    auto operandLayout =
        dyn_cast<RegistersLayoutAttr>(operandType.getEncoding());
    if (!operandLayout)
      return emitOptionalError(loc, "operand must have registers layout");
    auto expr =
        linearize(operandLayout.getMap().getResults(), operandType.getShape());
    auto *context = operandLayout.getContext();
    return RegistersLayoutAttr::get(
        context, AffineMap::get(3, 0, delinearize(expr, shape), context));
  }

  virtual LogicalResult verifyDotOpLayouts(DotOp op) const override {
    auto lhsType = op.getLhs().getType();
    auto rhsType = op.getRhs().getType();
    auto accumType = op.getAccum().getType();

    auto lhsLayout = dyn_cast<DotOpLoadLayoutAttr>(lhsType.getEncoding());
    if (!lhsLayout)
      return op->emitOpError("lhs must have dot op load layout");
    if (lhsLayout.getOperandIndex() != 0)
      return op->emitOpError("lhs layout with wrong operand index");

    auto rhsLayout = dyn_cast<DotOpLoadLayoutAttr>(rhsType.getEncoding());
    if (!rhsLayout)
      return op->emitOpError("rhs must have dot op load layout");
    if (rhsLayout.getOperandIndex() != 1)
      return op->emitOpError("rhs layout with wrong operand index");

    auto accumLayout = accumType.getEncoding();
    if (isa<NvidiaMmaLayoutAttr, RegistersLayoutAttr>(accumLayout)) {
      if (lhsLayout.getParent() != accumLayout)
        return op->emitOpError("mismatch layouts between lhs and accum");
      if (rhsLayout.getParent() != accumLayout)
        return op->emitOpError("mismatch layouts between rhs and accum");
      return success();
    }
    return op->emitOpError("invalid accum layout");
  }
};
} // namespace

void KgpuDialect::initialize() {
  addAttributes<
#define GET_ATTRDEF_LIST
#include "kapy/Dialect/Kgpu/IR/Attrs.cpp.inc"
      >();
  addTypes<
#define GET_TYPEDEF_LIST
#include "kapy/Dialect/Kgpu/IR/Types.cpp.inc"
      >();
  addOperations<
#define GET_OP_LIST
#include "kapy/Dialect/Kgpu/IR/Ops.cpp.inc"
      >();
  addInterfaces<KgpuOpAsmInterface>();
  addInterfaces<KgpuLayoutInterface>();
}

ModuleOp kapy::getModule(Operation *op) {
  return op->getParentOfType<ModuleOp>();
}

ModuleOp kapy::getModule(Value value) {
  if (auto *defOp = value.getDefiningOp())
    return getModule(defOp);
  if (auto blockArg = dyn_cast<BlockArgument>(value))
    return getModule(blockArg.getOwner()->getParentOp());
  llvm_unreachable("invalid value");
}

bool kapy::supportNvidiaMma(DotOp dotOp) {
  auto lhsElementType = dotOp.getLhs().getType().getElementType();
  auto rhsElementType = dotOp.getRhs().getType().getElementType();
  if (lhsElementType.isF32() && rhsElementType.isF32()) {
    auto precision = dotOp.getDotPrecision();
    return precision == DotPrecision::TF32 || precision == DotPrecision::TF32x3;
  }
  return supportNvidiaMma(lhsElementType) && supportNvidiaMma(rhsElementType);
}

bool kapy::supportNvidiaMma(Type elementType) {
  bool isF8 = elementType.isFloat8E4M3FNUZ() || elementType.isFloat8E5M2() ||
              elementType.isFloat8E5M2FNUZ();
  return isF8 || elementType.isBF16() || elementType.isF16() ||
         elementType.isF32() || elementType.isInteger(8);
}

static int getIdStrWidth(int id) {
  // Maximum id supported now is 9999.
  assert(id >= 0 && id < 10000);
  if (id >= 1000)
    return 4;
  if (id >= 100)
    return 3;
  if (id >= 10)
    return 2;
  return 1;
}

static std::string getElementString(ArrayRef<SmallVector<int, 2>> tuples,
                                    int maxTuples, int maxTidStrWidth,
                                    int maxEidStrWidth) {
  // `T` tid `:` eid
  auto maxTupleStrWidth = 1 + maxTidStrWidth + 1 + maxEidStrWidth;
  std::string result = "";
  for (int i = 0; i < maxTuples; ++i) {
    if (i > 0)
      result += "|";
    if (i < tuples.size()) {
      auto tid = tuples[i][1];
      auto tidStrWidth = getIdStrWidth(tid);
      for (int j = 0; j < maxTidStrWidth - tidStrWidth; ++j)
        result += " ";
      result += "T" + std::to_string(tid) + ":";

      auto eid = tuples[i][0];
      auto eidStrWidth = getIdStrWidth(eid);
      for (int j = 0; j < maxEidStrWidth - eidStrWidth; ++j)
        result += " ";
      result += std::to_string(eid);
    } else {
      for (int j = 0; j < maxTupleStrWidth; ++j)
        result += "-";
    }
  }
  return result;
}

static std::string getLinePrefixString(int64_t index,
                                       ArrayRef<int64_t> strides) {
  std::string result = index == 0 ? "[" : "\n ";
  for (int i = 1; i < strides.size(); ++i)
    result += index % strides[i] == 0 ? "[" : " ";
  return result;
}

static std::string getLineSuffixString(int64_t index, int64_t numElems,
                                       ArrayRef<int64_t> strides) {
  std::string result = "";
  for (int i = 1; i < strides.size(); ++i)
    if ((index + 1) % strides[i] == 0)
      result += "]";
  if (index + 1 == numElems)
    result += "]";
  else
    result += ",";
  return result;
}

std::string kapy::getLayoutString(RankedTensorType type, int numWarps) {
  auto rank = type.getRank();
  auto shape = type.getShape();
  auto numElems = product(shape);
  if (auto regsLayout = dyn_cast<RegistersLayoutAttr>(type.getEncoding())) {
    // Map from linearized tensor index to id tuples.
    // Note that one tensor element may hold by multiple threads.
    std::vector<SmallVector<SmallVector<int, 2>>> indexToTuples(numElems);
    auto maxElemId = regsLayout.getMaxElementId(shape, numWarps);
    auto map = regsLayout.getMap();
    auto maxTuples = 1;
    for (int elemId = 0; elemId < maxElemId; ++elemId) {
      for (int laneId = 0; laneId < numLanes; ++laneId) {
        for (int warpId = 0; warpId < numWarps; ++warpId) {
          auto indices = map.compose({elemId, laneId, warpId});
          auto index = linearize(indices, shape);
          auto threadId = laneId + warpId * numLanes;
          indexToTuples[index].push_back({elemId, threadId});
          maxTuples = std::max<int>(maxTuples, indexToTuples[index].size());
        }
      }
    }
    std::string result = "";
    auto maxTidStrWidth = getIdStrWidth(numLanes * numWarps);
    auto maxEidStrWidth = getIdStrWidth(maxElemId);
    auto strides = computeStrides(shape);
    for (int64_t index = 0; index < numElems; ++index) {
      // line prefix
      if (index % shape[rank - 1] == 0)
        result += getLinePrefixString(index, strides);
      // element
      result += getElementString(indexToTuples[index], maxTuples,
                                 maxTidStrWidth, maxEidStrWidth);
      // line suffix
      if ((index + 1) % shape[rank - 1] == 0)
        result += getLineSuffixString(index, numElems, strides);
      // ", " after each element (except end of line)
      else
        result += ", ";
    }
    return result;
  }
  llvm_unreachable("unsupported layout");
}

static SmallVector<AffineExpr, 4> computeIndexExprs(MLIRContext *context,
                                                    ArrayRef<int> tilePerLane,
                                                    ArrayRef<int> lanePerWarp,
                                                    ArrayRef<int> warpPerCTA,
                                                    ArrayRef<int> replication) {
  auto rank = tilePerLane.size();
  SmallVector<int, 4> elemPerLane(rank);
  for (int i = 0; i < rank; ++i)
    elemPerLane[i] = tilePerLane[i] * replication[i];

  auto elemIdExpr = getAffineDimExpr(0, context);
  auto laneIdExpr = getAffineDimExpr(1, context);
  auto warpIdExpr = getAffineDimExpr(2, context);

  auto elemIdExprs = delinearize(elemIdExpr, elemPerLane);
  auto laneIdExprs = delinearize(laneIdExpr, lanePerWarp);
  auto warpIdExprs = delinearize(warpIdExpr, warpPerCTA);

  SmallVector<AffineExpr, 4> indexExprs(rank);
  for (int i = 0; i < rank; ++i)
    indexExprs[i] = elemIdExprs[i] % tilePerLane[i] +
                    laneIdExprs[i] * tilePerLane[i] +
                    warpIdExprs[i] * (tilePerLane[i] * lanePerWarp[i]) +
                    elemIdExprs[i].floorDiv(tilePerLane[i]) *
                        (tilePerLane[i] * lanePerWarp[i] * warpPerCTA[i]);
  return indexExprs;
}

RegistersLayoutAttr RegistersLayoutAttr::get(MLIRContext *context,
                                             ArrayRef<int> tilePerLane,
                                             ArrayRef<int> lanePerWarp,
                                             ArrayRef<int> warpPerCTA,
                                             ArrayRef<int> replication) {
  assert(tilePerLane.size() == lanePerWarp.size());
  assert(tilePerLane.size() == warpPerCTA.size());
  assert(tilePerLane.size() == replication.size());
  auto rank = tilePerLane.size();
  auto indexExprs = computeIndexExprs(context, tilePerLane, lanePerWarp,
                                      warpPerCTA, replication);
  return get(context, AffineMap::get(3, 0, indexExprs, context));
}

RegistersLayoutAttr RegistersLayoutAttr::get(MLIRContext *context,
                                             ArrayRef<int64_t> shape,
                                             ArrayRef<int> tilePerLane,
                                             ArrayRef<int> lanePerWarp,
                                             ArrayRef<int> warpPerCTA) {
  assert(shape.size() == tilePerLane.size());
  assert(shape.size() == lanePerWarp.size());
  assert(shape.size() == warpPerCTA.size());
  auto rank = shape.size();
  SmallVector<int, 4> replication(rank, 1);
  for (int i = 0; i < rank; ++i) {
    auto tilePerCTAI = tilePerLane[i] * lanePerWarp[i] * warpPerCTA[i];
    replication[i] = ceilDiv<int>(shape[i], tilePerCTAI);
  }
  auto indexExprs = computeIndexExprs(context, tilePerLane, lanePerWarp,
                                      warpPerCTA, replication);
  return get(context, AffineMap::get(3, 0, indexExprs, context));
}

RegistersLayoutAttr RegistersLayoutAttr::get(MLIRContext *context,
                                             ArrayRef<int64_t> shape,
                                             ArrayRef<int> tilePerLane,
                                             int numWarps) {
  assert(shape.size() == tilePerLane.size());
  auto rank = shape.size();
  SmallVector<int, 4> lanePerWarp(rank);
  SmallVector<int, 4> warpPerCTA(rank);
  auto restThreads = numLanes * numWarps;
  auto restLanes = numLanes;
  auto restWarps = numWarps;
  for (int i = rank - 1; i > 0; --i) {
    auto numThreadsI =
        std::clamp<int>(restThreads, 1, shape[i] / tilePerLane[i]);
    lanePerWarp[i] = std::clamp<int>(numThreadsI, 1, restLanes);
    warpPerCTA[i] = std::clamp<int>(numThreadsI / lanePerWarp[i], 1, restWarps);
    restThreads /= (lanePerWarp[i] * warpPerCTA[i]);
    restLanes /= lanePerWarp[i];
    restWarps /= warpPerCTA[i];
  }
  // Make the axis 0 to fill the rest lanes and warps.
  lanePerWarp[0] = restLanes;
  warpPerCTA[0] = restWarps;
  return get(context, shape, tilePerLane, lanePerWarp, warpPerCTA);
}

static bool insideTensor(ArrayRef<int64_t> shape, AffineMap map, int elemId,
                         int laneId, int warpId) {
  auto indices = map.compose({elemId, laneId, warpId});
  for (auto it : llvm::enumerate(indices))
    if (it.value() >= shape[it.index()])
      return false;
  return true;
}

int RegistersLayoutAttr::getMaxElementId(ArrayRef<int64_t> shape,
                                         int numWarps) const {
  auto numElems = product(shape);
  for (int elemId = 0; elemId < numElems; ++elemId) {
    bool isMaxElemId = true;
    for (int laneId = 0; laneId < numLanes; ++laneId) {
      if (!isMaxElemId)
        continue;
      for (int warpId = 0; warpId < numWarps; ++warpId) {
        if (!isMaxElemId)
          continue;
        if (insideTensor(shape, getMap(), elemId, laneId, warpId))
          isMaxElemId = false;
      }
    }
    if (isMaxElemId)
      return elemId;
  }
  return numElems;
}

int RegistersLayoutAttr::getMaxLaneId(ArrayRef<int64_t> shape,
                                      int numWarps) const {
  auto maxElemId = getMaxElementId(shape, numWarps);
  for (int laneId = 0; laneId < numLanes; ++laneId) {
    bool isMaxLaneId = true;
    for (int elemId = 0; elemId < maxElemId; ++elemId) {
      if (!isMaxLaneId)
        continue;
      for (int warpId = 0; warpId < numWarps; ++warpId) {
        if (!isMaxLaneId)
          continue;
        if (insideTensor(shape, getMap(), elemId, laneId, warpId))
          isMaxLaneId = false;
      }
    }
    if (isMaxLaneId)
      return laneId;
  }
  return numLanes;
}

int RegistersLayoutAttr::getMaxWarpId(ArrayRef<int64_t> shape,
                                      int numWarps) const {
  auto maxElemId = getMaxElementId(shape, numWarps);
  for (auto warpId = 0; warpId < numWarps; ++warpId) {
    bool isMaxWarpId = true;
    for (auto elemId = 0; elemId < maxElemId; ++elemId) {
      if (!isMaxWarpId)
        continue;
      for (auto laneId = 0; laneId < numLanes; ++laneId) {
        if (!isMaxWarpId)
          continue;
        if (insideTensor(shape, getMap(), elemId, laneId, warpId))
          isMaxWarpId = false;
      }
    }
    if (isMaxWarpId)
      return warpId;
  }
  return numWarps;
}

RegistersLayoutAttr
NvidiaMmaLayoutAttr::toRegistersLayout(ArrayRef<int64_t> shape) const {
  return RegistersLayoutAttr::get(getContext(), shape, getTilePerLane(),
                                  getLanePerWarp(), getWarpPerCTARef());
}

SmallVector<int, 4> NvidiaMmaLayoutAttr::getTilePerLane() const {
  auto rank = getRank();
  SmallVector<int, 4> tilePerLane(rank, 1);
  tilePerLane[rank - 1] = 2;
  return tilePerLane;
}

SmallVector<int, 4> NvidiaMmaLayoutAttr::getLanePerWarp() const {
  auto rank = getRank();
  SmallVector<int, 4> lanePerWarp(rank, 1);
  lanePerWarp[rank - 1] = 4;
  lanePerWarp[rank - 2] = 8;
  return lanePerWarp;
}

SmallVector<int, 4> NvidiaMmaLayoutAttr::getWarpPerCTA() const {
  return SmallVector<int, 4>(getWarpPerCTARef());
}

SmallVector<int, 4>
NvidiaMmaLayoutAttr::getReplication(ArrayRef<int64_t> shape) const {
  auto rank = getRank();
  auto tilePerLane = getTilePerLane();
  auto lanePerWarp = getLanePerWarp();
  auto warpPerCTA = getWarpPerCTA();
  SmallVector<int, 4> replication(rank);
  for (int i = 0; i < rank; ++i) {
    auto tilePerCTAI = tilePerLane[i] * lanePerWarp[i] * warpPerCTA[i];
    replication[i] = ceilDiv<int>(shape[i], tilePerCTAI);
  }
  return replication;
}

SmallVector<int, 4>
NvidiaMmaLayoutAttr::getTilePerLaneForChild(int bitWidth) const {
  auto rank = getRank();
  SmallVector<int, 4> tilePerLane(rank, 1);
  tilePerLane[rank - 1] = 128 / bitWidth;
  return tilePerLane;
}

SmallVector<int, 4> NvidiaMmaLayoutAttr::getLanePerWarpForChild() const {
  auto rank = getRank();
  SmallVector<int, 4> lanePerWarp(rank, 1);
  lanePerWarp[rank - 1] = 2;
  lanePerWarp[rank - 2] = 16;
  return lanePerWarp;
}

SmallVector<int, 4>
NvidiaMmaLayoutAttr::getReplicationForChild(int index, int bitWidth,
                                            ArrayRef<int64_t> shape) const {
  auto rank = getRank();
  auto tilePerLane = getTilePerLaneForChild(bitWidth);
  auto lanePerWarp = getLanePerWarpForChild();
  auto warpPerCTA = getWarpPerCTA();
  auto kAxis = index == 0 ? rank - 1 : rank - 2;
  // Warps hold the same data on the k axis, so when we compute replication, we
  // set it to 1.
  warpPerCTA[kAxis] = 1;
  SmallVector<int, 4> replication(rank);
  for (int i = 0; i < rank; ++i) {
    auto tilePerCTAI = tilePerLane[i] * lanePerWarp[i] * warpPerCTA[i];
    replication[i] = ceilDiv<int>(shape[i], tilePerCTAI);
  }
  return replication;
}

int DotOpLoadLayoutAttr::getRank() const {
  if (auto nvmmaLayout = llvm::dyn_cast<NvidiaMmaLayoutAttr>(getParent()))
    return nvmmaLayout.getRank();
  if (auto regsLayout = llvm::dyn_cast<RegistersLayoutAttr>(getParent()))
    return regsLayout.getMap().getNumResults();
  llvm_unreachable("invalid parent layout");
}

RegistersLayoutAttr
DotOpLoadLayoutAttr::toRegistersLayout(ArrayRef<int64_t> shape) const {
  if (auto nvmmaLayout = llvm::dyn_cast<NvidiaMmaLayoutAttr>(getParent())) {
    auto tilePerLane = nvmmaLayout.getTilePerLaneForChild(getBitWidth());
    auto lanePerWarp = nvmmaLayout.getLanePerWarpForChild();
    auto warpPerCTA = nvmmaLayout.getWarpPerCTARef();
    auto replication = nvmmaLayout.getReplicationForChild(getOperandIndex(),
                                                          getBitWidth(), shape);
    return RegistersLayoutAttr::get(getContext(), tilePerLane, lanePerWarp,
                                    warpPerCTA, replication);
  }
  if (auto regsLayout = llvm::dyn_cast<RegistersLayoutAttr>(getParent())) {
    auto *context = getContext();
    auto elemIdExpr = getAffineDimExpr(0, context);
    auto laneIdExpr = getAffineDimExpr(1, context);
    auto warpIdExpr = getAffineDimExpr(2, context);
    auto parentExprs = regsLayout.getMap().getResults();
    auto rank = getRank();
    SmallVector<AffineExpr, 4> indexExprs(rank);
    // For each element in result, we need a whole row/column of lhs/rhs.
    if (getOperandIndex() == 0) {
      // Number of elements of a whole row is number of columns.
      auto numCols = shape[rank - 1];
      auto inputMap = AffineMap::get(
          3, 0, {elemIdExpr.floorDiv(numCols), laneIdExpr, warpIdExpr},
          context);
      if (rank == 3)
        indexExprs[0] = parentExprs[0].compose(inputMap);
      indexExprs[rank - 2] = parentExprs[rank - 2].compose(inputMap);
      indexExprs[rank - 1] = elemIdExpr % numCols;
    } else {
      // Number of elements of a whole column is number of rows.
      auto numRows = shape[rank - 2];
      auto inputMap = AffineMap::get(
          3, 0, {elemIdExpr.floorDiv(numRows), laneIdExpr, warpIdExpr},
          context);
      if (rank == 3)
        indexExprs[0] = parentExprs[0].compose(inputMap);
      indexExprs[rank - 2] = elemIdExpr % numRows;
      indexExprs[rank - 1] = parentExprs[rank - 1].compose(inputMap);
    }
    return RegistersLayoutAttr::get(context,
                                    AffineMap::get(3, 0, indexExprs, context));
  }
  llvm_unreachable("invalid parent layout");
}

LogicalResult ChangeOp::canonicalize(ChangeOp op, PatternRewriter &rewriter) {
  if (op.getType() == op.getOperand().getType()) {
    op.replaceAllUsesWith(op.getOperand());
    return success();
  }
  // Do not rewrite changes from nvidia mma layout to dot op load layout.
  if (hasLayout<NvidiaMmaLayoutAttr>(op.getOperand().getType()) &&
      hasLayout<DotOpLoadLayoutAttr>(op.getType()))
    return failure();
  auto *defOp = op.getOperand().getDefiningOp();
  if (!defOp)
    return failure();
  if (auto localLoadOp = dyn_cast<LocalLoadOp>(defOp)) {
    rewriter.replaceOpWithNewOp<LocalLoadOp>(op, op.getType(),
                                             localLoadOp.getSource());
    return success();
  }
  if (auto changeOp = dyn_cast<ChangeOp>(defOp)) {
    rewriter.replaceOpWithNewOp<ChangeOp>(op, op.getType(),
                                          changeOp.getOperand());
    return success();
  }
  if (auto splatOp = dyn_cast<SplatOp>(defOp)) {
    rewriter.replaceOpWithNewOp<SplatOp>(op, op.getType(),
                                         splatOp.getOperand());
    return success();
  }
  if (auto arangeOp = dyn_cast<ArangeOp>(defOp)) {
    rewriter.replaceOpWithNewOp<ArangeOp>(op, op.getType(), arangeOp.getStart(),
                                          arangeOp.getEnd());
    return success();
  }
  if (auto constantOp = dyn_cast<arith::ConstantOp>(defOp)) {
    if (auto splatAttr = dyn_cast<SplatElementsAttr>(constantOp.getValue())) {
      rewriter.replaceOpWithNewOp<arith::ConstantOp>(
          op, SplatElementsAttr::get(op.getType(),
                                     splatAttr.getSplatValue<Attribute>()));
      return success();
    }
  }
  return failure();
}

void AsyncCopyOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>>
        &effects) {
  effects.emplace_back(MemoryEffects::Read::get(), getSource(),
                       GlobalMemory::get());
  effects.emplace_back(MemoryEffects::Write::get(), getTarget(),
                       SharedMemory::get());
}

LogicalResult AsyncCopyOp::verify() {
  auto sourceType = cast<KapyMemRefType>(getSource().getType());
  if (!hasLayout<GlobalMemLayoutAttr>(sourceType))
    return emitOpError("source must have global memory layout");
  auto targetType = cast<KapyMemRefType>(getTarget().getType());
  if (!hasLayout<SharedMemLayoutAttr>(targetType))
    return emitOpError("target must have shared memory layout");
  auto maskType = getMask().getType();
  if (!hasLayout<RegistersLayoutAttr>(maskType))
    return emitOpError("mask must have registers layout");
  if (sourceType.getShape() != targetType.getShape())
    return emitOpError("source and target must have same shape");
  if (sourceType.getShape() != maskType.getShape())
    return emitOpError("source and mask must have same shape");
  if (sourceType.getElementType() != targetType.getElementType())
    return emitOpError("source and target must have same element type");
  return success();
}

void LocalAllocOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>>
        &effects) {
  // If the allocation has operand, mark it as no side-effect allow passes like
  // cse, dce to work in early compiler passes.
  // After the memory offset is computed, we attach the true side-effect.
  if (getOperand() && !getOperation()->hasAttr("allocation.offset"))
    return;
  effects.emplace_back(MemoryEffects::Allocate::get(), SharedMemory::get());
}

LogicalResult LocalAllocOp::verify() {
  auto resultType = getType();
  if (!hasLayout<SharedMemLayoutAttr>(resultType))
    return emitOpError("result must have shared memory layout");
  auto operand = getOperand();
  if (!operand)
    return success();
  auto operandType = operand.getType();
  if (!hasLayout<RegistersLayoutAttr>(operandType))
    return emitOpError("operand must have registers layout");
  if (operandType.getShape() != resultType.getShape())
    return emitOpError("operand and result must have same shape");
  if (operandType.getElementType() != resultType.getElementType())
    return emitOpError("operand and result must have same element type");
  return success();
}

LogicalResult LocalFreeOp::verify() {
  auto operandType = cast<KapyMemRefType>(getOperand().getType());
  if (!hasLayout<SharedMemLayoutAttr>(operandType))
    return emitOpError("operand must have shared memory layout");
  return success();
}

void LocalLoadOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>>
        &effects) {
  effects.emplace_back(MemoryEffects::Read::get(), getSource(),
                       SharedMemory::get());
}

LogicalResult LocalLoadOp::verify() {
  auto sourceType = cast<KapyMemRefType>(getSource().getType());
  if (!hasLayout<SharedMemLayoutAttr>(sourceType))
    return emitOpError("source must have shared memory layout");
  auto resultType = getType();
  if (!hasLayout<RegistersLayoutAttr>(resultType))
    return emitOpError("result must have registers layout");
  if (sourceType.getShape() != resultType.getShape())
    return emitOpError("source and result must have same shape");
  if (sourceType.getElementType() != resultType.getElementType())
    return emitOpError("source and result must have same element type");
  return success();
}
