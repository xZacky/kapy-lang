//===- Kgpu.cpp -------------------------------------------------*- C++ -*-===//
//
// This file implements the classes and functions in the KgpuDialect.
//
//===----------------------------------------------------------------------===//

#include "kapy/Dialect/Kgpu/IR/Kgpu.h"
#include "kapy/Support/CommonUtils.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/IR/OpImplementation.h"
#include "llvm/ADT/TypeSwitch.h"

using namespace mlir;
using namespace mlir::kapy;

#include "kapy/Dialect/Kgpu/IR/Dialect.cpp.inc"

#define GET_ATTRDEF_CLASSES
#include "kapy/Dialect/Kgpu/IR/Attrs.cpp.inc"

#define GET_OP_CLASSES
#include "kapy/Dialect/Kgpu/IR/Ops.cpp.inc"

namespace {

class KgpuLayoutInterface : public KapyLayoutInterface {
public:
  using KapyLayoutInterface::KapyLayoutInterface;

  virtual LogicalResult verifyLdMatrixOpLayouts(LdMatrixOp op) const override {
    auto loaderType = op.getLoader().getType();
    auto resultType = op.getType();
    auto loaderLayout = getLayout<FragmentsLayoutAttr>(loaderType);
    auto resultLayout = getLayout<FragmentsLayoutAttr>(resultType);
    auto bitWidth = getIntOrFloatBitWidth(resultType);
    auto vecWidth = 128 / bitWidth;
    auto packWidth = 32 / bitWidth;
    if (loaderLayout.isColMajor()) {
      if (loaderLayout.getLaneArray() != SmallVector<int64_t, 2>{16, 2})
        return op->emitOpError("loader has incompatible layout");
      if (loaderLayout.getLaneLoops() != SmallVector<int64_t, 2>{1, vecWidth})
        return op->emitOpError("loader has incompatible layout");
    } else {
      if (loaderLayout.getLaneArray() != SmallVector<int64_t, 2>{2, 16})
        return op->emitOpError("loader has incompatible layout");
      if (loaderLayout.getLaneLoops() != SmallVector<int64_t, 2>{vecWidth, 1})
        return op->emitOpError("loader has incompatible layout");
    }
    if (resultLayout.isRowMajor()) {
      if (resultLayout.getLaneArray() != SmallVector<int64_t, 2>{8, 4})
        return op->emitOpError("result has incompatible layout");
      if (resultLayout.getLaneLoops() != SmallVector<int64_t, 2>{1, packWidth})
        return op->emitOpError("result has incompatible layout");
    } else {
      if (resultLayout.getLaneArray() != SmallVector<int64_t, 2>{4, 8})
        return op->emitOpError("result has incompatible layout");
      if (resultLayout.getLaneLoops() != SmallVector<int64_t, 2>{packWidth, 1})
        return op->emitOpError("result has incompatible layout");
    }
    return success();
  }

  virtual Attribute
  inferTransposeOpLayout(Attribute sourceLayout) const override {
    return cast<FragmentsLayoutAttr>(sourceLayout).transpose();
  }

  virtual LogicalResult verifyMatmulOpLayouts(MatmulOp op) const override {
    auto lhsType = op.getLhs().getType();
    auto rhsType = op.getRhs().getType();
    auto accType = op.getAcc().getType();
    auto implWay = op.getMatmulImplWay();
    if (failed(verifyMatmulOpAccLayout(accType, implWay)))
      return op->emitOpError("acc has incompatible layout");
    if (failed(verifyMatmulOpLhsLayout(lhsType, implWay)))
      return op->emitOpError("lhs has incompatible layout");
    if (failed(verifyMatmulOpRhsLayout(rhsType, implWay)))
      return op->emitOpError("rhs has incompatible layout");
    return success();
  }

private:
  static LogicalResult verifyMatmulOpAccLayout(RankedTensorType accType,
                                               MatmulImplWay implWay) {
    auto accLayout = getLayout<FragmentsLayoutAttr>(accType);
    if (accLayout.isColMajor())
      return failure();
    if (accLayout.getLaneArray() != SmallVector<int64_t, 2>{8, 4})
      return failure();
    if (accLayout.getLaneLoops() != SmallVector<int64_t, 2>{1, 2})
      return failure();
    if (accLayout.getWarpLoops(accType.getShape())[0] < 2)
      return failure();
    return success();
  }

  static LogicalResult verifyMatmulOpLhsLayout(RankedTensorType lhsType,
                                               MatmulImplWay implWay) {
    auto lhsLayout = getLayout<FragmentsLayoutAttr>(lhsType);
    if (lhsLayout.isColMajor())
      return failure();
    if (lhsLayout.getLaneArray() != SmallVector<int64_t, 2>{8, 4})
      return failure();
    if (implWay == MatmulImplWay::MMA_M16N8K8_F16 ||
        implWay == MatmulImplWay::MMA_M16N8K16_F16) {
      if (lhsLayout.getLaneLoops() != SmallVector<int64_t, 2>{1, 2})
        return failure();
      return success();
    }
    if (implWay == MatmulImplWay::MMA_M16N8K8_TF32) {
      if (lhsLayout.getLaneLoops() != SmallVector<int64_t, 2>{1, 1})
        return failure();
      return success();
    }
    if (implWay == MatmulImplWay::MMA_M16N8K16_F8) {
      if (lhsLayout.getLaneLoops() != SmallVector<int64_t, 2>{1, 4})
        return failure();
      return success();
    }
    llvm_unreachable("unsupported matmul implement way");
  }

  static LogicalResult verifyMatmulOpRhsLayout(RankedTensorType rhsType,
                                               MatmulImplWay implWay) {
    auto rhsLayout = getLayout<FragmentsLayoutAttr>(rhsType);
    if (rhsLayout.isRowMajor())
      return failure();
    if (rhsLayout.getLaneArray() != SmallVector<int64_t, 2>{4, 8})
      return failure();
    if (implWay == MatmulImplWay::MMA_M16N8K8_F16 ||
        implWay == MatmulImplWay::MMA_M16N8K16_F16) {
      if (rhsLayout.getLaneLoops() != SmallVector<int64_t, 2>{2, 1})
        return failure();
      return success();
    }
    if (implWay == MatmulImplWay::MMA_M16N8K8_TF32) {
      if (rhsLayout.getLaneLoops() != SmallVector<int64_t, 2>{1, 1})
        return failure();
      return success();
    }
    if (implWay == MatmulImplWay::MMA_M16N8K16_F8) {
      if (rhsLayout.getLaneLoops() != SmallVector<int64_t, 2>{4, 1})
        return failure();
      return success();
    }
    llvm_unreachable("unsupported matmul implement way");
  }
};

} // namespace

void KgpuDialect::initialize() {
  addAttributes<
#define GET_ATTRDEF_LIST
#include "kapy/Dialect/Kgpu/IR/Attrs.cpp.inc"
      >();
  addOperations<
#define GET_OP_LIST
#include "kapy/Dialect/Kgpu/IR/Ops.cpp.inc"
      >();
  addInterfaces<KgpuLayoutInterface>();
}

SmallVector<int64_t, 2>
FragmentsLayoutAttr::getWarpLoops(ArrayRef<int64_t> shape) const {
  auto laneArray = getLaneArray();
  auto laneLoops = getLaneLoops();
  SmallVector<int64_t, 2> warpLoops(2);
  for (unsigned i = 0; i < 2; ++i) {
    warpLoops[i] = shape[i] / (laneArray[i] * laneLoops[i]);
    warpLoops[i] = std::max<int64_t>(warpLoops[i], 1);
  }
  return warpLoops;
}

SmallVector<int64_t, 2>
FragmentsLayoutAttr::getLoopSpace(ArrayRef<int64_t> shape) const {
  auto warpLoops = getWarpLoops(shape);
  auto laneLoops = getLaneLoops();
  SmallVector<int64_t, 2> loopSpace(2);
  for (unsigned i = 0; i < 2; ++i)
    loopSpace[i] = warpLoops[i] * laneLoops[i];
  return loopSpace;
}

AffineMap FragmentsLayoutAttr::getAffineMap(ArrayRef<int64_t> shape,
                                            MapOption option) const {
  auto laneArray = getLaneArray();
  auto laneLoops = getLaneLoops();
  auto loopSpace = getLoopSpace(shape);
  auto thisShape = llvm::to_vector<2>(shape);

  if (option == MapOption::FROM_VALUES) {
    auto indexX = getAffineDimExpr(0, getContext());
    auto indexY = getAffineDimExpr(1, getContext());
    std::array<int64_t, 3> idSplitX{/*any*/ 1, laneArray[0], laneLoops[0]};
    std::array<int64_t, 3> idSplitY{/*any*/ 1, laneArray[1], laneLoops[1]};
    auto idCompsX = delinearize(indexX, idSplitX);
    auto idCompsY = delinearize(indexY, idSplitY);

    auto laneIdX = idCompsX[1];
    auto laneIdY = idCompsY[1];
    AffineExpr laneId;
    if (isRowMajor())
      laneId = laneIdX * laneArray[1] + laneIdY;
    else
      laneId = laneIdX + laneIdY * laneArray[0];

    auto loopIvX = idCompsX[0] * laneLoops[0] + idCompsX[2];
    auto loopIvY = idCompsY[0] * laneLoops[1] + idCompsY[2];
    AffineExpr loopIv;
    if (isRowMajor())
      loopIv = loopIvX * loopSpace[1] + loopIvY;
    else
      loopIv = loopIvX + loopIvY * loopSpace[0];

    return AffineMap::get(2, 0, {laneId, loopIv}, getContext());
  }

  if (isColMajor()) {
    laneArray = kapy::transpose(laneArray);
    laneLoops = kapy::transpose(laneLoops);
    loopSpace = kapy::transpose(loopSpace);
    thisShape = kapy::transpose(thisShape);
  }
  auto laneId = getAffineDimExpr(0, getContext());
  auto loopIv = getAffineDimExpr(1, getContext());
  auto laneIds = delinearize(laneId, laneArray);
  auto loopIvs = delinearize(loopIv, loopSpace);

  SmallVector<AffineExpr, 2> indices(2);
  for (unsigned i = 0; i < 2; ++i) {
    std::array<int64_t, 2> loopSplit{/*any*/ 1, laneLoops[i]};
    auto ivComps = delinearize(loopIvs[i], loopSplit);

    std::array<int64_t, 2> coeffs;
    coeffs[0] = laneArray[i] * laneLoops[i];
    coeffs[1] = laneLoops[i];

    indices[i] = ivComps[0] * coeffs[0] + laneIds[i] * coeffs[1] + ivComps[1];

    if (option == MapOption::TO_TENSOR)
      if (laneArray[i] * laneLoops[i] > thisShape[i])
        indices[i] = indices[i] % thisShape[i];
  }
  if (isColMajor())
    indices = kapy::transpose(indices);

  return AffineMap::get(2, 0, indices, getContext());
}

FragmentsLayoutAttr FragmentsLayoutAttr::transpose() const {
  return FragmentsLayoutAttr::get(getContext(), //
                                  kapy::transpose(getLaneArray()),
                                  kapy::transpose(getLaneLoops()),
                                  getMajorAxis(), getMinorAxis());
}

FragmentsLayoutAttr FragmentsLayoutAttr::exchangeAxes() const {
  return FragmentsLayoutAttr::get(getContext(), getLaneArray(), getLaneLoops(),
                                  getMajorAxis(), getMinorAxis());
}

LogicalResult ChangeOp::canonicalize(ChangeOp op, PatternRewriter &rewriter) {
  auto source = op.getSource();
  auto resultType = op.getType();
  if (resultType == source.getType()) {
    op.replaceAllUsesWith(source);
    return success();
  }
  auto *defOp = source.getDefiningOp();
  if (!defOp)
    return failure();
  if (auto inOp = dyn_cast<ChangeOp>(defOp)) {
    rewriter.replaceOpWithNewOp<ChangeOp>(op, resultType, inOp.getSource());
    return success();
  }
  if (auto inOp = dyn_cast<SplatOp>(defOp)) {
    rewriter.replaceOpWithNewOp<SplatOp>(op, resultType, inOp.getSource());
    return success();
  }
  if (auto inOp = dyn_cast<arith::ConstantOp>(defOp)) {
    auto splatAttr = cast<SplatElementsAttr>(inOp.getValue());
    rewriter.replaceOpWithNewOp<arith::ConstantOp>(
        op, SplatElementsAttr::get(resultType,
                                   splatAttr.getSplatValue<Attribute>()));
    return success();
  }
  return failure();
}

int64_t kapy::getSharedMemorySize(ModuleOp module) {
  if (!module->hasAttr(sizeAttrName))
    llvm_unreachable("can not get named attribute");
  return cast<IntegerAttr>(module->getAttr(sizeAttrName)).getInt();
}

int64_t kapy::getSharedMemoryOffset(MkSharedOp op) {
  if (!op->hasAttr(offsetAttrName))
    llvm_unreachable("can not get named attribute");
  return cast<IntegerAttr>(op->getAttr(offsetAttrName)).getInt();
}

static unsigned getIdWidth(int64_t id) {
  assert(id < 10000);
  if (id >= 1000)
    return 4;
  if (id >= 100)
    return 3;
  if (id >= 10)
    return 2;
  return 1;
}

static std::string getElementString(std::array<int64_t, 2> idPair,
                                    int64_t maxLaneId, int64_t maxLoopIv) {
  auto maxLaneIdWidth = getIdWidth(maxLaneId);
  auto maxLoopIvWidth = getIdWidth(maxLoopIv);
  auto maxIdPairWidth = maxLaneIdWidth + maxLoopIvWidth + 2;
  std::string string = "";
  auto curLaneId = idPair[0];
  auto curLaneIdWidth = getIdWidth(curLaneId);
  for (unsigned i = 0; i < maxLaneIdWidth - curLaneIdWidth; ++i)
    string += ' ';
  string += 'T' + std::to_string(curLaneId) + ':';
  auto curLoopIv = idPair[1];
  auto curLoopIvWidth = getIdWidth(curLoopIv);
  for (unsigned i = 0; i < maxLoopIvWidth - curLoopIvWidth; ++i)
    string += ' ';
  string += std::to_string(curLoopIv);
  return string;
}

std::string kapy::getLayoutString(RankedTensorType tensorType) {
  auto numElems = tensorType.getNumElements();
  std::vector<std::vector<std::array<int64_t, 2>>> indexToIdPairs(numElems);
  auto layout = getLayout<FragmentsLayoutAttr>(tensorType);
  auto shape = tensorType.getShape();
  auto option = FragmentsLayoutAttr::MapOption::TO_TENSOR;
  auto map = layout.getAffineMap(shape, option);
  auto loopSize = product(layout.getLoopSpace(shape));
  for (int64_t laneId = 0; laneId < warpSize; ++laneId) {
    for (int64_t loopIv = 0; loopIv < loopSize; ++loopIv) {
      auto index = linearize(map.compose({laneId, loopIv}), shape);
      indexToIdPairs[index].push_back({laneId, loopIv});
    }
  }
  std::string string = "";
  auto numRounds = indexToIdPairs[0].size();
  for (unsigned i = 0; i < numRounds; ++i) {
    if (i > 0)
      string += "\n\n";
    for (unsigned index = 0; index < numElems; ++index) {
      if (index % shape[1] == 0)
        string += index == 0 ? "[[" : "\n [";
      auto idPair = indexToIdPairs[index][i];
      string += getElementString(idPair, warpSize - 1, loopSize - 1);
      if ((index + 1) % shape[1] == 0)
        string += (index + 1) == numElems ? "]]" : "],";
      else
        string += ", ";
    }
  }
  return string;
}
