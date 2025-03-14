//===- Kgpu.cpp -------------------------------------------------*- C++ -*-===//
//
// This file implements the classes and functions in the kgpu dialect.
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
    auto simdSize = 128 / bitWidth;
    auto wordSize = 32 / bitWidth;
    if (loaderLayout.isColMajor()) {
      if (loaderLayout.getLaneArray() != SmallVector<int64_t, 2>{16, 2})
        return op->emitOpError("loader has incompatible layout");
      if (loaderLayout.getLaneLoops() != SmallVector<int64_t, 2>{1, simdSize})
        return op->emitOpError("loader has incompatible layout");
    } else {
      if (loaderLayout.getLaneArray() != SmallVector<int64_t, 2>{2, 16})
        return op->emitOpError("loader has incompatible layout");
      if (loaderLayout.getLaneLoops() != SmallVector<int64_t, 2>{simdSize, 1})
        return op->emitOpError("loader has incompatible layout");
    }
    if (resultLayout.isRowMajor()) {
      if (resultLayout.getLaneArray() != SmallVector<int64_t, 2>{8, 4})
        return op->emitOpError("result has incompatible layout");
      if (resultLayout.getLaneLoops() != SmallVector<int64_t, 2>{1, wordSize})
        return op->emitOpError("result has incompatible layout");
    } else {
      if (resultLayout.getLaneArray() != SmallVector<int64_t, 2>{4, 8})
        return op->emitOpError("result has incompatible layout");
      if (resultLayout.getLaneLoops() != SmallVector<int64_t, 2>{wordSize, 1})
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
                                            unsigned option) const {
  assert(option == 1 || option == 2 || option == 3);

  auto laneArray = getLaneArray();
  auto laneLoops = getLaneLoops();
  auto loopSpace = getLoopSpace(shape);
  auto thisShape = llvm::to_vector<2>(shape);

  if (option == 3) {
    auto index0 = getAffineDimExpr(0, getContext());
    auto index1 = getAffineDimExpr(1, getContext());
    std::array<int64_t, 3> idSplit0{/*any*/ 1, laneArray[0], laneLoops[0]};
    std::array<int64_t, 3> idSplit1{/*any*/ 1, laneArray[1], laneLoops[1]};
    auto idComps0 = delinearize(index0, idSplit0);
    auto idComps1 = delinearize(index1, idSplit1);

    auto laneId0 = idComps0[1];
    auto laneId1 = idComps1[1];
    AffineExpr laneId;
    if (isRowMajor())
      laneId = laneId0 * laneArray[1] + laneId1;
    else
      laneId = laneId0 + laneId1 * laneArray[0];

    auto loopIv0 = idComps0[0] * laneLoops[0] + idComps0[2];
    auto loopIv1 = idComps1[0] * laneLoops[1] + idComps1[2];
    AffineExpr loopIv;
    if (isRowMajor())
      loopIv = loopIv0 * loopSpace[1] + loopIv1;
    else
      loopIv = loopIv0 + loopIv1 * loopSpace[0];

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

    if (option == 1)
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
  auto map = layout.getAffineMap(shape, 1);
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
