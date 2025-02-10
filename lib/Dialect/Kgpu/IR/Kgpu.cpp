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
    if (isa<SwizzlingLayoutAttr>(attr)) {
      os << "swizzling";
      return AliasResult::FinalAlias;
    }
    if (isa<FragmentsLayoutAttr>(attr)) {
      os << "fragments";
      return AliasResult::FinalAlias;
    }
    return OpAsmDialectInterface::getAlias(attr, os);
  }
};

class KgpuLayoutInterface : public KapyLayoutInterface {
public:
  using KapyLayoutInterface::KapyLayoutInterface;

  virtual Attribute
  inferTransposeOpLayout(Attribute sourceLayout) const override {
    return cast<FragmentsLayoutAttr>(sourceLayout).transpose();
  }

  virtual LogicalResult
  verifyMatmulOpLayouts(MatmulOp matmulOp) const override {
    auto lhsType = matmulOp.getLhs().getType();
    auto rhsType = matmulOp.getRhs().getType();
    auto accType = matmulOp.getAcc().getType();
    auto lhsLayout = cast<FragmentsLayoutAttr>(lhsType.getEncoding());
    auto rhsLayout = cast<FragmentsLayoutAttr>(rhsType.getEncoding());
    auto accLayout = cast<FragmentsLayoutAttr>(accType.getEncoding());

    auto implWay = matmulOp.getMatmulImplWay();
    if (failed(verifyMatmulAccLayout(accLayout, implWay)))
      return matmulOp->emitOpError("acc has incompatible layout");
    if (failed(verifyMatmulLhsLayout(lhsLayout, accLayout, implWay,
                                     lhsType.getShape())))
      return matmulOp->emitOpError("lhs has incompatible layout");
    if (failed(verifyMatmulRhsLayout(rhsLayout, accLayout, implWay,
                                     accType.getShape())))
      return matmulOp->emitOpError("rhs has incompatible layout");
    return success();
  }

private:
  static LogicalResult verifyMatmulAccLayout(FragmentsLayoutAttr accLayout,
                                             MatmulImplWay implWay) {
    if (implWay == MatmulImplWay::FMA) {
      if (accLayout.getLaneLoops() != SmallVector<int64_t, 2>{2, 2} &&
          accLayout.getLaneLoops() != SmallVector<int64_t, 2>{4, 4})
        return failure();
      return success();
    }
    if (!accLayout.isRowMajor())
      return failure();
    if (accLayout.getLaneArray() != SmallVector<int64_t, 2>{8, 4})
      return failure();
    if (accLayout.getLaneLoops() != SmallVector<int64_t, 2>{1, 2})
      return failure();
    if (implWay == MatmulImplWay::MMA_M16N8K8_F16 ||
        implWay == MatmulImplWay::MMA_M16N8K8_TF32) {
      if (accLayout.getWarpLoops()[0] < 2)
        return failure();
      return success();
    }
    if (implWay == MatmulImplWay::MMA_M16N8K16_F8 ||
        implWay == MatmulImplWay::MMA_M16N8K16_F16) {
      if (accLayout.getWarpLoops()[0] < 2 || accLayout.getWarpLoops()[1] < 2)
        return failure();
      return success();
    }
    llvm_unreachable("unsupported MatmulImplWay");
  }

  static LogicalResult verifyMatmulLhsLayout(FragmentsLayoutAttr lhsLayout,
                                             FragmentsLayoutAttr accLayout,
                                             MatmulImplWay implWay,
                                             ArrayRef<int64_t> lhsShape) {
    if (lhsLayout.getWarpArray() != accLayout.getWarpArray())
      return failure();
    if (implWay == MatmulImplWay::FMA) {
      if (lhsLayout.getWarpLoops() != accLayout.getWarpLoops())
        return failure();
      if (lhsLayout.getLaneArray() != accLayout.getLaneArray())
        return failure();
      if (lhsLayout.getLaneLoops()[0] != accLayout.getLaneLoops()[0])
        return failure();
      if (lhsLayout.getLaneLoops()[1] != lhsShape[1])
        return failure();
      return success();
    }
    if (!lhsLayout.isRowMajor())
      return failure();
    if (lhsLayout.getWarpLoops()[0] != accLayout.getWarpLoops()[0])
      return failure();
    if (lhsLayout.getLaneArray() != SmallVector<int64_t, 2>{8, 4})
      return failure();
    if (implWay == MatmulImplWay::MMA_M16N8K8_F16 ||
        implWay == MatmulImplWay::MMA_M16N8K16_F16) {
      if (lhsLayout.getWarpLoops()[1] != lhsShape[1] / 8)
        return failure();
      if (lhsLayout.getLaneLoops() != SmallVector<int64_t, 2>{1, 2})
        return failure();
      return success();
    }
    if (implWay == MatmulImplWay::MMA_M16N8K8_TF32) {
      if (lhsLayout.getWarpLoops()[1] != lhsShape[1] / 4)
        return failure();
      if (lhsLayout.getLaneLoops() != SmallVector<int64_t, 2>{1, 1})
        return failure();
      return success();
    }
    if (implWay == MatmulImplWay::MMA_M16N8K16_F8) {
      if (lhsLayout.getWarpLoops()[1] != lhsShape[1] / 16)
        return failure();
      if (lhsLayout.getLaneLoops() != SmallVector<int64_t, 2>{1, 4})
        return failure();
      return success();
    }
    llvm_unreachable("unsupported MatmulImplWay");
  }

  static LogicalResult verifyMatmulRhsLayout(FragmentsLayoutAttr rhsLayout,
                                             FragmentsLayoutAttr accLayout,
                                             MatmulImplWay implWay,
                                             ArrayRef<int64_t> rhsShape) {
    if (rhsLayout.getWarpArray() != accLayout.getWarpArray())
      return failure();
    if (implWay == MatmulImplWay::FMA) {
      if (rhsLayout.getWarpLoops() != accLayout.getWarpLoops())
        return failure();
      if (rhsLayout.getLaneArray() != accLayout.getLaneArray())
        return failure();
      if (rhsLayout.getLaneLoops()[0] != rhsShape[0])
        return failure();
      if (rhsLayout.getLaneLoops()[1] != accLayout.getLaneLoops()[1])
        return failure();
    }
    if (rhsLayout.isRowMajor())
      return failure();
    if (rhsLayout.getWarpLoops()[1] != accLayout.getWarpLoops()[1])
      return failure();
    if (rhsLayout.getLaneArray() != SmallVector<int64_t, 2>{4, 8})
      return failure();
    if (implWay == MatmulImplWay::MMA_M16N8K8_F16 ||
        implWay == MatmulImplWay::MMA_M16N8K16_F16) {
      if (rhsLayout.getWarpLoops()[0] != rhsShape[0] / 8)
        return failure();
      if (rhsLayout.getLaneLoops() != SmallVector<int64_t, 2>{2, 1})
        return failure();
      return success();
    }
    if (implWay == MatmulImplWay::MMA_M16N8K8_TF32) {
      if (rhsLayout.getWarpLoops()[0] != rhsShape[1] / 4)
        return failure();
      if (rhsLayout.getLaneLoops() != SmallVector<int64_t, 2>{1, 1})
        return failure();
      return success();
    }
    if (implWay == MatmulImplWay::MMA_M16N8K16_F8) {
      if (rhsLayout.getWarpLoops()[0] != rhsShape[0] / 16)
        return failure();
      if (rhsLayout.getLaneLoops() != SmallVector<int64_t, 2>{4, 1})
        return failure();
      return success();
    }
    llvm_unreachable("unsupported MatmulImplWay");
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

Type SharedMemRefType::parse(AsmParser &parser) {
  if (failed(parser.parseLess()))
    return Type();
  SmallVector<int64_t, 2> shape;
  if (failed(parser.parseDimensionList(shape, false)))
    return Type();
  Type elementType;
  if (failed(parser.parseType(elementType)))
    return Type();
  Attribute layout;
  if (succeeded(parser.parseOptionalComma()))
    if (failed(parser.parseAttribute(layout)))
      return Type();
  if (failed(parser.parseGreater()))
    return Type();
  return SharedMemRefType::get(shape, elementType, layout);
}

void SharedMemRefType::print(AsmPrinter &printer) const {
  printer << "<";
  for (auto size : getShape())
    printer << size << "x";
  printer << getElementType();
  if (getEncoding())
    printer << ", " << getEncoding();
  printer << ">";
}

SmallVector<int64_t, 2>
FragmentsLayoutAttr::getLoopSpace(ArrayRef<int64_t> shape) const {
  auto warpArray = getWarpArray();
  auto warpLoops = getWarpLoops();
  auto laneArray = getLaneArray();
  auto laneLoops = getLaneLoops();
  SmallVector<int64_t, 2> loopSpace(2);
  for (unsigned i = 0; i < 2; ++i) {
    auto numClones =
        shape[i] / (warpArray[i] * warpLoops[i] * laneArray[i] * laneLoops[i]);
    numClones = std::max<int64_t>(numClones, 1);
    loopSpace[i] = numClones * warpLoops[i] * laneLoops[i];
  }
  return loopSpace;
}

AffineMap FragmentsLayoutAttr::getAffineMap(ArrayRef<int64_t> shape,
                                            bool unique) const {
  auto Shape = llvm::to_vector<2>(shape);
  auto warpArray = getWarpArray();
  auto warpLoops = getWarpLoops();
  auto laneArray = getLaneArray();
  auto laneLoops = getLaneLoops();
  auto loopSpace = getLoopSpace(shape);

  if (!isRowMajor()) {
    Shape = kapy::transpose(Shape);
    warpArray = kapy::transpose(warpArray);
    warpLoops = kapy::transpose(warpLoops);
    laneArray = kapy::transpose(laneArray);
    laneLoops = kapy::transpose(laneLoops);
    loopSpace = kapy::transpose(loopSpace);
  }

  auto warpId1d = getAffineDimExpr(0, getContext());
  auto laneId1d = getAffineDimExpr(1, getContext());
  auto loopIv1d = getAffineDimExpr(2, getContext());

  auto warpId2d = delinearize(warpId1d, warpArray);
  auto laneId2d = delinearize(laneId1d, laneArray);
  auto loopIv2d = delinearize(loopIv1d, loopSpace);

  SmallVector<AffineExpr, 2> result2d(2);
  for (unsigned i = 0; i < 2; ++i) {
    std::array<int64_t, 3> loopSplit{/*any*/ 1, warpLoops[i], laneLoops[i]};
    auto loopIv3d = delinearize<3>(loopIv2d[i], loopSplit);

    std::array<int64_t, 4> coeffs;
    coeffs[0] = warpArray[i] * warpLoops[i] * laneArray[i] * laneLoops[i];
    coeffs[1] = warpLoops[i] * laneArray[i] * laneLoops[i];
    coeffs[2] = laneArray[i] * laneLoops[i];
    coeffs[3] = laneLoops[i];

    result2d[i] = loopIv3d[0] * coeffs[0] + warpId2d[i] * coeffs[1] +
                  loopIv3d[1] * coeffs[2] + laneId2d[i] * coeffs[3] +
                  loopIv3d[2];

    if (!unique &&
        warpArray[i] * warpLoops[i] * laneArray[i] * laneLoops[i] > Shape[i])
      result2d[i] = result2d[i] % Shape[i];
  }
  if (!isRowMajor())
    result2d = kapy::transpose(result2d);
  return AffineMap::get(3, 0, result2d, getContext());
}

FragmentsLayoutAttr FragmentsLayoutAttr::transpose() const {
  return FragmentsLayoutAttr::get(
      getContext(), kapy::transpose(getWarpArray()),
      kapy::transpose(getWarpLoops()), kapy::transpose(getLaneArray()),
      kapy::transpose(getLaneLoops()), !isRowMajor());
}

LogicalResult ChangeOp::canonicalize(ChangeOp op, PatternRewriter &rewriter) {
  if (op.getType() == op.getSource().getType()) {
    op.replaceAllUsesWith(op.getSource());
    return success();
  }
  auto *defOp = op.getSource().getDefiningOp();
  if (!defOp)
    return failure();
  if (auto changeOp = dyn_cast<ChangeOp>(defOp)) {
    rewriter.replaceOpWithNewOp<ChangeOp>(op, op.getType(),
                                          changeOp.getSource());
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
  if (auto splatOp = dyn_cast<SplatOp>(defOp)) {
    rewriter.replaceOpWithNewOp<SplatOp>(op, op.getType(), splatOp.getSource());
    return success();
  }
  if (auto loadOp = dyn_cast<LoadSharedOp>(defOp)) {
    rewriter.replaceOpWithNewOp<LoadSharedOp>(
        op, op.getType(), loadOp.getSource(), loadOp.getOffset0(),
        loadOp.getOffset1());
    return success();
  }
  return failure();
}

void CpAsyncGlobalToSharedOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>>
        &effects) {
  effects.emplace_back(MemoryEffects::Read::get(), &getSourceMutable(),
                       GlobalMemory::get());
  effects.emplace_back(MemoryEffects::Write::get(), &getTargetMutable(),
                       SharedMemory::get());
}

void AllocSharedOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>>
        &effects) {
  // If the allocation has source, mark it as no side effect allow passes like
  // CSE, DCE to work in early compiler passes.
  // After the memory offset is computed, we attach the true side effect.
  if (getSource() && !getOperation()->hasAttr(sharedOffsetAttrName))
    return;
  effects.emplace_back(MemoryEffects::Allocate::get(), SharedMemory::get());
}

void LoadSharedOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>>
        &effects) {
  effects.emplace_back(MemoryEffects::Read::get(), &getSourceMutable(),
                       SharedMemory::get());
}

unsigned kapy::getNvidiaCC(ModuleOp module) {
  if (!module->hasAttr(nvidiaCCAttrName))
    llvm_unreachable("can not get a named attribute");
  return cast<IntegerAttr>(module->getAttr(nvidiaCCAttrName)).getInt();
}

unsigned kapy::getNumWarps(ModuleOp module) {
  if (!module->hasAttr(numWarpsAttrName))
    llvm_unreachable("can not get a named attribute");
  return cast<IntegerAttr>(module->getAttr(numWarpsAttrName)).getInt();
}

int64_t kapy::getSharedMemoryNeeded(ModuleOp module) {
  if (!module->hasAttr(sharedNeededAttrName))
    llvm_unreachable("can not get named attribute");
  return cast<IntegerAttr>(module->getAttr(sharedNeededAttrName)).getInt();
}

int64_t kapy::getSharedMemoryOffset(Operation *op) {
  if (!op->hasAttr(sharedOffsetAttrName))
    return 0;
  return cast<IntegerAttr>(op->getAttr(sharedOffsetAttrName)).getInt();
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
                                    int64_t maxThread, int64_t maxLoopIv) {
  auto maxThreadWidth = getIdWidth(maxThread);
  auto maxLoopIvWidth = getIdWidth(maxLoopIv);
  auto maxIdPairWidth = 2 + maxThreadWidth + maxLoopIvWidth;
  std::string string = "";
  auto curThread = idPair[0];
  auto curThreadWidth = getIdWidth(curThread);
  for (unsigned i = 0; i < maxThreadWidth - curThreadWidth; ++i)
    string += ' ';
  string += 'T' + std::to_string(curThread) + ':';
  auto curLoopIv = idPair[1];
  auto curLoopIvWidth = getIdWidth(curLoopIv);
  for (unsigned i = 0; i < maxLoopIvWidth - curLoopIvWidth; ++i)
    string += ' ';
  string += std::to_string(curLoopIv);
  return string;
}

std::string kapy::getTensorLayoutString(RankedTensorType tensorType) {
  auto shape = tensorType.getShape();
  auto numElements = tensorType.getNumElements();
  // Map from linearized tensor index to the id pairs.
  std::vector<std::vector<std::array<int64_t, 2>>> linearToIdPairs(numElements);
  auto layout = cast<FragmentsLayoutAttr>(tensorType.getEncoding());
  auto map = layout.getAffineMap(shape);
  auto numWarps = product(layout.getWarpArray());
  auto loopSize = product(layout.getLoopSpace(shape));
  for (int64_t warpId = 0; warpId < numWarps; ++warpId) {
    for (int64_t laneId = 0; laneId < numLanes; ++laneId) {
      for (int64_t loopIv = 0; loopIv < loopSize; ++loopIv) {
        auto coords = map.compose({warpId, laneId, loopIv});
        auto linear = linearize(coords, shape);
        linearToIdPairs[linear].push_back({warpId * numLanes + laneId, loopIv});
      }
    }
  }
  std::string string = "";
  auto maxThread = numWarps * numLanes - 1;
  auto maxLoopIv = loopSize - 1;
  auto numRounds = linearToIdPairs[0].size();
  for (unsigned i = 0; i < numRounds; ++i) {
    if (i > 0)
      string += "\n\n";
    for (int64_t linear = 0; linear < numElements; ++linear) {
      if (linear % shape[1] == 0)
        string += linear == 0 ? "[[" : "\n [";
      auto idPair = linearToIdPairs[linear][i];
      string += getElementString(idPair, maxThread, maxLoopIv);
      if ((linear + 1) % shape[1] == 0)
        string += (linear + 1) == numElements ? "]]" : "],";
      else
        string += ", ";
    }
  }
  return string;
}
