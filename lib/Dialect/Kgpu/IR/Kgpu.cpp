//===- Kgpu.cpp -------------------------------------------------*- C++ -*-===//
//
// This file implements the functions in the Kgpu dialect.
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
    if (isa<SharedMemLayoutAttr>(attr)) {
      os << "shmem";
      return AliasResult::FinalAlias;
    }
    if (isa<FragmentsLayoutAttr>(attr)) {
      os << "frags";
      return AliasResult::FinalAlias;
    }
    if (isa<AxisSliceLayoutAttr>(attr)) {
      os << "slice";
      return AliasResult::FinalAlias;
    }
    if (isa<NvidiaMmaLayoutAttr>(attr)) {
      os << "nvmma";
      return AliasResult::FinalAlias;
    }
    if (isa<MmOperandLayoutAttr>(attr)) {
      os << "mmopd";
      return AliasResult::FinalAlias;
    }
    return OpAsmDialectInterface::getAlias(attr, os);
  }
};

class KgpuLayoutInterface : public KapyLayoutInterface {
public:
  using KapyLayoutInterface::KapyLayoutInterface;

  virtual FailureOr<Attribute>
  inferReduceOpLayout(Attribute operandLayout, unsigned axis,
                      std::optional<Location> loc) const override {
    return AxisSliceLayoutAttr::get(getContext(), operandLayout, axis);
  }

  virtual FailureOr<Attribute>
  inferUnsqueezeOpLayout(Attribute operandLayout, unsigned axis,
                         std::optional<Location> loc) const override {
    auto sliceLayout = dyn_cast<AxisSliceLayoutAttr>(operandLayout);
    if (!sliceLayout)
      return emitOptionalError(loc, "operand must have slice axis layout");
    if (sliceLayout.getAxis() != axis)
      return emitOptionalError(loc, "incompatible slice axis for operand");
    return sliceLayout.getParent();
  }

  virtual FailureOr<Attribute>
  inferTransposeOpLayout(Attribute operandLayout,
                         std::optional<Location> loc) const override {
    auto fragsLayout = dyn_cast<FragmentsLayoutAttr>(operandLayout);
    if (!fragsLayout)
      return emitOptionalError(loc, "operand must have fragments layout");
    return FragmentsLayoutAttr::get(getContext(),
                                    transpose(fragsLayout.getShapeOfWarpsRef()),
                                    transpose(fragsLayout.getWarpLoopsRef()),
                                    transpose(fragsLayout.getShapeOfLanesRef()),
                                    transpose(fragsLayout.getLaneLoopsRef()),
                                    fragsLayout.getMajorAxis() == 1 ? 0 : 1);
  }

  virtual LogicalResult verifyMatmulOpLayouts(MatmulOp op) const override {
    auto lhsType = op.getLhs().getType();
    auto rhsType = op.getRhs().getType();
    auto accumType = op.getAccum().getType();

    auto lhsLayout = dyn_cast<MmOperandLayoutAttr>(lhsType.getEncoding());
    if (!lhsLayout)
      return op->emitOpError("lhs must have matmul operand layout");
    if (lhsLayout.getOperandIndex() != 0)
      return op->emitOpError("lhs layout with wrong operand index");

    auto rhsLayout = dyn_cast<MmOperandLayoutAttr>(rhsType.getEncoding());
    if (!rhsLayout)
      return op->emitOpError("rhs must have matmul operand layout");
    if (rhsLayout.getOperandIndex() != 1)
      return op->emitOpError("rhs layout with wrong operand index");

    auto accumLayout = accumType.getEncoding();
    if (isa<FragmentsLayoutAttr, NvidiaMmaLayoutAttr>(accumLayout)) {
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

AffineMap SharedMemLayoutAttr::getMemRefMap() const {
  auto resultExpr = getAffineConstantExpr(0, getContext());
  unsigned numDims = 0;
  for (auto stride : getStrides()) {
    auto dimExpr = getAffineDimExpr(numDims++, getContext());
    auto intExpr = getAffineConstantExpr(stride, getContext());
    resultExpr = resultExpr + dimExpr * intExpr;
  }
  return AffineMap::get(numDims, 0, resultExpr);
}

FragmentsLayoutAttr FragmentsLayoutAttr::get(MLIRContext *context,
                                             ArrayRef<int64_t> shapeOfWarps,
                                             ArrayRef<int64_t> warpLoops,
                                             ArrayRef<int64_t> shapeOfLanes,
                                             ArrayRef<int64_t> laneLoops) {
  return FragmentsLayoutAttr::get(context, shapeOfWarps, warpLoops,
                                  shapeOfLanes, laneLoops, 1);
}

SmallVector<int64_t, 2> FragmentsLayoutAttr::getShape() const {
  auto shapeOfWarps = getShapeOfWarpsRef();
  auto warpLoops = getWarpLoopsRef();
  auto shapeOfLanes = getShapeOfLanesRef();
  auto laneLoops = getLaneLoopsRef();
  auto rank = getRank();
  SmallVector<int64_t, 2> layoutShape(rank);
  for (unsigned i = 0; i < rank; ++i)
    layoutShape[i] =
        shapeOfWarps[i] * warpLoops[i] * shapeOfLanes[i] * laneLoops[i];
  return layoutShape;
}

AffineMap FragmentsLayoutAttr::getLayoutMap() const {
  auto rank = getRank();

  auto shapeOfWarps = getShapeOfWarpsRef();
  SmallVector<int64_t, 2> stridesOfWarps;
  if (getMajorAxis() == 1)
    stridesOfWarps = computeStrides(shapeOfWarps);
  else
    stridesOfWarps = transpose(computeStrides(transpose(shapeOfWarps)));
  for (unsigned i = 0; i < rank; ++i)
    stridesOfWarps[i] *= numLanes;

  auto shapeOfLanes = getShapeOfLanesRef();
  SmallVector<int64_t, 2> stridesOfLanes;
  if (getMajorAxis() == 1)
    stridesOfLanes = computeStrides(shapeOfLanes);
  else
    stridesOfLanes = transpose(computeStrides(transpose(shapeOfLanes)));

  auto warpLoops = getWarpLoopsRef();
  auto laneLoops = getLaneLoopsRef();

  SmallVector<AffineExpr, 2> indexExprs;
  for (unsigned i = 0; i < rank; ++i)
    indexExprs.push_back(getAffineDimExpr(i, getContext()));

  auto threadExpr = getAffineConstantExpr(0, getContext());
  for (unsigned i = 0; i < rank; ++i) {
    auto tmpShape =
        ArrayRef{shapeOfWarps[i], warpLoops[i], shapeOfLanes[i], laneLoops[i]};
    auto tmpExprs = delinearize(indexExprs[i], tmpShape);
    threadExpr = threadExpr + (tmpExprs[0] * stridesOfWarps[i] +
                               tmpExprs[2] * stridesOfLanes[i]);
  }
  return AffineMap::get(rank, 0, threadExpr);
}

unsigned AxisSliceLayoutAttr::getRank() const {
  if (auto fragsLayout = llvm::dyn_cast<FragmentsLayoutAttr>(getParent()))
    return fragsLayout.getRank() - 1;
  if (auto nvmmaLayout = llvm::dyn_cast<NvidiaMmaLayoutAttr>(getParent()))
    return nvmmaLayout.getRank() - 1;
  llvm_unreachable("invalid parent layout");
}

SmallVector<int64_t, 2> AxisSliceLayoutAttr::getShape() const {
  if (auto fragsLayout = llvm::dyn_cast<FragmentsLayoutAttr>(getParent())) {
    auto layoutShape = fragsLayout.getShape();
    layoutShape.erase(layoutShape.begin() + getAxis());
    return layoutShape;
  }
  if (auto nvmmaLayout = llvm::dyn_cast<NvidiaMmaLayoutAttr>(getParent())) {
    auto layoutShape = nvmmaLayout.getShape();
    layoutShape.erase(layoutShape.begin() + getAxis());
    return layoutShape;
  }
  llvm_unreachable("invalid parent layout");
}

AffineMap AxisSliceLayoutAttr::getLayoutMap() const {
  auto axis = getAxis();
  auto rank = getRank();
  SmallVector<AffineExpr, 2> indexExprs;
  for (unsigned i = 0; i < rank + 1; ++i) {
    if (i != axis) {
      auto j = i > axis ? i - 1 : i;
      indexExprs.push_back(getAffineDimExpr(j, getContext()));
    } else {
      indexExprs.push_back(getAffineConstantExpr(0, getContext()));
    }
  }
  auto indicesMap = AffineMap::get(rank, 0, indexExprs, getContext());
  if (auto fragsLayout = llvm::dyn_cast<FragmentsLayoutAttr>(getParent()))
    return fragsLayout.getLayoutMap().compose(indicesMap);
  if (auto nvmmaLayout = llvm::dyn_cast<NvidiaMmaLayoutAttr>(getParent()))
    return nvmmaLayout.getLayoutMap().compose(indicesMap);
  llvm_unreachable("invalid parent layout");
}

SmallVector<int64_t, 2> NvidiaMmaLayoutAttr::getShapeOfLanes() const {
  return SmallVector<int64_t, 2>{8, 4};
}

SmallVector<int64_t, 2> NvidiaMmaLayoutAttr::getLaneLoops() const {
  return SmallVector<int64_t, 2>{1, 2};
}

FragmentsLayoutAttr NvidiaMmaLayoutAttr::toFragmentsLayout() const {
  return FragmentsLayoutAttr::get(getContext(), getShapeOfWarpsRef(),
                                  getWarpLoopsRef(), getShapeOfLanes(),
                                  getLaneLoops());
}

SmallVector<int64_t, 2> NvidiaMmaLayoutAttr::getShape() const {
  return toFragmentsLayout().getShape();
}

AffineMap NvidiaMmaLayoutAttr::getLayoutMap() const {
  return toFragmentsLayout().getLayoutMap();
}

unsigned MmOperandLayoutAttr::getRank() const {
  if (auto fragsLayout = llvm::dyn_cast<FragmentsLayoutAttr>(getParent()))
    return fragsLayout.getRank();
  if (auto nvmmaLayout = llvm::dyn_cast<NvidiaMmaLayoutAttr>(getParent()))
    return nvmmaLayout.getRank();
  llvm_unreachable("invalid parent layout");
}

LogicalResult ChangeOp::canonicalize(ChangeOp op, PatternRewriter &rewriter) {
  auto operandType = op.getOperand().getType();
  auto resultType = op.getType();
  if (resultType == operandType) {
    op.replaceAllUsesWith(op.getOperand());
    return success();
  }
  // Do not rewrite layout shortcut change.
  if (isLayoutShortcut(operandType.getEncoding(), resultType.getEncoding()))
    return failure();

  auto *defOp = op.getOperand().getDefiningOp();
  if (!defOp)
    return failure();
  if (auto changeOp = dyn_cast<ChangeOp>(defOp)) {
    rewriter.replaceOpWithNewOp<ChangeOp>(op, resultType,
                                          changeOp.getOperand());
    return success();
  }
  if (auto constantOp = dyn_cast<arith::ConstantOp>(defOp)) {
    if (auto splatAttr = dyn_cast<SplatElementsAttr>(constantOp.getValue())) {
      rewriter.replaceOpWithNewOp<arith::ConstantOp>(
          op, SplatElementsAttr::get(resultType,
                                     splatAttr.getSplatValue<Attribute>()));
      return success();
    }
  }
  if (auto arangeOp = dyn_cast<ArangeOp>(defOp)) {
    rewriter.replaceOpWithNewOp<ArangeOp>(op, resultType, arangeOp.getStart(),
                                          arangeOp.getEnd());
    return success();
  }
  if (auto splatOp = dyn_cast<SplatOp>(defOp)) {
    rewriter.replaceOpWithNewOp<SplatOp>(op, resultType, splatOp.getOperand());
    return success();
  }
  if (auto loadOp = dyn_cast<LocalLoadOp>(defOp)) {
    rewriter.replaceOpWithNewOp<LocalLoadOp>(op, resultType,
                                             loadOp.getSource());
    return success();
  }
  return failure();
}

void AsyncCopyOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>>
        &effects) {
  effects.emplace_back(MemoryEffects::Read::get(), &getSourceMutable(),
                       GlobalMemory::get());
  effects.emplace_back(MemoryEffects::Write::get(), &getTargetMutable(),
                       SharedMemory::get());
}

LogicalResult AsyncCopyOp::verify() {
  auto sourceType = cast<KapyMemRefType>(getSource().getType());
  if (!hasLayout<GlobalMemLayoutAttr>(sourceType))
    return emitOpError("source must have global memory layout");
  auto targetType = cast<KapyMemRefType>(getTarget().getType());
  if (!hasLayout<SharedMemLayoutAttr>(targetType))
    return emitOpError("target must have shared memory layout");
  if (sourceType.getShape() != targetType.getShape())
    return emitOpError("source and target must have same shape");
  if (sourceType.getElementType() != targetType.getElementType())
    return emitOpError("source and result must have same element type");
  return success();
}

void LocalAllocOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>>
        &effects) {
  // If the allocation has operand, mark it as no side effect allow passes like
  // CSE, DCE to work in early compiler passes.
  // After the memory offset is computed, we attach the true side effect.
  if (getOperand() && !getOperation()->hasAttr(shmemOffsetAttrName))
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
  if (!hasLayout<FragmentsLayoutAttr>(operandType))
    return emitOpError("operand must have fragments layout");
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
  effects.emplace_back(MemoryEffects::Read::get(), &getSourceMutable(),
                       SharedMemory::get());
}

LogicalResult LocalLoadOp::verify() {
  auto sourceType = cast<KapyMemRefType>(getSource().getType());
  if (!hasLayout<SharedMemLayoutAttr>(sourceType))
    return emitOpError("source must have shared memory layout");
  auto resultType = getType();
  if (sourceType.getShape() != resultType.getShape())
    return emitOpError("source and result must have same shape");
  if (sourceType.getElementType() != resultType.getElementType())
    return emitOpError("source and result must have same element type");
  return success();
}

int64_t kapy::getNvidiaCC(ModuleOp module) {
  return cast<IntegerAttr>(module->getAttr(nvidiaCCAttrName)).getInt();
}

int64_t kapy::getNumWarps(ModuleOp module) {
  return cast<IntegerAttr>(module->getAttr(numWarpsAttrName)).getInt();
}

int64_t kapy::getSharedMemNeeded(ModuleOp module) {
  if (!module->hasAttr(shmemNeededAttrName))
    llvm_unreachable("shared memory needed has not been analyzed yet");
  return cast<IntegerAttr>(module->getAttr(shmemNeededAttrName)).getInt();
}

int64_t kapy::getSharedMemOffset(Operation *op) {
  if (!op->hasAttr(shmemOffsetAttrName))
    return 0;
  return cast<IntegerAttr>(op->getAttr(shmemOffsetAttrName)).getInt();
}

bool kapy::supportNvidiaMma(MatmulOp matmulOp) {
  auto lhsElementType = matmulOp.getLhs().getType().getElementType();
  auto rhsElementType = matmulOp.getRhs().getType().getElementType();
  if (lhsElementType.isF32() && rhsElementType.isF32()) {
    auto matmulFormat = matmulOp.getMatmulFormat();
    return matmulFormat == MatmulFormat::TF32 ||
           matmulFormat == MatmulFormat::TF32X3;
  }
  return supportNvidiaMma(lhsElementType) && supportNvidiaMma(rhsElementType);
}

bool kapy::supportNvidiaMma(Type elementType) {
  bool isF8 = elementType.isFloat8E4M3FNUZ() || elementType.isFloat8E5M2() ||
              elementType.isFloat8E5M2FNUZ();
  return isF8 || elementType.isBF16() || elementType.isF16() ||
         elementType.isF32() || elementType.isInteger(8);
}

bool kapy::isNvidiaMmaToMmOperandShortcut(NvidiaMmaLayoutAttr nvmmaLayout,
                                          MmOperandLayoutAttr mmopdLayout) {
  if (nvmmaLayout != mmopdLayout.getParent())
    return false;
  if (mmopdLayout.getOperandIndex() != 0 || mmopdLayout.getBitWidth() != 16)
    return false;
  if (nvmmaLayout.getShapeOfWarpsRef()[1] != 1)
    return false;
  return true;
}

bool kapy::isNvidiaMmaToFragmentsShortcut(NvidiaMmaLayoutAttr nvmmaLayout,
                                          FragmentsLayoutAttr fragsLayout) {
  return nvmmaLayout.toFragmentsLayout() == fragsLayout;
}

bool kapy::isLayoutShortcut(Attribute oldLayout, Attribute newLayout) {
  auto nvmmaLayout = dyn_cast<NvidiaMmaLayoutAttr>(oldLayout);
  if (!nvmmaLayout)
    return false;
  auto mmopdLayout = dyn_cast<MmOperandLayoutAttr>(newLayout);
  if (mmopdLayout && isNvidiaMmaToMmOperandShortcut(nvmmaLayout, mmopdLayout))
    return true;
  auto fragsLayout = dyn_cast<FragmentsLayoutAttr>(newLayout);
  if (fragsLayout && isNvidiaMmaToFragmentsShortcut(nvmmaLayout, fragsLayout))
    return true;
  return false;
}

static AffineMap getTensorMapImpl(ArrayRef<int64_t> shape,
                                  ArrayRef<int64_t> layoutShape,
                                  AffineMap layoutMap) {
  auto rank = shape.size();
  auto *context = layoutMap.getContext();
  SmallVector<AffineExpr, 2> indexExprs;
  for (unsigned i = 0; i < rank; ++i)
    indexExprs.push_back(getAffineDimExpr(i, context));
  for (unsigned i = 0; i < rank; ++i)
    if (shape[i] > layoutShape[i])
      indexExprs[i] = indexExprs[i] % layoutShape[i];
  auto indicesMap = AffineMap::get(rank, 0, indexExprs, context);
  return AffineMap::get(rank, 0, layoutMap.getResult(0).compose(indicesMap));
}

AffineMap kapy::getTensorMap(ArrayRef<int64_t> shape, Attribute layout) {
  if (auto fragsLayout = dyn_cast<FragmentsLayoutAttr>(layout))
    return getTensorMapImpl(shape, fragsLayout.getShape(),
                            fragsLayout.getLayoutMap());
  if (auto sliceLayout = dyn_cast<AxisSliceLayoutAttr>(layout))
    return getTensorMapImpl(shape, sliceLayout.getShape(),
                            sliceLayout.getLayoutMap());
  if (auto nvmmaLayout = dyn_cast<NvidiaMmaLayoutAttr>(layout))
    return getTensorMapImpl(shape, nvmmaLayout.getShape(),
                            nvmmaLayout.getLayoutMap());
  llvm_unreachable("unsupported layout");
}

static unsigned getIdStrWidth(int64_t id) {
  // Maximum id supported now is 999.
  assert(id >= 0 && id < 1000);
  if (id >= 100)
    return 3;
  if (id >= 10)
    return 2;
  return 1;
}

static std::string getElementString(int64_t threadId, unsigned maxIdStrWidth) {
  std::string elemStr = "";
  auto curIdStrWidth = getIdStrWidth(threadId);
  for (unsigned i = 0; i < maxIdStrWidth - curIdStrWidth; ++i)
    elemStr += " ";
  elemStr += "T" + std::to_string(threadId);
  return elemStr;
}

static std::string getLinePrefixString(int64_t index,
                                       ArrayRef<int64_t> strides) {
  std::string prefixStr = index == 0 ? "[" : "\n ";
  for (unsigned i = 0; i < strides.size() - 1; ++i)
    prefixStr += index % strides[i] == 0 ? "[" : " ";
  return prefixStr;
}

static std::string getLineSuffixString(int64_t index, int64_t numElems,
                                       ArrayRef<int64_t> strides) {
  std::string suffixStr = "";
  for (unsigned i = 0; i < strides.size() - 1; ++i)
    if ((index + 1) % strides[i] == 0)
      suffixStr += "]";
  if (index + 1 == numElems)
    suffixStr += "]";
  else
    suffixStr += ",";
  return suffixStr;
}

std::string kapy::getLayoutString(RankedTensorType tensorType) {
  auto shape = tensorType.getShape();
  auto tensorMap = getTensorMap(shape, tensorType.getEncoding());
  auto rank = shape.size();
  auto numElems = product(shape);
  auto strides = computeStrides(shape);
  std::string layoutStr = "";
  for (int64_t index = 0; index < numElems; ++index) {
    auto indices = delinearize(index, shape);
    auto threadId = tensorMap.compose(indices)[0];
    // line prefix
    if (index % shape[rank - 1] == 0)
      layoutStr += getLinePrefixString(index, strides);
    // element
    layoutStr += getElementString(threadId, 3);
    // line suffix
    if ((index + 1) % shape[rank - 1] == 0)
      layoutStr += getLineSuffixString(index, numElems, strides);
    // ", " after each element (except end of line)
    else
      layoutStr += ", ";
  }
  return layoutStr;
}
