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
    if (isa<RegistersLayoutAttr>(attr)) {
      os << "regis";
      return AliasResult::FinalAlias;
    }
    if (isa<SliceAxisLayoutAttr>(attr)) {
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
    return SliceAxisLayoutAttr::get(getContext(), operandLayout, axis);
  }

  virtual FailureOr<Attribute>
  inferUnsqueezeOpLayout(Attribute operandLayout, unsigned axis,
                         std::optional<Location> loc) const override {
    auto sliceLayout = dyn_cast<SliceAxisLayoutAttr>(operandLayout);
    if (!sliceLayout)
      return emitOptionalError(loc, "operand must have slice axis layout");
    if (sliceLayout.getAxis() != axis)
      return emitOptionalError(loc, "incompatible slice axis for operand");
    return sliceLayout.getParent();
  }

  virtual FailureOr<Attribute>
  inferPermuteOpLayout(Attribute operandLayout, ArrayRef<int32_t> order,
                       std::optional<Location> loc) const override {
    auto regisLayout = dyn_cast<RegistersLayoutAttr>(operandLayout);
    if (!regisLayout)
      return emitOptionalError(loc, "operand must have registers layout");
    return RegistersLayoutAttr::get(
        getContext(), permute(regisLayout.getShapeOfWarpsRef(), order),
        permute(regisLayout.getWarpLoopsRef(), order),
        permute(regisLayout.getShapeOfLanesRef(), order),
        permute(regisLayout.getLaneLoopsRef(), order),
        permute(regisLayout.getOrderRef(), order));
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
    if (isa<RegistersLayoutAttr, NvidiaMmaLayoutAttr>(accumLayout)) {
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

RegistersLayoutAttr RegistersLayoutAttr::get(MLIRContext *context,
                                             ArrayRef<int64_t> shapeOfWarps,
                                             ArrayRef<int64_t> warpLoops,
                                             ArrayRef<int64_t> shapeOfLanes,
                                             ArrayRef<int64_t> laneLoops) {
  auto order = makeIota<unsigned>(shapeOfWarps.size());
  return RegistersLayoutAttr::get(context, shapeOfWarps, warpLoops,
                                  shapeOfLanes, laneLoops, order);
}

AffineMap RegistersLayoutAttr::getLayoutMap() const {
  auto rank = getRank();
  auto order = getOrderRef();

  auto shapeOfWarps = getShapeOfWarpsRef();
  auto stridesOfWarps =
      permute(computeStrides(permute(shapeOfWarps, order)), inverse(order));
  for (unsigned i = 0; i < rank; ++i)
    stridesOfWarps[i] *= numLanes;

  auto shapeOfLanes = getShapeOfLanesRef();
  auto stridesOfLanes =
      permute(computeStrides(permute(shapeOfLanes, order)), inverse(order));

  auto warpLoops = getWarpLoopsRef();
  auto laneLoops = getLaneLoopsRef();

  SmallVector<AffineExpr, 4> indexExprs;
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

AffineMap RegistersLayoutAttr::getTensorMap(ArrayRef<int64_t> shape) const {
  auto shapeOfWarps = getShapeOfWarpsRef();
  auto warpLoops = getWarpLoopsRef();
  auto shapeOfLanes = getShapeOfLanesRef();
  auto laneLoops = getLaneLoopsRef();

  auto rank = getRank();
  SmallVector<int64_t, 4> layoutShape(rank);
  for (unsigned i = 0; i < rank; ++i)
    layoutShape[i] =
        shapeOfWarps[i] * warpLoops[i] * shapeOfLanes[i] * laneLoops[i];

  SmallVector<AffineExpr, 4> indexExprs;
  for (unsigned i = 0; i < rank; ++i)
    indexExprs.push_back(getAffineDimExpr(i, getContext()));
  for (unsigned i = 0; i < rank; ++i)
    if (shape[i] > layoutShape[i])
      indexExprs[i] = indexExprs[i] % layoutShape[i];

  auto threadExpr = getLayoutMap().getResult(0);
  auto indexMap = AffineMap::get(rank, 0, indexExprs, getContext());
  threadExpr = threadExpr.compose(indexMap);
  return AffineMap::get(rank, 0, threadExpr);
}

unsigned SliceAxisLayoutAttr::getRank() const {
  if (auto regisLayout = llvm::dyn_cast<RegistersLayoutAttr>(getParent()))
    return regisLayout.getRank() - 1;
  if (auto sliceLayout = llvm::dyn_cast<SliceAxisLayoutAttr>(getParent()))
    return sliceLayout.getRank() - 1;
  if (auto nvmmaLayout = llvm::dyn_cast<NvidiaMmaLayoutAttr>(getParent()))
    return nvmmaLayout.getRank() - 1;
  llvm_unreachable("invalid parent layout");
}

SmallVector<int64_t, 4>
SliceAxisLayoutAttr::unsqueeze(ArrayRef<int64_t> shape) const {
  auto rank = shape.size();
  auto axis = getAxis();
  SmallVector<int64_t, 4> newShape;
  for (unsigned i = 0; i < rank + 1; ++i)
    if (i != axis)
      newShape.push_back(i > axis ? shape[i - 1] : shape[i]);
    else
      newShape.push_back(1);
  return newShape;
}

SmallVector<unsigned, 4> NvidiaMmaLayoutAttr::getOrder() const {
  return makeIota<unsigned>(getRank());
}

SmallVector<int64_t, 4> NvidiaMmaLayoutAttr::getShapeOfLanes() const {
  auto rank = getRank();
  SmallVector<int64_t, 4> shapeOfLanes(rank, 1);
  shapeOfLanes[rank - 1] = 4;
  shapeOfLanes[rank - 2] = 8;
  return shapeOfLanes;
}

SmallVector<int64_t, 4> NvidiaMmaLayoutAttr::getLaneLoops() const {
  auto rank = getRank();
  SmallVector<int64_t, 4> loopsPerLane(rank, 1);
  loopsPerLane[rank - 1] = 2;
  return loopsPerLane;
}

RegistersLayoutAttr NvidiaMmaLayoutAttr::toRegistersLayout() const {
  return RegistersLayoutAttr::get(getContext(), getShapeOfWarpsRef(),
                                  getWarpLoopsRef(), getShapeOfLanes(),
                                  getLaneLoops());
}

unsigned MmOperandLayoutAttr::getRank() const {
  if (auto regisLayout = llvm::dyn_cast<RegistersLayoutAttr>(getParent()))
    return regisLayout.getRank();
  if (auto nvmmaLayout = llvm::dyn_cast<NvidiaMmaLayoutAttr>(getParent()))
    return nvmmaLayout.getRank();
  llvm_unreachable("invalid parent layout");
}

LogicalResult ChangeOp::canonicalize(ChangeOp op, PatternRewriter &rewriter) {
  if (op.getType() == op.getOperand().getType()) {
    op.replaceAllUsesWith(op.getOperand());
    return success();
  }
  // Do not rewrite changes from nvidia mma layout to matmul operand layout.
  if (hasLayout<NvidiaMmaLayoutAttr>(op.getOperand().getType()) &&
      hasLayout<MmOperandLayoutAttr>(op.getType()))
    return failure();
  auto *defOp = op.getOperand().getDefiningOp();
  if (!defOp)
    return failure();
  if (auto loadOp = dyn_cast<LocalLoadOp>(defOp)) {
    rewriter.replaceOpWithNewOp<LocalLoadOp>(op, op.getType(),
                                             loadOp.getSource());
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
  if (sourceType.getShape() != resultType.getShape())
    return emitOpError("source and result must have same shape");
  if (sourceType.getElementType() != resultType.getElementType())
    return emitOpError("source and result must have same element type");
  return success();
}

int64_t kapy::getNvidiaCC(ModuleOp module) {
  if (!module->hasAttr(nvidiaCCAttrName))
    return 80;
  return cast<IntegerAttr>(module->getAttr(nvidiaCCAttrName)).getInt();
}

int64_t kapy::getNumWarps(ModuleOp module) {
  if (!module->hasAttr(numWarpsAttrName))
    return 4;
  return cast<IntegerAttr>(module->getAttr(numWarpsAttrName)).getInt();
}

bool kapy::supportNvidiaMma(MatmulOp op) {
  auto lhsElementType = op.getLhs().getType().getElementType();
  auto rhsElementType = op.getRhs().getType().getElementType();
  if (lhsElementType.isF32() && rhsElementType.isF32()) {
    auto matmulFormat = op.getMatmulFormat();
    return matmulFormat == MatmulFormat::TF32 ||
           matmulFormat == MatmulFormat::TF32x3;
  }
  return supportNvidiaMma(lhsElementType) && supportNvidiaMma(rhsElementType);
}

bool kapy::supportNvidiaMma(Type elementType) {
  bool isF8 = elementType.isFloat8E4M3FNUZ() || elementType.isFloat8E5M2() ||
              elementType.isFloat8E5M2FNUZ();
  return isF8 || elementType.isBF16() || elementType.isF16() ||
         elementType.isF32() || elementType.isInteger(8);
}

static unsigned getIdStrWidth(int64_t id) {
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

static std::string getLayoutStringImpl(RegistersLayoutAttr regisLayout,
                                       ArrayRef<int64_t> shape) {
  auto rank = shape.size();
  auto numElems = product(shape);
  auto strides = computeStrides(shape);
  auto tensorMap = regisLayout.getTensorMap(shape);
  auto numWarps = product(regisLayout.getShapeOfWarpsRef());
  auto maxIdStrWidth = getIdStrWidth(numLanes * numWarps - 1);
  std::string layoutStr = "";
  for (int64_t index = 0; index < numElems; ++index) {
    auto indices = delinearize(index, shape);
    auto threadId = tensorMap.compose(indices)[0];
    // line prefix
    if (index % shape[rank - 1] == 0)
      layoutStr += getLinePrefixString(index, strides);
    // element
    layoutStr += getElementString(threadId, maxIdStrWidth);
    // line suffix
    if ((index + 1) % shape[rank - 1] == 0)
      layoutStr += getLineSuffixString(index, numElems, strides);
    // ", " after each element (except end of line)
    else
      layoutStr += ", ";
  }
  return layoutStr;
}

std::string kapy::getLayoutString(RankedTensorType type) {
  if (auto regisLayout = dyn_cast<RegistersLayoutAttr>(type.getEncoding()))
    return getLayoutStringImpl(regisLayout, type.getShape());
  if (auto nvmmaLayout = dyn_cast<NvidiaMmaLayoutAttr>(type.getEncoding())) {
    auto regisLayout = nvmmaLayout.toRegistersLayout();
    return getLayoutStringImpl(regisLayout, type.getShape());
  }
  llvm_unreachable("unsupported layout");
}
