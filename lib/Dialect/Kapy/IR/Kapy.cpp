//===- Kapy.cpp -------------------------------------------------*- C++ -*-===//
//
// Copyright 2018-2020 Philippe Tillet
// Copyright 2020-2022 OpenAI
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.
//
//===----------------------------------------------------------------------===//
//
// This file is modified from the triton project.
// https://github.com/triton-lang/triton
//
//===----------------------------------------------------------------------===//

#include "kapy/Dialect/Kapy/IR/Kapy.h"
#include "kapy/Support/CommonUtils.h"
#include "mlir/IR//DialectImplementation.h"
#include "mlir/Interfaces/FunctionImplementation.h"
#include "mlir/Transforms/InliningUtils.h"
#include "llvm/ADT/TypeSwitch.h"

using namespace mlir;
using namespace mlir::kapy;

#include "kapy/Dialect/Kapy/IR/Dialect.cpp.inc"
#include "kapy/Dialect/Kapy/IR/Enums.cpp.inc"

#define GET_ATTRDEF_CLASSES
#include "kapy/Dialect/Kapy/IR/Attrs.cpp.inc"

#define GET_OP_CLASSES
#include "kapy/Dialect/Kapy/IR/Ops.cpp.inc"

static constexpr char inlineableAttrName[] = "inlineable";

namespace {

class KapyOpAsmInterface : public OpAsmDialectInterface {
public:
  using OpAsmDialectInterface::OpAsmDialectInterface;

  virtual AliasResult getAlias(Attribute attr,
                               llvm::raw_ostream &os) const override {
    if (auto encoding = dyn_cast<EncodingAttr>(attr)) {
      switch (encoding.getMemory()) {
      case (MemorySpace::GLOBAL_MEMORY):
        os << "global";
        break;
      case (MemorySpace::SHARED_MEMORY):
        os << "shared";
        break;
      case (MemorySpace::REGISTER_FILE):
        os << "values";
        break;
      }
      return AliasResult::FinalAlias;
    }
    return OpAsmDialectInterface::getAlias(attr, os);
  }
};

class KapyInlinerInterface : public DialectInlinerInterface {
public:
  using DialectInlinerInterface::DialectInlinerInterface;

  virtual bool isLegalToInline(Operation *caller, Operation *callee,
                               bool wouldBeCloned) const override {
    auto funcOp = dyn_cast<FuncOp>(callee);
    if (!funcOp)
      return true;
    if (funcOp->hasAttr(inlineableAttrName))
      return funcOp->getAttrOfType<BoolAttr>(inlineableAttrName).getValue();
    return true;
  }

  virtual bool isLegalToInline(Region *parent, Region *region,
                               bool wouldBeCloned,
                               IRMapping &mapping) const override {
    return true;
  }

  virtual bool isLegalToInline(Operation *op, Region *region,
                               bool wouldBeCloned,
                               IRMapping &mapping) const override {
    return true;
  }

  virtual void handleTerminator(Operation *op, Block *block) const override {
    auto returnOp = dyn_cast<ReturnOp>(op);
    if (!returnOp)
      return;
    OpBuilder builder(op);
    builder.create<cf::BranchOp>(op->getLoc(), block, returnOp.getOperands());
    op->erase();
  }

  virtual void handleTerminator(Operation *op,
                                ValueRange valuesToReplace) const override {
    auto returnOp = cast<ReturnOp>(op);
    assert(returnOp.getNumOperands() == valuesToReplace.size());
    for (auto it : llvm::enumerate(returnOp.getOperands()))
      valuesToReplace[it.index()].replaceAllUsesWith(it.value());
  }
};

} // namespace

void KapyDialect::initialize() {
  addAttributes<
#define GET_ATTRDEF_LIST
#include "kapy/Dialect/Kapy/IR/Attrs.cpp.inc"
      >();
  addOperations<
#define GET_OP_LIST
#include "kapy/Dialect/Kapy/IR/Ops.cpp.inc"
      >();
  addInterfaces<KapyOpAsmInterface>();
  addInterfaces<KapyInlinerInterface>();
}

Operation *KapyDialect::materializeConstant(OpBuilder &builder, Attribute value,
                                            Type type, Location loc) {
  return arith::ConstantOp::materialize(builder, value, type, loc);
}

LogicalResult FPToFPOp::verify() {
  if (isDownCast() && !getRoundingMode())
    return emitOpError("rounding mode is required for down cast");
  return success();
}

bool FPToFPOp::isUpCast() {
  auto oldBitWidth = getIntOrFloatBitWidth(getSource().getType());
  auto newBitWidth = getIntOrFloatBitWidth(getType());
  return newBitWidth > oldBitWidth;
}

bool FPToFPOp::isDownCast() {
  auto oldBitWidth = getIntOrFloatBitWidth(getSource().getType());
  auto newBitWidth = getIntOrFloatBitWidth(getType());
  return newBitWidth < oldBitWidth;
}

LogicalResult SvGlobalOp::canonicalize(SvGlobalOp op,
                                       PatternRewriter &rewriter) {
  auto inOp = op.getSource().getDefiningOp<SvGlobalOp>();
  if (!inOp)
    return failure();
  Value startX = rewriter.create<arith::AddIOp>(op.getLoc(), inOp.getStartX(),
                                                op.getStartX());
  Value startY = rewriter.create<arith::AddIOp>(op.getLoc(), inOp.getStartY(),
                                                op.getStartY());
  Value endX = rewriter.create<arith::AddIOp>(op.getLoc(), inOp.getStartX(),
                                              op.getEndX());
  Value endY = rewriter.create<arith::AddIOp>(op.getLoc(), inOp.getStartY(),
                                              op.getEndY());
  rewriter.replaceOpWithNewOp<SvGlobalOp>(op, op.getType(), inOp.getSource(),
                                          startX, endX, startY, endY);
  return success();
}

LogicalResult SvGlobalOp::verify() {
  auto sourceType = getSource().getType();
  auto resultType = getType();
  if (sourceType.getElementType() != resultType.getElementType())
    return emitOpError("source and result must have same element type");
  if (!hasLayout(sourceType) && !hasLayout(resultType))
    return success();
  if (!hasLayout(sourceType) || !hasLayout(resultType))
    return emitOpError("source and result must both have or without layout");
  if (getLayout(sourceType) != getLayout(resultType))
    return emitOpError("source and result must have same layout");
  return success();
}

void LdGlobalOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>>
        &effects) {
  effects.emplace_back(MemoryEffects::Read::get(), &getSourceMutable(),
                       GlobalMemory::get());
  if (isVolatile())
    effects.emplace_back(MemoryEffects::Write::get(),
                         SideEffects::DefaultResource::get());
}

void StGlobalOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>>
        &effects) {
  effects.emplace_back(MemoryEffects::Write::get(), &getTargetMutable(),
                       GlobalMemory::get());
}

void AtomicRMWOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>>
        &effects) {
  effects.emplace_back(MemoryEffects::Read::get(), &getSourceMutable(),
                       GlobalMemory::get());
  effects.emplace_back(MemoryEffects::Write::get(), &getSourceMutable(),
                       GlobalMemory::get());
}

LogicalResult AtomicRMWOp::verify() {
  auto elementType = getSource().getType().getElementType();
  switch (getKind()) {
  case (AtomicRMWKind::ADD): {
    if (elementType.isF16() || elementType.isBF16() || elementType.isF32())
      return success();
    else
      return emitOpError("unsupported element type");
  }
  case (AtomicRMWKind::MAX):
  case (AtomicRMWKind::MIN): {
    if (elementType.isF16() || elementType.isBF16())
      return success();
    else
      return emitOpError("unsupported element type");
  }
  }
  return emitOpError("usupported atomic rmw kind");
}

void MkSharedOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>>
        &effects) {
  if (std::distance(getResult().use_begin(), getResult().use_end()) == 0)
    return;
  effects.emplace_back(MemoryEffects::Allocate::get(), SharedMemory::get());
}

LogicalResult SvSharedOp::canonicalize(SvSharedOp op,
                                       PatternRewriter &rewriter) {
  auto inOp = op.getSource().getDefiningOp<SvSharedOp>();
  if (!inOp)
    return failure();
  Value startX = rewriter.create<arith::AddIOp>(op.getLoc(), inOp.getStartX(),
                                                op.getStartX());
  Value startY = rewriter.create<arith::AddIOp>(op.getLoc(), inOp.getStartY(),
                                                op.getStartY());
  Value endX = rewriter.create<arith::AddIOp>(op.getLoc(), inOp.getStartX(),
                                              op.getEndX());
  Value endY = rewriter.create<arith::AddIOp>(op.getLoc(), inOp.getStartY(),
                                              op.getEndY());
  rewriter.replaceOpWithNewOp<SvSharedOp>(op, op.getType(), inOp.getSource(),
                                          startX, endX, startY, endY);
  return success();
}

LogicalResult SvSharedOp::verify() {
  auto sourceType = getSource().getType();
  auto resultType = getType();
  if (sourceType.getElementType() != resultType.getElementType())
    return emitOpError("source and result must have same element type");
  if (!hasLayout(sourceType) && !hasLayout(resultType))
    return success();
  if (!hasLayout(sourceType) || !hasLayout(resultType))
    return emitOpError("source and result must both have or without layout");
  if (getLayout(sourceType) != getLayout(resultType))
    return emitOpError("source and result must have same layout");
  return success();
}

void LdSharedOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>>
        &effects) {
  effects.emplace_back(MemoryEffects::Read::get(), &getSourceMutable(),
                       SharedMemory::get());
}

void StSharedOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>>
        &effects) {
  effects.emplace_back(MemoryEffects::Write::get(), &getTargetMutable(),
                       SharedMemory::get());
}

void LdMatrixOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>>
        &effects) {
  effects.emplace_back(MemoryEffects::Read::get(), &getSourceMutable(),
                       SharedMemory::get());
}

LogicalResult LdMatrixOp::verify() {
  auto loaderType = getLoader().getType();
  auto resultType = getType();
  if (getIntOrFloatBitWidth(resultType) > 32)
    return emitOpError("element type can not more than 32 bits");
  if (!hasLayout(loaderType) && !hasLayout(resultType))
    return success();
  if (!hasLayout(loaderType) || !hasLayout(resultType))
    return emitOpError("loader and result must both have or without layout");
  auto &dialect = getLayout(loaderType).getDialect();
  auto *interface = cast<KapyLayoutInterface>(&dialect);
  return interface->verifyLdMatrixOpLayouts(*this);
}

void CpAsyncGlobalToSharedOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>>
        &effects) {
  effects.emplace_back(MemoryEffects::Read::get(), &getSourceMutable(),
                       GlobalMemory::get());
  effects.emplace_back(MemoryEffects::Write::get(), &getTargetMutable(),
                       SharedMemory::get());
}

void WarpIdOp::inferResultRanges(ArrayRef<ConstantIntRanges> argRanges,
                                 SetIntRangeFn setResultRange) {
  auto numWarps = getNumWarps((*this)->getParentOfType<ModuleOp>());
  auto minAPInt = APInt(32, 0);
  auto maxAPInt = APInt(32, numWarps - 1);
  auto intRange = ConstantIntRanges::range(minAPInt, maxAPInt, false);
  setResultRange(getResult(), intRange);
}

OpFoldResult ArangeOp::fold(FoldAdaptor adaptor) {
  if (adaptor.getStart() + 1 == adaptor.getEnd())
    return SplatElementsAttr::get(getType(), adaptor.getStartAttr());
  return OpFoldResult();
}

LogicalResult ArangeOp::verify() {
  auto start = getStart();
  auto end = getEnd();
  if (start >= end)
    return emitOpError("start must smaller than end");
  auto axis = getAxis();
  auto shape = getType().getShape();
  if (axis == 0 && shape[1] != 1)
    return emitOpError("result must be a column vector");
  if (axis == 1 && shape[0] != 1)
    return emitOpError("result must be a row vector");
  if (end - start != shape[axis])
    return emitOpError("number of elements in result is incorrect");
  return success();
}

LogicalResult MatmulOp::verify() {
  auto lhsType = getLhs().getType();
  auto rhsType = getRhs().getType();
  auto accType = getAcc().getType();

  auto lhsShape = lhsType.getShape();
  auto rhsShape = rhsType.getShape();
  auto accShape = accType.getShape();
  if (lhsShape[1] != rhsShape[0])
    return emitOpError(
        "number of columns of lhs must be equal to number of rows of rhs");
  if (accShape[0] != lhsShape[0])
    return emitOpError(
        "number of rows of acc must be equal to number of rows of lhs");
  if (accShape[1] != rhsShape[1])
    return emitOpError(
        "number of columns of acc must be equal to number of columns of rhs");

  if (lhsType.getElementTypeBitWidth() != rhsType.getElementTypeBitWidth())
    return emitOpError("lhs and rhs must have same element bit width");

  auto lhsElementType = lhsType.getElementType();
  auto rhsElementType = rhsType.getElementType();
  auto accElementType = accType.getElementType();

  // TODO: Support more.
  if (!accElementType.isF16() && !accElementType.isF32())
    return emitOpError("result must have f16 or f32 element type");

  switch (getMatmulImplWay()) {
  case (MatmulImplWay::MMA_M16N8K8_F16): {
    if (!lhsElementType.isF16() && !lhsElementType.isBF16())
      return emitOpError(
          "mma_m16n8k8_f16 operands must have f16 or bf16 element type");
    if (!rhsElementType.isF16() && !rhsElementType.isBF16())
      return emitOpError(
          "mma_m16n8k8_f16 operands must have f16 or bf16 element type");
    if (accShape[0] < 16 || accShape[1] < 8)
      return emitOpError("mma_m16n8k8_f16 acc is at least 16x8");
    if (lhsShape[0] < 16 || lhsShape[1] < 8)
      return emitOpError("mma_m16n8k8_f16 lhs is at least 16x8");
    if (rhsShape[0] < 8 || rhsShape[1] < 8)
      return emitOpError("mma_m16n8k8_f16 rhs is at least 8x8");
    break;
  }

  case (MatmulImplWay::MMA_M16N8K16_F16): {
    if (!lhsElementType.isF16() && !lhsElementType.isBF16())
      return emitOpError(
          "mma_m16n8k16_f16 operands must have f16 or bf16 element type");
    if (!rhsElementType.isF16() && !rhsElementType.isBF16())
      return emitOpError(
          "mma_m16n8k16_f16 operands must have f16 or bf16 element type");
    if (accShape[0] < 16 || accShape[1] < 8)
      return emitOpError("mma_m16n8k16_f16 acc is at least 16x8");
    if (lhsShape[0] < 16 || lhsShape[1] < 16)
      return emitOpError("mma_m16n8k16_f16 lhs is at least 16x16");
    if (rhsShape[0] < 16 || rhsShape[1] < 8)
      return emitOpError("mma_m16n8k16_f16 rhs is at least 16x8");
    break;
  }

  case (MatmulImplWay::MMA_M16N8K8_TF32): {
    if (!lhsElementType.isF32() || !rhsElementType.isF32())
      return emitOpError(
          "mma_m16n8k8_tf32 operands must have f32 element type");

    if (accShape[0] < 16 || accShape[1] < 8)
      return emitOpError("mma_m16n8k8_tf32 acc is at least 16x8");
    if (lhsShape[0] < 16 || lhsShape[1] < 8)
      return emitOpError("mma_m16n8k8_tf32 lhs is at least 16x8");
    if (rhsShape[0] < 8 || rhsShape[1] < 8)
      return emitOpError("mma_m16n8k8_tf32 rhs is at least 8x8");
    break;
  }

  case (MatmulImplWay::MMA_M16N8K16_F8): {
    if (!lhsElementType.isFloat8E4M3() && !lhsElementType.isFloat8E5M2())
      return emitOpError(
          "mma_m16n8k16_f8 operands must have f8E4M3 or f8E5M2 element type");
    if (!rhsElementType.isFloat8E4M3() && !rhsElementType.isFloat8E5M2())
      return emitOpError(
          "mma_m16n8k16_f8 operands must have f8E4M3 or f8E5M2 element type");
    if (accShape[0] < 16 || accShape[1] < 8)
      return emitOpError("mma_m16n8k16_f8 acc is at least 16x8");
    if (lhsShape[0] < 16 || lhsShape[1] < 16)
      return emitOpError("mma_m16n8k16_f8 lhs is at least 16x16");
    if (rhsShape[0] < 16 || rhsShape[1] < 8)
      return emitOpError("mma_m16n8k16_f8 rhs is at least 16x8");
    break;
  }
  }

  if (!hasLayout(lhsType) && !hasLayout(rhsType) && !hasLayout(accType))
    return success();
  if (!hasLayout(lhsType) || !hasLayout(rhsType) || !hasLayout(accType))
    return emitOpError("lhs and rhs and acc must all have or without layout");
  auto &dialect = getLayout(accType).getDialect();
  auto *interface = cast<KapyLayoutInterface>(&dialect);
  return interface->verifyMatmulOpLayouts(*this);
}

ParseResult ReduceOp::parse(OpAsmParser &parser, OperationState &state) {
  OpAsmParser::UnresolvedOperand source;
  Type sourceType, resultType;
  if (parser.parseOperand(source) ||
      parser.parseOptionalAttrDict(state.attributes))
    return failure();

  SmallVector<OpAsmParser::Argument, 2> bodyArgs;
  if (parser.parseKeyword("lambda") ||
      parser.parseArgumentList(bodyArgs, OpAsmParser::Delimiter::Paren, true))
    return failure();

  auto *region = state.addRegion();
  if (parser.parseRegion(*region, bodyArgs))
    return failure();

  if (parser.parseColonType(sourceType) || parser.parseArrow() ||
      parser.parseType(resultType) ||
      parser.resolveOperand(source, sourceType, state.operands))
    return failure();
  state.addTypes(resultType);

  return success();
}

void ReduceOp::print(OpAsmPrinter &printer) {
  printer << " " << getSource();
  printer.printOptionalAttrDict((*this)->getAttrs());
  printer << " lambda(";
  llvm::interleaveComma(getBody()->getArguments(), printer, [&](Value value) {
    printer << value << ": " << value.getType();
  });
  printer << ") ";
  printer.printRegion(getRegion(), false);
  printer << " : " << getSource().getType() << " -> " << getType();
}

LogicalResult
ReduceOp::inferReturnTypes(MLIRContext *context, std::optional<Location> loc,
                           ValueRange operands, DictionaryAttr attrs,
                           OpaqueProperties props, RegionRange regions,
                           SmallVectorImpl<Type> &returnTypes) {
  auto axis = props.as<Properties *>()->getAxis().getInt();
  auto sourceType = cast<RankedTensorType>(operands[0].getType());
  auto shape = llvm::to_vector<2>(sourceType.getShape());
  if (axis < 0 || axis >= shape.size())
    return emitOptionalError(loc, "invalid axis");
  shape[axis] = 1;
  returnTypes.push_back(cloneWithShape(sourceType, shape));
  return success();
}

LogicalResult ReduceOp::verifyRegions() {
  auto elementType = getSource().getType().getElementType();
  auto *body = getBody();
  if (body->getNumArguments() != 2)
    return emitOpError("body must take 2 arguments");

  const auto &bodyArgTypes = body->getArgumentTypes();
  for (unsigned i = 0; i < 2; ++i)
    if (bodyArgTypes[i] != elementType)
      return emitOpError("body argument ") << i << " type mismatch";

  auto returnOp = dyn_cast<ReturnOp>(body->getTerminator());
  if (!returnOp)
    return emitOpError("body must have a ")
           << ReturnOp::getOperationName() << " as terminator";
  if (returnOp.getNumOperands() != 1)
    return emitOpError("body must return one value");
  if (returnOp.getOperand(0).getType() != elementType)
    return emitOpError("body return type mismatch");

  return success();
}

OpFoldResult SplatOp::fold(FoldAdaptor adaptor) {
  if (auto source = adaptor.getSource())
    return SplatElementsAttr::get(getType(), source);
  return OpFoldResult();
}

LogicalResult BroadcastOp::canonicalize(BroadcastOp op,
                                        PatternRewriter &rewriter) {
  auto source = op.getSource();
  auto resultType = op.getType();
  if (resultType == source.getType()) {
    op.replaceAllUsesWith(source);
    return success();
  }
  auto *defOp = source.getDefiningOp();
  if (!defOp)
    return failure();
  if (auto inOp = dyn_cast<BroadcastOp>(defOp)) {
    rewriter.replaceOpWithNewOp<BroadcastOp>(op, resultType, inOp.getSource());
    return success();
  }
  if (auto inOp = dyn_cast<SplatOp>(defOp)) {
    rewriter.replaceOpWithNewOp<SplatOp>(op, resultType, inOp.getSource());
    return success();
  }
  return failure();
}

OpFoldResult BroadcastOp::fold(FoldAdaptor adaptor) {
  auto source = adaptor.getSource();
  if (auto splatAttr = dyn_cast_if_present<SplatElementsAttr>(source))
    return splatAttr.resizeSplat(getType());
  return OpFoldResult();
}

LogicalResult BroadcastOp::verify() {
  auto oldShape = getSource().getType().getShape();
  auto newShape = getType().getShape();
  if (oldShape.size() != newShape.size())
    return emitOpError("source and result must have same rank");
  for (unsigned i = 0; i < oldShape.size(); ++i) {
    if (oldShape[i] == 1)
      continue;
    if (oldShape[i] != newShape[i])
      return emitOpError("incompatible shape between source and result");
  }
  return success();
}

LogicalResult
TransposeOp::inferReturnTypes(MLIRContext *context, std::optional<Location> loc,
                              ValueRange operands, DictionaryAttr attrs,
                              OpaqueProperties props, RegionRange regions,
                              SmallVectorImpl<Type> &returnTypes) {
  auto sourceType = cast<RankedTensorType>(operands[0].getType());
  if (sourceType.getRank() != 2)
    return emitOptionalError(loc, "source must have rank 2");
  auto shape = transpose(sourceType.getShape());
  auto resultType = cloneWithShape(sourceType, shape);
  if (!hasLayout(sourceType)) {
    returnTypes.push_back(resultType);
    return success();
  }
  auto sourceLayout = getLayout(sourceType);
  auto &dialect = sourceLayout.getDialect();
  auto *interface = cast<KapyLayoutInterface>(&dialect);
  auto resultLayout = interface->inferTransposeOpLayout(sourceLayout);
  returnTypes.push_back(cloneWithLayout(resultType, resultLayout));
  return success();
}

LogicalResult TransposeOp::canonicalize(TransposeOp op,
                                        PatternRewriter &rewriter) {
  auto *defOp = op.getSource().getDefiningOp();
  if (!defOp)
    return failure();
  if (auto inOp = dyn_cast<TransposeOp>(defOp)) {
    op.replaceAllUsesWith(inOp.getSource());
    return success();
  }
  if (auto inOp = dyn_cast<SplatOp>(defOp)) {
    rewriter.replaceOpWithNewOp<SplatOp>(op, op.getType(), inOp.getSource());
    return success();
  }
  return failure();
}

OpFoldResult TransposeOp::fold(FoldAdaptor adaptor) {
  auto source = adaptor.getSource();
  if (auto splatAttr = dyn_cast_if_present<SplatElementsAttr>(source))
    return splatAttr.resizeSplat(getType());
  return OpFoldResult();
}

void ElementwiseExternLibOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>>
        &effects) {
  if (isPure())
    return;
  effects.emplace_back(MemoryEffects::Read::get(),
                       SideEffects::DefaultResource::get());
  effects.emplace_back(MemoryEffects::Write::get(),
                       SideEffects::DefaultResource::get());
}

void ElementwiseInlineAsmOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>>
        &effects) {
  if (isPure())
    return;
  effects.emplace_back(MemoryEffects::Read::get(),
                       SideEffects::DefaultResource::get());
  effects.emplace_back(MemoryEffects::Write::get(),
                       SideEffects::DefaultResource::get());
}

void FuncOp::build(OpBuilder &builder, OperationState &state, StringRef name,
                   FunctionType funcType, bool inlineable,
                   ArrayRef<DictionaryAttr> argAttrs) {
  state.addAttribute(SymbolTable::getSymbolAttrName(),
                     builder.getStringAttr(name));
  state.addAttribute(getFunctionTypeAttrName(state.name),
                     TypeAttr::get(funcType));
  state.addAttribute(inlineableAttrName, builder.getBoolAttr(inlineable));
  state.addRegion();

  if (!argAttrs.empty()) {
    assert(funcType.getNumInputs() == argAttrs.size());
    function_interface_impl::addArgAndResultAttrs(
        builder, state, argAttrs, {}, getArgAttrsAttrName(state.name),
        getResAttrsAttrName(state.name));
  }
}

ParseResult FuncOp::parse(OpAsmParser &parser, OperationState &state) {
  auto funcType = [](Builder &builder, ArrayRef<Type> inputs,
                     ArrayRef<Type> results,
                     function_interface_impl::VariadicFlag, std::string &) {
    return builder.getFunctionType(inputs, results);
  };

  return function_interface_impl::parseFunctionOp(
      parser, state, false, getFunctionTypeAttrName(state.name), funcType,
      getArgAttrsAttrName(state.name), getResAttrsAttrName(state.name));
}

void FuncOp::print(OpAsmPrinter &printer) {
  function_interface_impl::printFunctionOp(
      printer, *this, false, getFunctionTypeAttrName(), getArgAttrsAttrName(),
      getResAttrsAttrName());
}

LogicalResult CallOp::verifySymbolUses(SymbolTableCollection &symbolTable) {
  auto symbolAttr = this->getProperties().getCallee();
  if (!symbolAttr)
    return emitOpError("requires a symbol reference attribute");

  auto funcOp = symbolTable.lookupNearestSymbolFrom<FuncOp>(*this, symbolAttr);
  if (!funcOp)
    return emitOpError("'")
           << symbolAttr.getValue() << "' doesn't reference a valid function";

  auto funcType = funcOp.getFunctionType();
  if (funcType.getNumInputs() != getNumOperands())
    return emitOpError("has incorrect number of operands");
  for (unsigned i = 0; i < funcType.getNumInputs(); ++i)
    if (getOperand(i).getType() != funcType.getInput(i))
      return emitOpError("operand ") << i << " type mismatch";

  if (funcType.getNumResults() != getNumResults())
    return emitOpError("has incorrect number of results");
  for (unsigned i = 0; i < funcType.getNumResults(); ++i)
    if (getResult(i).getType() != funcType.getResult(i))
      return emitOpError("result ") << i << " type mismatch";

  return success();
}

LogicalResult ReturnOp::verify() {
  auto funcOp = dyn_cast<FuncOp>((*this)->getParentOp());
  if (!funcOp)
    return success();
  auto results = funcOp.getFunctionType().getResults();
  if (getNumOperands() != results.size())
    return emitOpError("has incorrect number of operands");
  for (unsigned i = 0; i < results.size(); ++i)
    if (getOperand(i).getType() != results[i])
      return emitOpError("operand ") << i << " type mismatch";
  return success();
}

unsigned kapy::getIntOrFloatBitWidth(Type type) {
  if (auto shapedType = dyn_cast<ShapedType>(type))
    return shapedType.getElementTypeBitWidth();
  return type.getIntOrFloatBitWidth();
}

int64_t kapy::getNvidiaCC(ModuleOp module) {
  if (!module->hasAttr(nvidiaCCAttrName))
    llvm_unreachable("can not get a named attribute");
  return cast<IntegerAttr>(module->getAttr(nvidiaCCAttrName)).getInt();
}

int64_t kapy::getNumWarps(ModuleOp module) {
  if (!module->hasAttr(numWarpsAttrName))
    llvm_unreachable("can not get a named attribute");
  return cast<IntegerAttr>(module->getAttr(numWarpsAttrName)).getInt();
}

int64_t kapy::getAlignment(Operation *op) {
  if (!op->hasAttr(alignmentAttrName))
    llvm_unreachable("can not get a named attribute");
  return cast<IntegerAttr>(op->getAttr(alignmentAttrName)).getInt();
}

bool kapy::isGlobalRead(Operation *op) {
  auto effectOp = dyn_cast<MemoryEffectOpInterface>(op);
  if (!effectOp)
    return false;
  SmallVector<SideEffects::EffectInstance<MemoryEffects::Effect>> effects;
  effectOp.getEffects(effects);
  for (auto &effect : effects)
    if (effect.getResource() == GlobalMemory::get() &&
        isa<MemoryEffects::Read>(effect.getEffect()))
      return true;
  return false;
}

bool kapy::isGlobalWrite(Operation *op) {
  auto effectOp = dyn_cast<MemoryEffectOpInterface>(op);
  if (!effectOp)
    return false;
  SmallVector<SideEffects::EffectInstance<MemoryEffects::Effect>> effects;
  effectOp.getEffects(effects);
  for (auto &effect : effects)
    if (effect.getResource() == GlobalMemory::get() &&
        isa<MemoryEffects::Write>(effect.getEffect()))
      return true;
  return false;
}

bool kapy::isSharedRead(Operation *op) {
  auto effectOp = dyn_cast<MemoryEffectOpInterface>(op);
  if (!effectOp)
    return false;
  SmallVector<SideEffects::EffectInstance<MemoryEffects::Effect>> effects;
  effectOp.getEffects(effects);
  for (auto &effect : effects)
    if (effect.getResource() == SharedMemory::get() &&
        isa<MemoryEffects::Read>(effect.getEffect()))
      return true;
  return false;
}

bool kapy::isSharedWrite(Operation *op) {
  auto effectOp = dyn_cast<MemoryEffectOpInterface>(op);
  if (!effectOp)
    return false;
  SmallVector<SideEffects::EffectInstance<MemoryEffects::Effect>> effects;
  effectOp.getEffects(effects);
  for (auto &effect : effects)
    if (effect.getResource() == SharedMemory::get() &&
        isa<MemoryEffects::Write>(effect.getEffect()))
      return true;
  return false;
}

bool kapy::inGlobalMemory(RankedTensorType tensorType) {
  auto encoding = cast<EncodingAttr>(tensorType.getEncoding());
  return encoding.getMemory() == MemorySpace::GLOBAL_MEMORY;
}

bool kapy::inSharedMemory(RankedTensorType tensorType) {
  auto encoding = cast<EncodingAttr>(tensorType.getEncoding());
  return encoding.getMemory() == MemorySpace::SHARED_MEMORY;
}

bool kapy::inRegisterFile(RankedTensorType tensorType) {
  auto encoding = cast<EncodingAttr>(tensorType.getEncoding());
  return encoding.getMemory() == MemorySpace::REGISTER_FILE;
}

bool kapy::hasLayout(RankedTensorType tensorType) {
  auto encoding = cast<EncodingAttr>(tensorType.getEncoding());
  return encoding.getLayout() != nullptr;
}

Attribute kapy::getLayout(RankedTensorType tensorType) {
  auto encoding = cast<EncodingAttr>(tensorType.getEncoding());
  return encoding.getLayout();
}

RankedTensorType kapy::cloneWithShape(RankedTensorType tensorType,
                                      ArrayRef<int64_t> shape) {
  return RankedTensorType::get(shape, tensorType.getElementType(),
                               tensorType.getEncoding());
}

RankedTensorType kapy::cloneWithLayout(RankedTensorType tensorType,
                                       Attribute layout) {
  auto encoding = cast<EncodingAttr>(tensorType.getEncoding());
  auto *context = layout.getContext();
  encoding = EncodingAttr::get(context, encoding.getMemory(), layout);
  return RankedTensorType::get(tensorType.getShape(),
                               tensorType.getElementType(), encoding);
}
