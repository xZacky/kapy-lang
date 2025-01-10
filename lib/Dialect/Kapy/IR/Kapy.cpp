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
#include "kapy/Dialect/Kapy/IR/Utils.h"

#include "mlir/IR/DialectImplementation.h"
#include "mlir/Interfaces/FunctionImplementation.h"
#include "mlir/Transforms/InliningUtils.h"
#include "llvm/ADT/TypeSwitch.h"

using namespace mlir;
using namespace mlir::kapy;

#include "kapy/Dialect/Kapy/IR/Dialect.cpp.inc"
#include "kapy/Dialect/Kapy/IR/Enums.cpp.inc"

#define GET_ATTRDEF_CLASSES
#include "kapy/Dialect/Kapy/IR/Attrs.cpp.inc"

#define GET_TYPEDEF_CLASSES
#include "kapy/Dialect/Kapy/IR/Types.cpp.inc"

#define GET_OP_CLASSES
#include "kapy/Dialect/Kapy/IR/Ops.cpp.inc"

namespace {

class KapyOpAsmInterface : public OpAsmDialectInterface {
public:
  using OpAsmDialectInterface::OpAsmDialectInterface;

  virtual AliasResult getAlias(Attribute attr,
                               llvm::raw_ostream &os) const override {
    if (isa<GlobalMemLayoutAttr>(attr)) {
      os << "glmem";
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
    if (funcOp->hasAttr("noinline"))
      return !funcOp->getAttrOfType<BoolAttr>("noinline").getValue();
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
                                ValueRange values) const override {
    auto returnOp = cast<ReturnOp>(op);
    assert(returnOp.getNumOperands() == values.size());
    for (auto it : llvm::enumerate(returnOp.getOperands()))
      values[it.index()].replaceAllUsesWith(it.value());
  }
};

} // namespace

void KapyDialect::initialize() {
  addAttributes<
#define GET_ATTRDEF_LIST
#include "kapy/Dialect/Kapy/IR/Attrs.cpp.inc"
      >();
  addTypes<
#define GET_TYPEDEF_LIST
#include "kapy/Dialect/Kapy/IR/Types.cpp.inc"
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

Attribute GlobalMemLayoutAttr::parse(AsmParser &parser, Type type) {
  if (failed(parser.parseLess()))
    return Attribute();
  SmallVector<int64_t, 4> strides;
  auto parseStride = [&]() -> ParseResult {
    auto stride = ShapedType::kDynamic;
    if (succeeded(parser.parseOptionalQuestion())) {
      strides.push_back(stride);
      return success();
    }
    if (succeeded(parser.parseInteger(stride))) {
      strides.push_back(stride);
      return success();
    }
    return failure();
  };
  if (failed(parser.parseCommaSeparatedList(AsmParser::Delimiter::Square,
                                            parseStride)))
    return Attribute();
  if (failed(parser.parseGreater()))
    return Attribute();
  return parser.getChecked<GlobalMemLayoutAttr>(parser.getContext(), strides);
}

void GlobalMemLayoutAttr::print(AsmPrinter &printer) const {
  printer << "<[";
  auto printStride = [&](int64_t stride) {
    if (ShapedType::isDynamic(stride))
      printer << "?";
    else
      printer << stride;
  };
  llvm::interleaveComma(getStrides(), printer, printStride);
  printer << "]>";
}

AffineMap GlobalMemLayoutAttr::getMemRefMap() const {
  auto resultExpr = getAffineConstantExpr(0, getContext());
  unsigned numDims = 0;
  unsigned numSyms = 0;
  for (auto stride : getStrides()) {
    auto dimExpr = getAffineDimExpr(numDims++, getContext());
    if (ShapedType::isDynamic(stride)) {
      auto symExpr = getAffineSymbolExpr(numSyms++, getContext());
      resultExpr = resultExpr + dimExpr * symExpr;
    } else {
      auto intExpr = getAffineConstantExpr(stride, getContext());
      resultExpr = resultExpr + dimExpr * intExpr;
    }
  }
  return AffineMap::get(numDims, numSyms, resultExpr);
}

Type KapyMemRefType::parse(AsmParser &parser) {
  if (failed(parser.parseLess()))
    return Type();
  SmallVector<int64_t, 4> shape;
  if (failed(parser.parseDimensionList(shape, false)))
    return Type();
  Type elementType;
  if (failed(parser.parseType(elementType)))
    return Type();
  Attribute layout;
  if (succeeded(parser.parseOptionalComma())) {
    if (parser.parseAttribute(layout))
      return Type();
  }
  if (failed(parser.parseGreater()))
    return Type();
  return KapyMemRefType::get(shape, elementType, layout);
}

void KapyMemRefType::print(AsmPrinter &printer) const {
  printer << "<";
  for (auto dim : getShape())
    printer << dim << "x";
  printer << getElementType();
  if (getEncoding())
    printer << ", " << getEncoding();
  printer << ">";
}

LogicalResult FpToFpOp::verify() {
  auto oldBitWidth = getIntOrFloatBitWidth(getOperand().getType());
  auto newBitWidth = getIntOrFloatBitWidth(getType());
  if (oldBitWidth > newBitWidth && !getRoundingMode())
    return emitOpError("rounding mode is required for floating-point downcast");
  return success();
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
    return emitOpError("start must less than end");
  auto shape = getType().getShape();
  if (shape.size() != 1)
    return emitOpError("result must be a 1d tensor");
  if (end - start != shape[0])
    return emitOpError("number of elements in result tensor is ")
           << shape[0] << ", doesn't match size of range [" << start << ", "
           << end << ")";
  return success();
}

LogicalResult MakeMemRefOp::verify() {
  auto type = getType();
  if (type.getRank() != getParentShape().size())
    return emitOpError(
        "result rank must same as number of operands for parent shape");
  if (type.getRank() != getStrides().size())
    return emitOpError(
        "result rank must same as number of operands for strides");
  auto glmemLayout = dyn_cast_or_null<GlobalMemLayoutAttr>(type.getEncoding());
  if (!glmemLayout)
    return emitOpError("result must have global memory layout");
  if (type.getRank() != glmemLayout.getStrides().size())
    return emitOpError("result rank must same as layout's strides size");
  return success();
}

LogicalResult MoveMemRefOp::canonicalize(MoveMemRefOp op,
                                         PatternRewriter &rewriter) {
  auto constantOp = op.getOffset().getDefiningOp<arith::ConstantOp>();
  if (!constantOp)
    return failure();
  auto intAttr = dyn_cast<IntegerAttr>(constantOp.getValue());
  if (!intAttr)
    return failure();
  if (intAttr.getInt() == 0) {
    op.replaceAllUsesWith(op.getSource());
    return success();
  }
  return failure();
}

LogicalResult MoveMemRefOp::verify() {
  if (!hasLayout<GlobalMemLayoutAttr>(getType()))
    return emitOpError("source must have global memory layout");
  return success();
}

void LoadOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>>
        &effects) {
  effects.emplace_back(MemoryEffects::Read::get(), getSource(),
                       GlobalMemory::get());
  if (getIsVolatile())
    effects.emplace_back(MemoryEffects::Write::get(),
                         SideEffects::DefaultResource::get());
}

LogicalResult LoadOp::verify() {
  auto sourceType = cast<KapyMemRefType>(getSource().getType());
  if (!hasLayout<GlobalMemLayoutAttr>(sourceType))
    return emitOpError("source must have global memory layout");
  auto resultType = getType();
  if (auto tensorType = dyn_cast<RankedTensorType>(resultType)) {
    if (sourceType.getShape() != tensorType.getShape())
      return emitOpError("source and result must have same shape");
    if (sourceType.getElementType() != tensorType.getElementType())
      return emitOpError("source and result must have same element type");
  } else {
    if (sourceType.getRank() != 0)
      return emitOpError("source must have rank 0 when result is scalar");
    if (sourceType.getElementType() != resultType)
      return emitOpError("source element type must same as result type");
  }
  return success();
}

LogicalResult StoreOp::verify() {
  auto targetType = cast<KapyMemRefType>(getTarget().getType());
  if (!hasLayout<GlobalMemLayoutAttr>(targetType))
    return emitOpError("target must have global memory layout");
  auto valueType = getValue().getType();
  if (auto tensorType = dyn_cast<RankedTensorType>(valueType)) {
    if (targetType.getShape() != tensorType.getShape())
      return emitOpError("target and result must have same shape");
    if (targetType.getElementType() != tensorType.getElementType())
      return emitOpError("target and result must have same element type");
  } else {
    if (targetType.getRank() != 0)
      return emitOpError("target must have rank 0 when value is scalar");
    if (targetType.getElementType() != valueType)
      return emitOpError("target element type must be same as value type");
  }
  return success();
}

LogicalResult AtomicRMWOp::verify() {
  auto sourceType = cast<KapyMemRefType>(getSource().getType());
  if (!hasLayout<GlobalMemLayoutAttr>(sourceType))
    return emitOpError("source must have global memory layout");
  auto valueType = getValue().getType();
  if (auto tensorType = dyn_cast<RankedTensorType>(valueType)) {
    if (sourceType.getShape() != tensorType.getShape())
      return emitOpError("source and value must have same shape");
    if (sourceType.getElementType() != tensorType.getElementType())
      return emitOpError("source and value must have same element type");
  } else {
    if (sourceType.getRank() != 0)
      return emitOpError("source must have rank 0 when value is scalar");
    if (sourceType.getElementType() != valueType)
      return emitOpError("source element type must be same as value type");
  }
  return success();
}

LogicalResult AtomicCASOp::verify() {
  auto sourceType = cast<KapyMemRefType>(getSource().getType());
  if (!hasLayout<GlobalMemLayoutAttr>(sourceType))
    return emitOpError("source must have global memory layout");
  auto valueType = getValue().getType();
  if (auto tensorType = dyn_cast<RankedTensorType>(valueType)) {
    if (sourceType.getShape() != tensorType.getShape())
      return emitOpError("source and value must have same shape");
    if (sourceType.getElementType() != tensorType.getElementType())
      return emitOpError("source and value must have same element type");
  } else {
    if (sourceType.getRank() != 0)
      return emitOpError("source must have rank 0 when value is scalar");
    if (sourceType.getElementType() != valueType)
      return emitOpError("source element type must be same as value type");
  }
  return success();
}

LogicalResult MatmulOp::verify() {
  if (getNumOperands() != 3)
    return emitOpError("expected 3 operands");

  auto lhsType = getLhs().getType();
  auto rhsType = getRhs().getType();
  auto accumType = getAccum().getType();

  if (lhsType.getElementTypeBitWidth() != rhsType.getElementTypeBitWidth())
    return emitOpError("lhs and rhs must have same element bit-width");

  auto lhsRank = lhsType.getRank();
  auto rhsRank = rhsType.getRank();
  auto accumRank = accumType.getRank();
  auto lhsShape = lhsType.getShape();
  auto rhsShape = rhsType.getShape();
  auto accumShape = accumType.getShape();

  if (lhsRank != 2 && lhsRank != 3)
    return emitOpError("operands must be 2d or 3d");
  if (lhsRank != rhsRank || lhsRank != accumRank)
    return emitOpError("requires all operands to have same rank");

  if (lhsShape[lhsRank - 1] != rhsShape[rhsRank - 2])
    return emitOpError("the last dimension of lhs must be equal to the second "
                       "to last dimension of rhs");

  if (lhsRank == 3 &&
      (lhsShape[0] != rhsShape[0] || lhsShape[0] != accumShape[0]))
    return emitOpError("the first dimension of all operands must be equal");

  if (accumShape[accumRank - 2] != lhsShape[lhsRank - 2] ||
      accumShape[accumRank - 1] != rhsShape[rhsRank - 1])
    return emitOpError(
        "result shape must be compatible with lhs and rhs shapes");

  auto lhsLayout = lhsType.getEncoding();
  auto rhsLayout = rhsType.getEncoding();
  if (!lhsLayout && !rhsLayout)
    return success();
  if (!lhsLayout || !rhsLayout)
    return emitOpError("lhs and rhs must both have or without layout");
  auto &dialect = lhsLayout.getDialect();
  auto *interface = cast<KapyLayoutInterface>(&dialect);
  return interface->verifyMatmulOpLayouts(*this);
}

ParseResult ReduceOp::parse(OpAsmParser &parser, OperationState &state) {
  OpAsmParser::UnresolvedOperand unresolvedOpd;
  Type operandType, type;
  if (parser.parseOperand(unresolvedOpd) ||
      parser.parseOptionalAttrDict(state.attributes))
    return failure();

  SmallVector<OpAsmParser::Argument, 2> bodyArgs;
  if (parser.parseKeyword("lambda") ||
      parser.parseArgumentList(bodyArgs, OpAsmParser::Delimiter::Paren, true))
    return failure();

  auto *region = state.addRegion();
  if (parser.parseRegion(*region, bodyArgs))
    return failure();

  if (parser.parseColonType(operandType) || parser.parseArrow() ||
      parser.parseType(type) ||
      parser.resolveOperand(unresolvedOpd, operandType, state.operands))
    return failure();
  state.addTypes(type);

  return success();
}

void ReduceOp::print(OpAsmPrinter &printer) {
  printer << " " << getOperand();
  printer.printOptionalAttrDict((*this)->getAttrs());
  printer << " lambda(";
  llvm::interleaveComma(getBody()->getArguments(), printer, [&](Value value) {
    printer << value << ": " << value.getType();
  });
  printer << ") ";
  printer.printRegion(getRegion(), false);
  printer << " : " << getOperand().getType() << " -> " << getType();
}

LogicalResult
ReduceOp::inferReturnTypes(MLIRContext *context, std::optional<Location> loc,
                           ValueRange operands, DictionaryAttr attrs,
                           OpaqueProperties props, RegionRange regions,
                           SmallVectorImpl<Type> &types) {
  auto axis = props.as<Properties *>()->getAxis().getInt();
  auto operandType = cast<RankedTensorType>(operands[0].getType());
  auto elementType = operandType.getElementType();
  auto shape = llvm::to_vector<4>(operandType.getShape());
  shape.erase(shape.begin() + axis);
  if (shape.empty()) {
    types.push_back(elementType);
    return success();
  }
  auto operandLayout = operandType.getEncoding();
  if (!operandLayout) {
    types.push_back(RankedTensorType::get(shape, elementType));
    return success();
  }
  auto &dialect = operandLayout.getDialect();
  auto *interface = cast<KapyLayoutInterface>(&dialect);
  auto inferLayout = interface->inferReduceOpLayout(operandLayout, axis, loc);
  if (failed(inferLayout))
    return failure();
  types.push_back(
      RankedTensorType::get(shape, elementType, inferLayout.value()));
  return success();
}

LogicalResult ReduceOp::verifyRegions() {
  auto elementType = getOperand().getType().getElementType();
  auto *body = getBody();
  if (body->getNumArguments() != 2)
    return emitOpError("block must take 2 arguments");

  const auto &bodyArgTypes = body->getArgumentTypes();
  for (unsigned i = 0; i < 2; ++i)
    if (bodyArgTypes[i] != elementType)
      return emitOpError("block argument ") << i << " type mismatch";

  auto yieldOp = dyn_cast<YieldOp>(body->getTerminator());
  if (!yieldOp)
    return emitOpError("block must be terminated with a ")
           << YieldOp::getOperationName() << ", but got "
           << body->getTerminator();
  if (yieldOp.getOperand().getType() != elementType)
    return emitOpError("block terminator type mismatch");

  return success();
}

OpFoldResult SplatOp::fold(FoldAdaptor adaptor) {
  if (auto operand = adaptor.getOperand())
    return SplatElementsAttr::get(getType(), operand);
  return OpFoldResult();
}

LogicalResult
UnsqueezeOp::inferReturnTypes(MLIRContext *context, std::optional<Location> loc,
                              ValueRange operands, DictionaryAttr attrs,
                              OpaqueProperties props, RegionRange regions,
                              SmallVectorImpl<Type> &types) {
  auto axis = props.as<Properties *>()->getAxis().getInt();
  auto operandType = cast<RankedTensorType>(operands[0].getType());
  auto elementType = operandType.getElementType();
  auto shape = llvm::to_vector<4>(operandType.getShape());
  shape.insert(shape.begin() + axis, 1);
  auto operandLayout = operandType.getEncoding();
  if (!operandLayout) {
    types.push_back(RankedTensorType::get(shape, elementType));
    return success();
  }
  auto &dialect = operandLayout.getDialect();
  auto *interface = cast<KapyLayoutInterface>(&dialect);
  auto inferLayout =
      interface->inferUnsqueezeOpLayout(operandLayout, axis, loc);
  if (failed(inferLayout))
    return failure();
  types.push_back(
      RankedTensorType::get(shape, elementType, inferLayout.value()));
  return success();
}

LogicalResult UnsqueezeOp::canonicalize(UnsqueezeOp op,
                                        PatternRewriter &rewriter) {
  auto defOp = op.getOperand().getDefiningOp();
  if (!defOp)
    return failure();
  if (auto splatOp = dyn_cast<SplatOp>(defOp)) {
    rewriter.replaceOpWithNewOp<SplatOp>(op, op.getType(),
                                         splatOp.getOperand());
    return success();
  }
  if (auto broadcastOp = dyn_cast<BroadcastOp>(defOp)) {
    auto tmpOp = rewriter.create<UnsqueezeOp>(
        op.getLoc(), broadcastOp.getOperand(), op.getAxis());
    auto newOp =
        rewriter.create<BroadcastOp>(broadcastOp.getLoc(), op.getType(), tmpOp);
    rewriter.replaceOp(op, newOp);
    return success();
  }
  return failure();
}

OpFoldResult UnsqueezeOp::fold(FoldAdaptor adaptor) {
  auto operand = adaptor.getOperand();
  if (auto splatAttr = dyn_cast_or_null<SplatElementsAttr>(operand))
    return splatAttr.resizeSplat(cast<ShapedType>(getType()));
  if (auto denseAttr = dyn_cast_or_null<DenseElementsAttr>(operand))
    return denseAttr.reshape(cast<ShapedType>(getType()));
  return OpFoldResult();
}

LogicalResult UnsqueezeOp::verify() {
  auto axis = getAxis();
  if (axis < 0 || axis > getOperand().getType().getRank())
    return emitOpError("invalid axis");
  return success();
}

LogicalResult BroadcastOp::canonicalize(BroadcastOp op,
                                        PatternRewriter &rewriter) {
  if (op.getType() == op.getOperand().getType()) {
    op.replaceAllUsesWith(op.getOperand());
    return success();
  }
  auto *defOp = op.getOperand().getDefiningOp();
  if (!defOp)
    return failure();
  if (auto splatOp = dyn_cast<SplatOp>(defOp)) {
    rewriter.replaceOpWithNewOp<SplatOp>(op, op.getType(),
                                         splatOp.getOperand());
    return success();
  }
  if (auto broadcastOp = dyn_cast<BroadcastOp>(defOp)) {
    rewriter.replaceOpWithNewOp<BroadcastOp>(op, op.getType(),
                                             broadcastOp.getOperand());
    return success();
  }
  return failure();
}

OpFoldResult BroadcastOp::fold(FoldAdaptor adaptor) {
  auto operand = adaptor.getOperand();
  if (auto splatAttr = dyn_cast_or_null<SplatElementsAttr>(operand))
    return splatAttr.resizeSplat(cast<ShapedType>(getType()));
  return OpFoldResult();
}

LogicalResult BroadcastOp::verify() {
  auto oldShape = getOperand().getType().getShape();
  auto newShape = getType().getShape();
  if (oldShape.size() != newShape.size())
    return emitOpError("operand and result must have same rank");
  for (unsigned i = 0; i < oldShape.size(); ++i) {
    if (oldShape[i] == 1)
      continue;
    if (oldShape[i] != newShape[i])
      return emitOpError(
          "non-broadcast dimension of operand and result must be same");
  }
  return success();
}

LogicalResult
PermuteOp::inferReturnTypes(MLIRContext *context, std::optional<Location> loc,
                            ValueRange operands, DictionaryAttr attrs,
                            OpaqueProperties props, RegionRange regions,
                            SmallVectorImpl<Type> &types) {
  auto order = props.as<Properties *>()->getOrder().asArrayRef();
  auto operandType = cast<RankedTensorType>(operands[0].getType());
  auto elementType = operandType.getElementType();
  auto shape = permute(operandType.getShape(), order);
  auto operandLayout = operandType.getEncoding();
  if (!operandLayout) {
    types.push_back(RankedTensorType::get(shape, elementType));
    return success();
  }
  auto &dialect = operandType.getEncoding().getDialect();
  auto *interface = cast<KapyLayoutInterface>(&dialect);
  auto inferLayout = interface->inferPermuteOpLayout(operandLayout, order, loc);
  if (failed(inferLayout))
    return failure();
  types.push_back(
      RankedTensorType::get(shape, elementType, inferLayout.value()));
  return success();
}

LogicalResult PermuteOp::canonicalize(PermuteOp op, PatternRewriter &rewriter) {
  if (isIota(op.getOrder())) {
    op.replaceAllUsesWith(op.getOperand());
    return success();
  }
  auto defOp = op.getOperand().getDefiningOp();
  if (!defOp)
    return failure();
  if (auto splatOp = dyn_cast<SplatOp>(defOp)) {
    rewriter.replaceOpWithNewOp<SplatOp>(op, op.getType(),
                                         splatOp.getOperand());
    return success();
  }
  if (auto permuteOp = dyn_cast<PermuteOp>(defOp)) {
    auto order = permute(permuteOp.getOrder(), op.getOrder());
    rewriter.replaceOpWithNewOp<PermuteOp>(op, permuteOp.getOperand(), order);
    return success();
  }
  return failure();
}

OpFoldResult PermuteOp::fold(FoldAdaptor adaptor) {
  auto operand = adaptor.getOperand();
  if (auto splatAttr = dyn_cast_or_null<SplatElementsAttr>(operand))
    return splatAttr.resizeSplat(cast<ShapedType>(getType()));
  return OpFoldResult();
}

LogicalResult PermuteOp::verify() {
  auto order = llvm::to_vector<4>(getOrder());
  std::stable_sort(order.begin(), order.end(),
                   [](auto a, auto b) { return a < b; });
  if (!isIota(order))
    return emitOpError("invalid order");
  return success();
}

void ElementwiseExternOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>>
        &effects) {
  if (getIsPure())
    return;
  effects.emplace_back(MemoryEffects::Read::get(),
                       SideEffects::DefaultResource::get());
  effects.emplace_back(MemoryEffects::Write::get(),
                       SideEffects::DefaultResource::get());
}

void ElementwiseInlineOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>>
        &effects) {
  if (getIsPure())
    return;
  effects.emplace_back(MemoryEffects::Read::get(),
                       SideEffects::DefaultResource::get());
  effects.emplace_back(MemoryEffects::Write::get(),
                       SideEffects::DefaultResource::get());
}

LogicalResult ElementwiseInlineOp::verify() {
  if (getNumOperands() >= 1) {
    auto tensorType = dyn_cast<RankedTensorType>(getOperand(0).getType());
    auto numElements = tensorType ? tensorType.getNumElements() : 0;
    if (numElements % this->getNumPackedValues() != 0)
      return emitOpError("number of elements")
             << numElements
             << "must be a multiple of the attribute num_packed_values = "
             << getNumPackedValues();
  }
  return success();
}

void FuncOp::build(OpBuilder &builder, OperationState &state, StringRef name,
                   FunctionType type, ArrayRef<NamedAttribute> attrs,
                   ArrayRef<DictionaryAttr> argAttrs) {
  state.addAttribute(SymbolTable::getSymbolAttrName(),
                     builder.getStringAttr(name));
  state.addAttribute(getFunctionTypeAttrName(state.name), TypeAttr::get(type));
  state.attributes.append(attrs.begin(), attrs.end());
  state.addRegion();

  if (argAttrs.empty())
    return;
  assert(type.getNumInputs() == argAttrs.size());
  function_interface_impl::addArgAndResultAttrs(
      builder, state, argAttrs, {}, getArgAttrsAttrName(state.name),
      getResAttrsAttrName(state.name));
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
  auto funcOp = cast<FuncOp>((*this)->getParentOp());
  auto results = funcOp.getFunctionType().getResults();
  if (getNumOperands() != results.size())
    return emitOpError("has incorrect number of operands");
  for (unsigned i = 0; i < results.size(); ++i)
    if (getOperand(i).getType() != results[i])
      return emitOpError("operand ") << i << " type mismatch";
  return success();
}

unsigned kapy::getIntOrFloatBitWidth(Type type) {
  if (auto tensorType = dyn_cast<RankedTensorType>(type))
    return tensorType.getElementTypeBitWidth();
  if (auto memrefType = dyn_cast<KapyMemRefType>(type))
    return memrefType.getElementTypeBitWidth();
  return type.getIntOrFloatBitWidth();
}

RankedTensorType kapy::cloneWith(RankedTensorType type, Type elementType) {
  return RankedTensorType::get(type.getShape(), elementType,
                               type.getEncoding());
}

RankedTensorType kapy::cloneWith(RankedTensorType type, Attribute layout) {
  return RankedTensorType::get(type.getShape(), type.getElementType(), layout);
}

KapyMemRefType kapy::cloneWith(KapyMemRefType type, Type elementType) {
  return KapyMemRefType::get(type.getShape(), elementType, type.getEncoding());
}

KapyMemRefType kapy::cloneWith(KapyMemRefType type, Attribute layout) {
  return KapyMemRefType::get(type.getShape(), type.getElementType(), layout);
}
