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
    if (isa<Strided2dLayoutAttr>(attr)) {
      os << "strided2d";
      return AliasResult::FinalAlias;
    }
    return OpAsmDialectInterface::getAlias(attr, os);
  }
};

class KapyInlinerInterface : public DialectInlinerInterface {
public:
  using DialectInlinerInterface::DialectInlinerInterface;

  static constexpr char inlineAttrName[] = "inline";

  virtual bool isLegalToInline(Operation *caller, Operation *callee,
                               bool wouldBeCloned) const override {
    auto funcOp = dyn_cast<FuncOp>(callee);
    if (!funcOp)
      return true;
    if (funcOp->hasAttr(inlineAttrName))
      return funcOp->getAttrOfType<BoolAttr>(inlineAttrName).getValue();
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

Attribute Strided2dLayoutAttr::parse(AsmParser &parser, Type type) {
  if (failed(parser.parseLess()))
    return Attribute();

  SmallVector<int64_t, 2> strides;
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

  assert(strides.size() == 2);
  return Strided2dLayoutAttr::get(parser.getContext(), strides[0], strides[1]);
}

void Strided2dLayoutAttr::print(AsmPrinter &printer) const {
  auto printQuestionOrInt = [&](int64_t value) {
    if (ShapedType::isDynamic(value))
      printer << "?";
    else
      printer << value;
  };

  printer << "<[";
  llvm::interleaveComma(getStrides(), printer, printQuestionOrInt);
  printer << "]>";
}

AffineMap Strided2dLayoutAttr::getAffineMap() const {
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

Type GlobalMemRefType::parse(AsmParser &parser) {
  if (failed(parser.parseLess()))
    return Type();
  SmallVector<int64_t, 2> shape;
  if (failed(parser.parseDimensionList(shape)))
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
  return GlobalMemRefType::get(shape, elementType, layout);
}

void GlobalMemRefType::print(AsmPrinter &printer) const {
  printer << "<";
  for (auto size : getShape()) {
    if (ShapedType::isDynamic(size))
      printer << "?";
    else
      printer << size;
    printer << "x";
  }
  printer << getElementType();
  if (getEncoding())
    printer << ", " << getEncoding();
  printer << ">";
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

void LoadGlobalOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>>
        &effects) {
  effects.emplace_back(MemoryEffects::Read::get(), &getSourceMutable(),
                       GlobalMemory::get());
  if (getIsVolatile())
    effects.emplace_back(MemoryEffects::Write::get(),
                         SideEffects::DefaultResource::get());
}

void StoreGlobalOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>>
        &effects) {
  effects.emplace_back(MemoryEffects::Write::get(), &getTargetMutable(),
                       GlobalMemory::get());
}

void AtomicRMWGlobalOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>>
        &effects) {
  effects.emplace_back(MemoryEffects::Read::get(), &getSourceMutable(),
                       GlobalMemory::get());
  effects.emplace_back(MemoryEffects::Write::get(), &getSourceMutable(),
                       GlobalMemory::get());
}

LogicalResult MatmulOp::verify() {
  auto lhsType = getLhs().getType();
  auto rhsType = getRhs().getType();
  auto accType = getAcc().getType();

  if (lhsType.getElementTypeBitWidth() != rhsType.getElementTypeBitWidth())
    return emitOpError("lhs and rhs must have same element bit width");

  auto implWay = getMatmulImplWay();
  auto lhsElementType = lhsType.getElementType();
  auto rhsElementType = rhsType.getElementType();
  auto accElementType = accType.getElementType();

  if (!accElementType.isF16() && !accElementType.isF32())
    return emitOpError("result must have f16 or f32 element type");

  if (implWay == MatmulImplWay::FMA) {
    if (lhsElementType != accElementType || rhsElementType != accElementType)
      return emitOpError("fma operands and result must have same element type");
  } else if (implWay == MatmulImplWay::MMA_M16N8K8_F16) {
    if (!lhsElementType.isF16() && !lhsElementType.isBF16())
      return emitOpError(
          "mma_m16n8k8_f16 operands must have f16 or bf16 element type");
    if (!rhsElementType.isF16() && !rhsElementType.isBF16())
      return emitOpError(
          "mma_m16n8k8_f16 operands must have f16 or bf16 element type");
  } else if (implWay == MatmulImplWay::MMA_M16N8K16_F16) {
    if (!lhsElementType.isF16() && !lhsElementType.isBF16())
      return emitOpError(
          "mma_m16n8k16_f16 operands must have f16 or bf16 element type");
    if (!rhsElementType.isF16() && !rhsElementType.isBF16())
      return emitOpError(
          "mma_m16n8k16_f16 operands must have f16 or bf16 element type");
  } else if (implWay == MatmulImplWay::MMA_M16N8K8_TF32) {
    if (!lhsElementType.isF32() || !rhsElementType.isF32())
      return emitOpError(
          "mma_m16n8k8_tf32 operands must have f32 element type");
  } else if (implWay == MatmulImplWay::MMA_M16N8K16_F8) {
    if (!lhsElementType.isFloat8E4M3() && !lhsElementType.isFloat8E5M2())
      return emitOpError(
          "mma_m16n8k16_f8 operands must have f8E4M3 or f8E5M2 element type");
    if (!rhsElementType.isFloat8E4M3() && !rhsElementType.isFloat8E5M2())
      return emitOpError(
          "mma_m16n8k16_f8 operands must have f8E4M3 or f8E5M2 element type");
  }

  auto lhsShape = lhsType.getShape();
  auto rhsShape = rhsType.getShape();
  auto accShape = accType.getShape();
  if (lhsShape[1] != rhsShape[0])
    return emitOpError("the number of columns of lhs must be equal to the "
                       "number of rows of rhs");
  if (accShape[0] != lhsShape[0])
    return emitOpError(
        "the number of rows of acc must be equal to the number of rows of lhs");
  if (accShape[1] != rhsShape[1])
    return emitOpError("the number of columns of acc must be equal to the "
                       "number of columns of rhs");

  auto lhsLayout = lhsType.getEncoding();
  auto rhsLayout = rhsType.getEncoding();
  if (!lhsLayout && !rhsLayout)
    return success();
  if (!lhsLayout || !rhsLayout)
    return emitOpError("lhs and rhs must both have or without layout");
  auto *interface = cast<KapyLayoutInterface>(&lhsLayout.getDialect());
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
  returnTypes.push_back(RankedTensorType::get(
      shape, sourceType.getElementType(), sourceType.getEncoding()));
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
  if (op.getType() == op.getSource().getType()) {
    op.replaceAllUsesWith(op.getSource());
    return success();
  }
  auto *defOp = op.getSource().getDefiningOp();
  if (!defOp)
    return failure();
  if (auto splatOp = dyn_cast<SplatOp>(defOp)) {
    rewriter.replaceOpWithNewOp<SplatOp>(op, op.getType(), splatOp.getSource());
    return success();
  }
  if (auto broadcastOp = dyn_cast<BroadcastOp>(defOp)) {
    rewriter.replaceOpWithNewOp<BroadcastOp>(op, op.getType(),
                                             broadcastOp.getSource());
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
  auto sourceLayout = sourceType.getEncoding();
  if (!sourceLayout) {
    returnTypes.push_back(
        RankedTensorType::get(shape, sourceType.getElementType()));
    return success();
  }
  auto *interface = cast<KapyLayoutInterface>(&sourceLayout.getDialect());
  auto resultLayout = interface->inferTransposeOpLayout(sourceLayout);
  returnTypes.push_back(
      RankedTensorType::get(shape, sourceType.getElementType(), resultLayout));
  return success();
}

LogicalResult TransposeOp::canonicalize(TransposeOp op,
                                        PatternRewriter &rewriter) {
  auto *defOp = op.getSource().getDefiningOp();
  if (!defOp)
    return failure();
  if (auto transposeOp = dyn_cast<TransposeOp>(defOp)) {
    op.replaceAllUsesWith(transposeOp.getSource());
    return success();
  }
  if (auto splatOp = dyn_cast<SplatOp>(defOp)) {
    rewriter.replaceOpWithNewOp<SplatOp>(op, op.getType(), splatOp.getSource());
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

bool kapy::isGlobalMemoryRead(Operation *op) {
  auto effectOp = dyn_cast<MemoryEffectOpInterface>(op);
  if (!effectOp)
    return false;
  SmallVector<SideEffects::EffectInstance<MemoryEffects::Effect>> effects;
  effectOp.getEffects(effects);
  for (auto &effect : effects)
    if (isa<MemoryEffects::Read>(effect.getEffect()) &&
        effect.getResource() == GlobalMemory::get())
      return true;
  return false;
}

bool kapy::isGlobalMemoryWrite(Operation *op) {
  auto effectOp = dyn_cast<MemoryEffectOpInterface>(op);
  if (!effectOp)
    return false;
  SmallVector<SideEffects::EffectInstance<MemoryEffects::Effect>> effects;
  effectOp.getEffects(effects);
  for (auto &effect : effects)
    if (isa<MemoryEffects::Write>(effect.getEffect()) &&
        effect.getResource() == GlobalMemory::get())
      return true;
  return false;
}

unsigned kapy::getAlignment(Operation *op) {
  if (!op->hasAttr(alignmentAttrName))
    llvm_unreachable("can not get a named attribute");
  return cast<IntegerAttr>(op->getAttr(alignmentAttrName)).getInt();
}
