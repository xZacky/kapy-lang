//===- ConversionPass.cpp ---------------------------------------*- C++ -*-===//
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

#include "kapy/Analysis/LayoutUtils.h"
#include "kapy/Conversion/KapyToKgpu/Passes.h"
#include "kapy/Conversion/KapyToKgpu/TypeConverter.h"
#include "kapy/Dialect/Kapy/IR/Kapy.h"
#include "kapy/Dialect/Kgpu/IR/Kgpu.h"

using namespace mlir;
using namespace mlir::kapy;

/// Pass named attributes (e.g. kapy.alignment).
static void addNamedAttributes(Operation *op, DictionaryAttr attrs) {
  for (auto it : attrs)
    if (!op->hasAttr(it.getName()))
      op->setAttr(it.getName(), it.getValue());
}

namespace {

template <typename OpT>
class GenericOpConversion : public OpConversionPattern<OpT> {
public:
  using OpConversionPattern<OpT>::OpConversionPattern;
  using OpAdaptor = typename OpT::Adaptor;

  virtual LogicalResult
  matchAndRewrite(OpT op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    SmallVector<Type> newTypes;
    const TypeConverter *typeConverter = this->getTypeConverter();
    if (failed(typeConverter->convertTypes(op->getResultTypes(), newTypes)))
      return failure();
    rewriter.replaceOpWithNewOp<OpT>(op, newTypes, adaptor.getOperands(),
                                     op->getAttrs());
    return success();
  }
};

class ConstantOpConversion : public OpConversionPattern<arith::ConstantOp> {
public:
  using OpConversionPattern::OpConversionPattern;

  virtual LogicalResult
  matchAndRewrite(arith::ConstantOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto newType =
        cast<RankedTensorType>(typeConverter->convertType(op.getType()));
    auto value = cast<DenseElementsAttr>(adaptor.getValue());
    // This is a hack. We just want to add layout.
    value = value.reshape(newType);
    auto newOp =
        rewriter.replaceOpWithNewOp<arith::ConstantOp>(op, newType, value);
    addNamedAttributes(newOp, adaptor.getAttributes());
    return success();
  }
};

class BroadcastOpConversion : public OpConversionPattern<BroadcastOp> {
public:
  using OpConversionPattern::OpConversionPattern;

  virtual LogicalResult
  matchAndRewrite(BroadcastOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto source = adaptor.getSource();
    auto sourceType = cast<RankedTensorType>(source.getType());
    auto newType = RankedTensorType::get(op.getType().getShape(),
                                         sourceType.getElementType(),
                                         sourceType.getEncoding());
    auto newOp = rewriter.replaceOpWithNewOp<BroadcastOp>(op, newType, source);
    addNamedAttributes(newOp, adaptor.getAttributes());
    return success();
  }
};

class TransposeOpConversion : public OpConversionPattern<TransposeOp> {
public:
  using OpConversionPattern::OpConversionPattern;

  virtual LogicalResult
  matchAndRewrite(TransposeOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto newOp =
        rewriter.replaceOpWithNewOp<TransposeOp>(op, adaptor.getSource());
    addNamedAttributes(newOp, adaptor.getAttributes());
    return success();
  }
};

class MatmulOpConversion : public OpConversionPattern<MatmulOp> {
public:
  using OpConversionPattern::OpConversionPattern;

  virtual LogicalResult
  matchAndRewrite(MatmulOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto *context = op.getContext();
    auto [lhsLayout, rhsLayout, accLayout] = getOperandLayouts(op);

    auto acc = adaptor.getAcc();
    auto accType = cast<RankedTensorType>(acc.getType());
    accType = RankedTensorType::get(
        accType.getShape(), accType.getElementType(),
        EncodingAttr::get(context, MemorySpace::REGISTER_FILE, accLayout));
    acc = rewriter.create<ChangeOp>(acc.getLoc(), accType, acc);

    auto lhs = adaptor.getLhs();
    auto lhsType = cast<RankedTensorType>(lhs.getType());
    lhsType = RankedTensorType::get(
        lhsType.getShape(), lhsType.getElementType(),
        EncodingAttr::get(context, MemorySpace::REGISTER_FILE, lhsLayout));
    lhs = rewriter.create<ChangeOp>(lhs.getLoc(), lhsType, lhs);

    auto rhs = adaptor.getRhs();
    auto rhsType = cast<RankedTensorType>(rhs.getType());
    rhsType = RankedTensorType::get(
        rhsType.getShape(), rhsType.getElementType(),
        EncodingAttr::get(context, MemorySpace::REGISTER_FILE, rhsLayout));
    rhs = rewriter.create<ChangeOp>(rhs.getLoc(), rhsType, rhs);

    auto newOp = rewriter.replaceOpWithNewOp<MatmulOp>(op, lhs, rhs, acc,
                                                       op.getMatmulImplWay());
    addNamedAttributes(newOp, adaptor.getAttributes());
    return success();
  }
};

class ReduceOpConversion : public OpConversionPattern<ReduceOp> {
public:
  using OpConversionPattern::OpConversionPattern;

  virtual LogicalResult
  matchAndRewrite(ReduceOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto newOp = rewriter.create<ReduceOp>(op.getLoc(), adaptor.getSource(),
                                           op.getAxis());
    addNamedAttributes(newOp, adaptor.getAttributes());
    auto &newRegion = newOp.getRegion();
    rewriter.inlineRegionBefore(op.getRegion(), newRegion, newRegion.end());
    rewriter.replaceOp(op, newOp.getResult());
    return success();
  }
};

class FuncOpConversion : public OpConversionPattern<FuncOp> {
public:
  using OpConversionPattern::OpConversionPattern;

  virtual LogicalResult
  matchAndRewrite(FuncOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto newOp = rewriter.replaceOpWithNewOp<FuncOp>(op, op.getName(),
                                                     op.getFunctionType());
    addNamedAttributes(newOp, adaptor.getAttributes());
    auto &newBody = newOp.getBody();
    rewriter.inlineRegionBefore(op.getBody(), newBody, newBody.end());
    return rewriter.convertRegionTypes(&newBody, *typeConverter);
  }
};

class ForOpConversion : public OpConversionPattern<scf::ForOp> {
public:
  using OpConversionPattern::OpConversionPattern;

  virtual LogicalResult
  matchAndRewrite(scf::ForOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto newOp =
        cast<scf::ForOp>(rewriter.cloneWithoutRegions(*op.getOperation()));
    addNamedAttributes(newOp, adaptor.getAttributes());
    auto &newRegion = newOp.getRegion();
    rewriter.inlineRegionBefore(op.getRegion(), newRegion, newRegion.end());
    if (failed(rewriter.convertRegionTypes(&newRegion, *typeConverter)))
      return rewriter.notifyMatchFailure(op, "could not convert body types");

    // Change the clone to use the updated operands. We could have cloned with a
    // IRMapping, but this seems a bit more direct.
    newOp->setOperands(adaptor.getOperands());

    // Update the result types to the new converted types.
    SmallVector<Type> newTypes;
    for (auto type : op.getResultTypes()) {
      auto newType = typeConverter->convertType(type);
      if (!newType)
        return rewriter.notifyMatchFailure(op, "not a 1:1 type conversion");
      newTypes.push_back(newType);
    }
    for (auto [newResult, newType] : zip(newOp.getResults(), newTypes))
      newResult.setType(newType);

    rewriter.replaceOp(op, newOp.getResults());
    return success();
  }
};

class IfOpConversion : public OpConversionPattern<scf::IfOp> {
public:
  using OpConversionPattern::OpConversionPattern;

  virtual LogicalResult
  matchAndRewrite(scf::IfOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto newOp =
        cast<scf::IfOp>(rewriter.cloneWithoutRegions(*op.getOperation()));
    addNamedAttributes(newOp, adaptor.getAttributes());
    auto &newThen = newOp.getThenRegion();
    rewriter.inlineRegionBefore(op.getThenRegion(), newThen, newThen.end());
    auto &newElse = newOp.getElseRegion();
    rewriter.inlineRegionBefore(op.getElseRegion(), newElse, newElse.end());

    newOp->setOperands(adaptor.getOperands());

    SmallVector<Type> newTypes;
    for (auto type : op.getResultTypes()) {
      auto newType = typeConverter->convertType(type);
      if (!newType)
        return rewriter.notifyMatchFailure(op, "not a 1:1 type conversion");
      newTypes.push_back(newType);
    }
    for (auto [newResult, newType] : zip(newOp.getResults(), newTypes))
      newResult.setType(newType);

    rewriter.replaceOp(op, newOp.getResults());
    return success();
  }
};

class WhileOpConversion : public OpConversionPattern<scf::WhileOp> {
public:
  using OpConversionPattern::OpConversionPattern;

  virtual LogicalResult
  matchAndRewrite(scf::WhileOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    SmallVector<Type> newTypes;
    if (failed(typeConverter->convertTypes(op.getResultTypes(), newTypes)))
      return failure();
    auto newOp = rewriter.create<scf::WhileOp>(op.getLoc(), newTypes,
                                               adaptor.getOperands());
    addNamedAttributes(newOp, adaptor.getAttributes());
    for (unsigned i : {0, 1}) {
      auto &newRegion = newOp.getRegion(i);
      rewriter.inlineRegionBefore(op.getRegion(i), newRegion, newRegion.end());
      if (failed(rewriter.convertRegionTypes(&newRegion, *typeConverter)))
        return rewriter.notifyMatchFailure(op, "could not convert body types");
    }
    rewriter.replaceOp(op, newOp.getResults());
    return success();
  }
};

class ConditionOpConversion : public OpConversionPattern<scf::ConditionOp> {
public:
  using OpConversionPattern::OpConversionPattern;

  virtual LogicalResult
  matchAndRewrite(scf::ConditionOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto operands = adaptor.getOperands();
    rewriter.modifyOpInPlace(op, [&]() { op->setOperands(operands); });
    return success();
  }
};

class BranchOpConversion : public OpConversionPattern<cf::BranchOp> {
public:
  using OpConversionPattern::OpConversionPattern;

  virtual LogicalResult
  matchAndRewrite(cf::BranchOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto newOp = rewriter.replaceOpWithNewOp<cf::BranchOp>(
        op, op.getSuccessor(), adaptor.getOperands());
    addNamedAttributes(newOp, adaptor.getAttributes());
    auto *newRegion = newOp.getSuccessor()->getParent();
    return rewriter.convertRegionTypes(newRegion, *typeConverter);
  }
};

class CondBranchOpConversion : public OpConversionPattern<cf::CondBranchOp> {
public:
  using OpConversionPattern::OpConversionPattern;

  virtual LogicalResult
  matchAndRewrite(cf::CondBranchOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto newOp = rewriter.replaceOpWithNewOp<cf::CondBranchOp>(
        op, adaptor.getCondition(), op.getTrueDest(),
        adaptor.getTrueDestOperands(), op.getFalseDest(),
        adaptor.getFalseDestOperands());
    addNamedAttributes(newOp, adaptor.getAttributes());
    auto *trueRegion = newOp.getTrueDest()->getParent();
    if (failed(rewriter.convertRegionTypes(trueRegion, *typeConverter)))
      return failure();
    auto *falseRegion = newOp.getFalseDest()->getParent();
    if (failed(rewriter.convertRegionTypes(falseRegion, *typeConverter)))
      return failure();
    return success();
  }
};

} // namespace

static void populateArithConversionPatterns(KgpuTypeConverter &typeConverter,
                                            RewritePatternSet &patterns) {
  auto *context = patterns.getContext();
  patterns.add<ConstantOpConversion>(typeConverter, context);
  patterns.add<GenericOpConversion<arith::AddIOp>,
               GenericOpConversion<arith::AddFOp>>(typeConverter, context);
  patterns.add<GenericOpConversion<arith::SubIOp>,
               GenericOpConversion<arith::SubFOp>>(typeConverter, context);
  patterns.add<GenericOpConversion<arith::MulIOp>,
               GenericOpConversion<arith::MulFOp>>(typeConverter, context);
  patterns.add<GenericOpConversion<arith::DivUIOp>,
               GenericOpConversion<arith::DivSIOp>,
               GenericOpConversion<arith::DivFOp>>(typeConverter, context);
  patterns.add<GenericOpConversion<arith::FloorDivSIOp>>(typeConverter,
                                                         context);
  patterns.add<GenericOpConversion<arith::CeilDivUIOp>,
               GenericOpConversion<arith::CeilDivSIOp>>(typeConverter, context);
  patterns.add<GenericOpConversion<arith::RemUIOp>,
               GenericOpConversion<arith::RemSIOp>,
               GenericOpConversion<arith::RemFOp>>(typeConverter, context);
  patterns.add<GenericOpConversion<arith::AndIOp>,
               GenericOpConversion<arith::OrIOp>,
               GenericOpConversion<arith::XOrIOp>>(typeConverter, context);
  patterns.add<GenericOpConversion<arith::ShLIOp>>(typeConverter, context);
  patterns.add<GenericOpConversion<arith::ShRUIOp>,
               GenericOpConversion<arith::ShRSIOp>>(typeConverter, context);
  patterns.add<GenericOpConversion<arith::MaxUIOp>,
               GenericOpConversion<arith::MaxSIOp>>(typeConverter, context);
  patterns.add<GenericOpConversion<arith::MaximumFOp>,
               GenericOpConversion<arith::MaxNumFOp>>(typeConverter, context);
  patterns.add<GenericOpConversion<arith::MinUIOp>,
               GenericOpConversion<arith::MinSIOp>>(typeConverter, context);
  patterns.add<GenericOpConversion<arith::MinimumFOp>,
               GenericOpConversion<arith::MinNumFOp>>(typeConverter, context);
  patterns.add<GenericOpConversion<arith::CmpIOp>,
               GenericOpConversion<arith::CmpFOp>>(typeConverter, context);
  patterns.add<GenericOpConversion<arith::SelectOp>>(typeConverter, context);
  patterns.add<GenericOpConversion<arith::TruncIOp>,
               GenericOpConversion<arith::TruncFOp>>(typeConverter, context);
  patterns.add<GenericOpConversion<arith::ExtUIOp>,
               GenericOpConversion<arith::ExtSIOp>,
               GenericOpConversion<arith::ExtFOp>>(typeConverter, context);
  patterns.add<GenericOpConversion<arith::UIToFPOp>,
               GenericOpConversion<arith::FPToUIOp>>(typeConverter, context);
  patterns.add<GenericOpConversion<arith::SIToFPOp>,
               GenericOpConversion<arith::FPToSIOp>>(typeConverter, context);
  patterns.add<GenericOpConversion<arith::BitcastOp>>(typeConverter, context);
}

static void populateMathConversionPatterns(KgpuTypeConverter &typeConverter,
                                           RewritePatternSet &patterns) {
  auto *context = patterns.getContext();
  patterns.add<GenericOpConversion<math::ExpOp>, //
               GenericOpConversion<math::Exp2Op>>(typeConverter, context);
  patterns.add<GenericOpConversion<math::FloorOp>,
               GenericOpConversion<math::CeilOp>>(typeConverter, context);
  patterns.add<GenericOpConversion<math::SinOp>, //
               GenericOpConversion<math::CosOp>>(typeConverter, context);
  patterns.add<GenericOpConversion<math::LogOp>, //
               GenericOpConversion<math::Log2Op>>(typeConverter, context);
  patterns.add<GenericOpConversion<math::ErfOp>>(typeConverter, context);
  patterns.add<GenericOpConversion<math::AbsIOp>,
               GenericOpConversion<math::AbsFOp>>(typeConverter, context);
  patterns.add<GenericOpConversion<math::SqrtOp>,
               GenericOpConversion<math::RsqrtOp>>(typeConverter, context);
  patterns.add<GenericOpConversion<math::FmaOp>>(typeConverter, context);
}

static void populateKapyConversionPatterns(KgpuTypeConverter &typeConverter,
                                           RewritePatternSet &patterns) {
  auto *context = patterns.getContext();
  patterns.add<GenericOpConversion<FPToFPOp>>(typeConverter, context);
  patterns.add<GenericOpConversion<ClampFOp>>(typeConverter, context);
  patterns.add<GenericOpConversion<MkGlobalOp>>(typeConverter, context);
  patterns.add<GenericOpConversion<SvGlobalOp>>(typeConverter, context);
  patterns.add<GenericOpConversion<LdGlobalOp>>(typeConverter, context);
  patterns.add<GenericOpConversion<StGlobalOp>>(typeConverter, context);
  patterns.add<GenericOpConversion<AtomicRMWGlobalOp>>(typeConverter, context);
  patterns.add<GenericOpConversion<MkSharedOp>>(typeConverter, context);
  patterns.add<GenericOpConversion<SvSharedOp>>(typeConverter, context);
  patterns.add<GenericOpConversion<LdSharedOp>>(typeConverter, context);
  patterns.add<GenericOpConversion<StSharedOp>>(typeConverter, context);
  patterns.add<GenericOpConversion<CpAsyncGlobalToSharedOp>>(typeConverter,
                                                             context);
  patterns.add<GenericOpConversion<SplatOp>>(typeConverter, context);
  patterns.add<BroadcastOpConversion>(typeConverter, context);
  patterns.add<TransposeOpConversion>(typeConverter, context);
  patterns.add<GenericOpConversion<ArangeOp>>(typeConverter, context);
  patterns.add<MatmulOpConversion>(typeConverter, context);
  patterns.add<ReduceOpConversion>(typeConverter, context);
  patterns.add<GenericOpConversion<ElementwiseExternLibOp>>(typeConverter,
                                                            context);
  patterns.add<GenericOpConversion<ElementwiseInlineAsmOp>>(typeConverter,
                                                            context);
  patterns.add<GenericOpConversion<CallOp>>(typeConverter, context);
  patterns.add<FuncOpConversion>(typeConverter, context);
  patterns.add<GenericOpConversion<ReturnOp>>(typeConverter, context);
}

static void populateSCFConversionPatterns(KgpuTypeConverter &typeConverter,
                                          RewritePatternSet &patterns) {
  auto *context = patterns.getContext();
  patterns.add<ForOpConversion>(typeConverter, context);
  patterns.add<IfOpConversion>(typeConverter, context);
  patterns.add<WhileOpConversion>(typeConverter, context);
  patterns.add<ConditionOpConversion>(typeConverter, context);
  patterns.add<GenericOpConversion<scf::YieldOp>>(typeConverter, context);
}

static void populateCFConversionPatterns(KgpuTypeConverter &typeConverter,
                                         RewritePatternSet &patterns) {
  auto *context = patterns.getContext();
  patterns.add<BranchOpConversion>(typeConverter, context);
  patterns.add<CondBranchOpConversion>(typeConverter, context);
}

namespace {

class KgpuConversionTarget : public ConversionTarget {
public:
  explicit KgpuConversionTarget(MLIRContext *context,
                                KgpuTypeConverter &typeConverter)
      : ConversionTarget(*context) {
    addLegalDialect<KgpuDialect>();

    addIllegalOp<scf::ExecuteRegionOp, scf::ForallOp, scf::InParallelOp,
                 scf::IndexSwitchOp, scf::ParallelOp, scf::ReduceOp,
                 scf::ReduceReturnOp>();

    addDynamicallyLegalDialect<KapyDialect, arith::ArithDialect,
                               math::MathDialect, cf::ControlFlowDialect,
                               scf::SCFDialect>([&](Operation *op) {
      bool hasLegalRegions = true;
      for (auto &region : op->getRegions())
        hasLegalRegions &= typeConverter.isLegal(op);
      return hasLegalRegions && typeConverter.isLegal(op);
    });
  }
};

#define GEN_PASS_DEF_CONVERTKAPYTOKGPU
#include "kapy/Conversion/KapyToKgpu/Passes.h.inc"

class ConvertKapyToKgpuPass
    : public impl::ConvertKapyToKgpuBase<ConvertKapyToKgpuPass> {
public:
  virtual void runOnOperation() override {
    auto module = getOperation();

    auto *context = &getContext();
    RewritePatternSet patterns(context);
    KgpuTypeConverter typeConverter(context);
    KgpuConversionTarget convTarget(context, typeConverter);
    populateArithConversionPatterns(typeConverter, patterns);
    populateMathConversionPatterns(typeConverter, patterns);
    populateKapyConversionPatterns(typeConverter, patterns);
    populateSCFConversionPatterns(typeConverter, patterns);
    populateCFConversionPatterns(typeConverter, patterns);

    if (failed(applyPartialConversion(module, convTarget, std::move(patterns))))
      return signalPassFailure();
  }
};

} // namespace

std::unique_ptr<Pass> kapy::createConvertKapyToKgpuPass() {
  return std::make_unique<ConvertKapyToKgpuPass>();
}

void kapy::registerConvertKapyToKgpuPass() {
  registerPass([]() { return createConvertKapyToKgpuPass(); });
}
