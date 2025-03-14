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

#include "kapy/Conversion/KapyToKgpu/ConversionTarget.h"
#include "kapy/Conversion/KapyToKgpu/Passes.h"
#include "kapy/Conversion/KapyToKgpu/TypeConverter.h"
#include "kapy/Dialect/Kapy/IR/Kapy.h"
#include "kapy/Dialect/Kgpu/IR/Kgpu.h"
#include "kapy/Support/LayoutUtils.h"

using namespace mlir;
using namespace mlir::kapy;

/// Pass named attributes (e.g. kapy.alignment).
static void addNamedAttributes(Operation *op, DictionaryAttr attrs) {
  for (auto it : attrs)
    if (!op->hasAttr(it.getName()))
      op->setAttr(it.getName(), it.getValue());
}

static LogicalResult inlineAndConvertRegion(ConversionPatternRewriter &rewriter,
                                            const TypeConverter &typeConverter,
                                            Region &region, Region &parent) {
  rewriter.inlineRegionBefore(region, parent, parent.end());
  return rewriter.convertRegionTypes(&parent, typeConverter);
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
    SmallVector<Type> resultTypes;
    const TypeConverter *typeConverter = this->getTypeConverter();
    if (failed(typeConverter->convertTypes(op->getResultTypes(), resultTypes)))
      return failure();
    rewriter.replaceOpWithNewOp<OpT>(op, resultTypes, adaptor.getOperands(),
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
    auto resultType =
        cast<RankedTensorType>(typeConverter->convertType(op.getType()));
    auto value = cast<DenseElementsAttr>(adaptor.getValue());
    // This is a hack. We just want to add layout.
    value = value.reshape(resultType);
    auto newOp =
        rewriter.replaceOpWithNewOp<arith::ConstantOp>(op, resultType, value);
    addNamedAttributes(newOp, adaptor.getAttributes());
    return success();
  }
};

class LdMatrixOpConversion : public OpConversionPattern<LdMatrixOp> {
public:
  using OpConversionPattern::OpConversionPattern;

  virtual LogicalResult
  matchAndRewrite(LdMatrixOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto [loaderLayout, resultLayout] = getDefaultLayouts(op);

    auto loader = adaptor.getLoader();
    auto loaderType = cast<RankedTensorType>(loader.getType());
    loaderType = cloneWithLayout(loaderType, loaderLayout);
    loader = rewriter.create<ChangeOp>(loader.getLoc(), loaderType, loader);

    auto resultType = op.getType();
    resultType = cloneWithLayout(resultType, resultLayout);

    auto newOp = rewriter.replaceOpWithNewOp<LdMatrixOp>(
        op, resultType, adaptor.getSource(), loader);
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
    auto resultType = cloneWithShape(sourceType, op.getType().getShape());
    auto newOp =
        rewriter.replaceOpWithNewOp<BroadcastOp>(op, resultType, source);
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
    auto [lhsLayout, rhsLayout, accLayout] = getDefaultLayouts(op);

    auto lhs = adaptor.getLhs();
    auto lhsType = cast<RankedTensorType>(lhs.getType());
    lhsType = cloneWithLayout(lhsType, lhsLayout);
    lhs = rewriter.create<ChangeOp>(lhs.getLoc(), lhsType, lhs);

    auto rhs = adaptor.getRhs();
    auto rhsType = cast<RankedTensorType>(rhs.getType());
    rhsType = cloneWithLayout(rhsType, rhsLayout);
    rhs = rewriter.create<ChangeOp>(rhs.getLoc(), rhsType, rhs);

    auto acc = adaptor.getAcc();
    auto accType = cast<RankedTensorType>(acc.getType());
    accType = cloneWithLayout(accType, accLayout);
    acc = rewriter.create<ChangeOp>(acc.getLoc(), accType, acc);

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
    auto newOp = rewriter.replaceOpWithNewOp<ReduceOp>(op, adaptor.getSource(),
                                                       op.getAxis());
    addNamedAttributes(newOp, adaptor.getAttributes());
    return inlineAndConvertRegion(rewriter, *typeConverter, op.getRegion(),
                                  newOp.getRegion());
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
    return inlineAndConvertRegion(rewriter, *typeConverter, op.getBody(),
                                  newOp.getBody());
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
    if (failed(inlineAndConvertRegion(rewriter, *typeConverter, op.getRegion(),
                                      newOp.getRegion())))
      return failure();

    newOp->setOperands(adaptor.getOperands());

    SmallVector<Type> resultTypes;
    for (auto type : op.getResultTypes()) {
      type = typeConverter->convertType(type);
      if (!type)
        return failure();
      resultTypes.push_back(type);
    }
    for (auto [result, type] : zip(newOp.getResults(), resultTypes))
      result.setType(type);

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
    if (failed(inlineAndConvertRegion(rewriter, *typeConverter,
                                      op.getThenRegion(),
                                      newOp.getThenRegion())))
      return failure();
    if (failed(inlineAndConvertRegion(rewriter, *typeConverter,
                                      op.getElseRegion(),
                                      newOp.getElseRegion())))
      return failure();

    newOp->setOperands(adaptor.getOperands());

    SmallVector<Type> resultTypes;
    for (auto type : op.getResultTypes()) {
      type = typeConverter->convertType(type);
      if (!type)
        return failure();
      resultTypes.push_back(type);
    }
    for (auto [result, type] : zip(newOp.getResults(), resultTypes))
      result.setType(type);

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
    SmallVector<Type> resultTypes;
    if (failed(typeConverter->convertTypes(op.getResultTypes(), resultTypes)))
      return failure();
    auto newOp = rewriter.create<scf::WhileOp>(op.getLoc(), resultTypes,
                                               adaptor.getOperands());
    addNamedAttributes(newOp, adaptor.getAttributes());
    if (failed(inlineAndConvertRegion(rewriter, *typeConverter, op.getBefore(),
                                      newOp.getBefore())))
      return failure();
    if (failed(inlineAndConvertRegion(rewriter, *typeConverter, op.getAfter(),
                                      newOp.getAfter())))
      return failure();
    rewriter.replaceOp(op, newOp.getResults());
    return success();
  }
};

} // namespace

static void populateArithConversionPatterns(const TypeConverter &typeConverter,
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
  patterns.add<GenericOpConversion<arith::TruncIOp>,
               GenericOpConversion<arith::TruncFOp>>(typeConverter, context);
  patterns.add<GenericOpConversion<arith::ExtUIOp>,
               GenericOpConversion<arith::ExtSIOp>,
               GenericOpConversion<arith::ExtFOp>>(typeConverter, context);
  patterns.add<GenericOpConversion<arith::SIToFPOp>,
               GenericOpConversion<arith::FPToSIOp>>(typeConverter, context);
  patterns.add<GenericOpConversion<arith::BitcastOp>>(typeConverter, context);
  patterns.add<GenericOpConversion<arith::SelectOp>>(typeConverter, context);
}

static void populateMathConversionPatterns(const TypeConverter &typeConverter,
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

static void populateKapyConversionPatterns(const TypeConverter &typeConverter,
                                           RewritePatternSet &patterns) {
  auto *context = patterns.getContext();
  patterns.add<GenericOpConversion<FPToFPOp>>(typeConverter, context);
  patterns.add<GenericOpConversion<ClampFOp>>(typeConverter, context);
  patterns.add<GenericOpConversion<LdGlobalOp>>(typeConverter, context);
  patterns.add<GenericOpConversion<StGlobalOp>>(typeConverter, context);
  patterns.add<GenericOpConversion<LdSharedOp>>(typeConverter, context);
  patterns.add<GenericOpConversion<StSharedOp>>(typeConverter, context);
  patterns.add<LdMatrixOpConversion>(typeConverter, context);
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

static void populateSCFConversionPatterns(const TypeConverter &typeConverter,
                                          RewritePatternSet &patterns) {
  auto *context = patterns.getContext();
  patterns.add<ForOpConversion>(typeConverter, context);
  patterns.add<IfOpConversion>(typeConverter, context);
  patterns.add<WhileOpConversion>(typeConverter, context);
  patterns.add<GenericOpConversion<scf::ConditionOp>>(typeConverter, context);
  patterns.add<GenericOpConversion<scf::YieldOp>>(typeConverter, context);
}

namespace {

#define GEN_PASS_DEF_CONVERTKAPYTOKGPU
#include "kapy/Conversion/KapyToKgpu/Passes.h.inc"

class ConvertKapyToKgpuPass
    : public impl::ConvertKapyToKgpuBase<ConvertKapyToKgpuPass> {
public:
  virtual void runOnOperation() override {
    auto *context = &getContext();
    RewritePatternSet patterns(context);
    KapyToKgpuTypeConverter typeConverter(context);
    KapyToKgpuConversionTarget convTarget(context, typeConverter);
    populateArithConversionPatterns(typeConverter, patterns);
    populateMathConversionPatterns(typeConverter, patterns);
    populateKapyConversionPatterns(typeConverter, patterns);
    populateSCFConversionPatterns(typeConverter, patterns);

    auto module = getOperation();
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
