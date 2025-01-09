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

#include "kapy/Analysis/Layout.h"
#include "kapy/Conversion/KapyToKgpu/ConversionTarget.h"
#include "kapy/Conversion/KapyToKgpu/Passes.h"
#include "kapy/Conversion/KapyToKgpu/TypeConverter.h"
#include "kapy/Dialect/Kapy/IR/Kapy.h"
#include "kapy/Dialect/Kapy/IR/Utils.h"
#include "kapy/Dialect/Kgpu/IR/Kgpu.h"

using namespace mlir;
using namespace mlir::kapy;

/// Pass named attributes (e.g. kapy.divisibility).
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
    auto shapedType =
        cast<ShapedType>(typeConverter->convertType(op.getType()));
    auto value = cast<DenseElementsAttr>(adaptor.getValue());
    if (isa<RankedTensorType>(shapedType)) {
      if (value.getElementType().isInteger(1) && value.isSplat()) {
        // Workaround until https://reviews.llvm.org/D133743 is included.
        value = DenseElementsAttr::get(shapedType, value.getSplatValue<bool>());
      } else {
        // This is a hack. We just want to add encoding.
        value = value.reshape(shapedType);
      }
    }
    auto newOp =
        rewriter.replaceOpWithNewOp<arith::ConstantOp>(op, shapedType, value);
    addNamedAttributes(newOp, adaptor.getAttributes());
    return success();
  }
};

class UnsqueezeOpConversion : public OpConversionPattern<UnsqueezeOp> {
public:
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(UnsqueezeOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto operand = adaptor.getOperand();
    auto operandType = cast<RankedTensorType>(operand.getType());
    if (!operandType.getEncoding())
      return failure();
    auto operandLayout = cast<RegistersLayoutAttr>(operandType.getEncoding());
    auto axis = op.getAxis();

    auto shapeOfWarps = operandLayout.getShapeOfWarps();
    shapeOfWarps.insert(shapeOfWarps.begin() + axis, 1);
    auto loopsPerWarp = operandLayout.getLoopsPerWarp();
    loopsPerWarp.insert(loopsPerWarp.begin() + axis, 1);
    auto shapeOfLanes = operandLayout.getShapeOfLanes();
    shapeOfLanes.insert(shapeOfLanes.begin() + axis, 1);
    auto loopsPerLane = operandLayout.getLoopsPerLane();
    loopsPerLane.insert(loopsPerLane.begin() + axis, 1);

    auto regisLayout =
        RegistersLayoutAttr::get(op.getContext(), shapeOfWarps, loopsPerWarp,
                                 shapeOfLanes, loopsPerLane);
    auto sliceLayout =
        SliceAxisLayoutAttr::get(op.getContext(), regisLayout, axis);

    operandType = cloneWith(operandType, sliceLayout);
    operand = rewriter.create<ChangeOp>(operand.getLoc(), operandType, operand);
    auto newOp = rewriter.replaceOpWithNewOp<UnsqueezeOp>(op, operand, axis);
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
    auto operand = adaptor.getOperand();
    auto operandType = cast<RankedTensorType>(operand.getType());
    auto operandLayout = operandType.getEncoding();
    if (!operandLayout)
      return failure();
    auto newType = cloneWith(op.getType(), operandLayout);
    auto newOp = rewriter.replaceOpWithNewOp<BroadcastOp>(op, newType, operand);
    addNamedAttributes(newOp, adaptor.getAttributes());
    return success();
  }
};

class PermuteOpConversion : public OpConversionPattern<PermuteOp> {
public:
  using OpConversionPattern::OpConversionPattern;

  virtual LogicalResult
  matchAndRewrite(PermuteOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto operand = adaptor.getOperand();
    auto newOp =
        rewriter.replaceOpWithNewOp<PermuteOp>(op, operand, op.getOrder());
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
    auto numWarps = getTypeConverter<KgpuTypeConverter>()->getNumWarps();
    auto numThreads = numLanes * numWarps;
    auto type = op.getType();
    auto rank = type.getRank();
    auto shape = type.getShape();
    auto numElems = product(shape);

    SmallVector<int64_t, 4> loopsPerWarp(rank, 1);
    SmallVector<int64_t, 4> loopsPerLane(rank, 1);
    if (shape[rank - 1] >= 32 && shape[rank - 2] >= 32 &&
        numElems / numThreads >= 16) {
      loopsPerLane[rank - 1] = 4;
      loopsPerLane[rank - 2] = 4;
    } else {
      loopsPerLane[rank - 1] = 2;
      loopsPerLane[rank - 2] = 2;
    }

    auto accumLayout = getRegistersLayout(op.getContext(), loopsPerWarp,
                                          loopsPerLane, shape, numWarps);
    auto accumType = cloneWith(type, accumLayout);
    auto accum = adaptor.getAccum();
    accum = rewriter.create<ChangeOp>(accum.getLoc(), accumType, accum);

    auto lhs = adaptor.getLhs();
    auto lhsType = cast<RankedTensorType>(lhs.getType());
    auto lhsLayout = MmOperandLayoutAttr::get(op.getContext(), accumLayout, 0,
                                              lhsType.getElementType());
    lhsType = cloneWith(lhsType, lhsLayout);
    lhs = rewriter.create<ChangeOp>(lhs.getLoc(), lhsType, lhs);

    auto rhs = adaptor.getRhs();
    auto rhsType = cast<RankedTensorType>(rhs.getType());
    auto rhsLayout = MmOperandLayoutAttr::get(op.getContext(), accumLayout, 1,
                                              rhsType.getElementType());
    rhsType = cloneWith(rhsType, rhsLayout);
    rhs = rewriter.create<ChangeOp>(rhs.getLoc(), rhsType, rhs);

    auto newOp = rewriter.replaceOpWithNewOp<MatmulOp>(op, lhs, rhs, accum,
                                                       op.getMatmulFormat());
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
    auto newOp = rewriter.create<ReduceOp>(op.getLoc(), adaptor.getOperand(),
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
    for (auto i : {0, 1}) {
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

static void populateArithOpsConversionPatterns(KgpuTypeConverter &typeConverter,
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

static void populateMathOpsConversionPatterns(KgpuTypeConverter &typeConverter,
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

static void populateKapyOpsConversionPatterns(KgpuTypeConverter &typeConverter,
                                              RewritePatternSet &patterns) {
  auto *context = patterns.getContext();
  patterns.add<GenericOpConversion<FpToFpOp>>(typeConverter, context);
  patterns.add<GenericOpConversion<ClampFOp>>(typeConverter, context);
  patterns.add<GenericOpConversion<MulhiUIOp>>(typeConverter, context);
  patterns.add<GenericOpConversion<ArangeOp>>(typeConverter, context);
  patterns.add<GenericOpConversion<LoadOp>>(typeConverter, context);
  patterns.add<GenericOpConversion<StoreOp>>(typeConverter, context);
  patterns.add<GenericOpConversion<AtomicRMWOp>>(typeConverter, context);
  patterns.add<GenericOpConversion<AtomicCASOp>>(typeConverter, context);
  patterns.add<GenericOpConversion<SplatOp>>(typeConverter, context);
  patterns.add<UnsqueezeOpConversion>(typeConverter, context);
  patterns.add<BroadcastOpConversion>(typeConverter, context);
  patterns.add<PermuteOpConversion>(typeConverter, context);
  patterns.add<MatmulOpConversion>(typeConverter, context);
  patterns.add<ReduceOpConversion>(typeConverter, context);
  patterns.add<GenericOpConversion<ElementwiseExternOp>>(typeConverter,
                                                         context);
  patterns.add<GenericOpConversion<ElementwiseInlineOp>>(typeConverter,
                                                         context);
  patterns.add<GenericOpConversion<CallOp>>(typeConverter, context);
  patterns.add<FuncOpConversion>(typeConverter, context);
  patterns.add<GenericOpConversion<ReturnOp>>(typeConverter, context);
}

static void populateSCFOpsConversionPatterns(KgpuTypeConverter &typeConverter,
                                             RewritePatternSet &patterns) {
  auto *context = patterns.getContext();
  patterns.add<ForOpConversion>(typeConverter, context);
  patterns.add<IfOpConversion>(typeConverter, context);
  patterns.add<WhileOpConversion, ConditionOpConversion>(typeConverter,
                                                         context);
  patterns.add<GenericOpConversion<scf::YieldOp>>(typeConverter, context);
}

static void populateCFOpsConversionPatterns(KgpuTypeConverter &typeConverter,
                                            RewritePatternSet &patterns) {
  auto *context = patterns.getContext();
  patterns.add<BranchOpConversion>(typeConverter, context);
  patterns.add<CondBranchOpConversion>(typeConverter, context);
}

namespace {

#define GEN_PASS_DECL_CONVERTKAPYTOKGPU
#define GEN_PASS_DEF_CONVERTKAPYTOKGPU
#include "kapy/Conversion/KapyToKgpu/Passes.h.inc"

class ConvertKapyToKgpuPass
    : public impl::ConvertKapyToKgpuBase<ConvertKapyToKgpuPass> {
public:
  ConvertKapyToKgpuPass() = default;
  ConvertKapyToKgpuPass(int64_t nvidiaCC, int64_t numWarps) {
    this->nvidiaCC = nvidiaCC;
    this->numWarps = numWarps;
  }

  virtual void runOnOperation() override {
    auto module = getOperation();
    auto *context = &getContext();

    RewritePatternSet patterns(context);
    KgpuTypeConverter typeConverter(context, numWarps);
    KgpuConversionTarget convTarget(context, typeConverter);
    populateArithOpsConversionPatterns(typeConverter, patterns);
    populateMathOpsConversionPatterns(typeConverter, patterns);
    populateKapyOpsConversionPatterns(typeConverter, patterns);
    populateSCFOpsConversionPatterns(typeConverter, patterns);
    populateCFOpsConversionPatterns(typeConverter, patterns);

    auto i64Type = IntegerType::get(context, 64);
    module->setAttr("kgpu.nvidia_cc",
                    IntegerAttr::get(i64Type, APInt(64, nvidiaCC.getValue())));
    module->setAttr("kgpu.num_warps",
                    IntegerAttr::get(i64Type, APInt(64, numWarps.getValue())));

    if (failed(applyPartialConversion(module, convTarget, std::move(patterns))))
      return signalPassFailure();
  }
};

} // namespace

std::unique_ptr<Pass> kapy::createConvertKapyToKgpuPass() {
  return std::make_unique<ConvertKapyToKgpuPass>();
}

std::unique_ptr<Pass> kapy::createConvertKapyToKgpuPass(int64_t nvidiaCC,
                                                        int64_t numWarps) {
  return std::make_unique<ConvertKapyToKgpuPass>(nvidiaCC, numWarps);
}
