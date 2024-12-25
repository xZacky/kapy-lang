//===- Combine.cpp ----------------------------------------------*- C++ -*-===//
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

#include "kapy/Analysis/Utils.h"
#include "kapy/Dialect/Kapy/IR/Kapy.h"
#include "kapy/Dialect/Kapy/Transform/Passes.h"

#include "mlir/IR/Dominance.h"
#include "mlir/IR/Matchers.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

using namespace mlir;
using namespace mlir::kapy;
using llvm::MapVector;

static bool isZero(Value value) {
  if (matchPattern(value, m_Zero()) || matchPattern(value, m_AnyZeroFloat()))
    return true;
  if (auto broadcastOp = value.getDefiningOp<BroadcastOp>())
    if (matchPattern(broadcastOp.getOperand(), m_Zero()) ||
        matchPattern(broadcastOp.getOperand(), m_AnyZeroFloat()))
      return true;
  return false;
}

static bool isCombinable(Value offset0, Value offset1) {
  auto getAPInt = [](Value value) -> std::optional<APInt> {
    DenseElementsAttr denseAttr;
    auto *defOp = value.getDefiningOp();
    if (defOp) {
      if (auto splatOp = dyn_cast<SplatOp>(defOp)) {
        value = splatOp.getOperand();
      } else if (matchPattern(defOp, m_Constant(&denseAttr)) &&
                 denseAttr.isSplat()) {
        auto splatValue = denseAttr.getSplatValue<Attribute>();
        if (auto intAttr = dyn_cast_or_null<IntegerAttr>(splatValue))
          return intAttr.getValue();
      }
    }
    APInt apInt;
    if (matchPattern(value, m_ConstantInt(&apInt)))
      return apInt;
    return std::nullopt;
  };

  auto apInt0 = getAPInt(offset0);
  auto apInt1 = getAPInt(offset1);
  if (apInt0.has_value() && apInt1.has_value()) {
    bool overflow = false;
    (void)apInt0.value().sadd_ov(apInt1.value(), overflow);
    return !overflow;
  }
  return false;
}

namespace {
#include "kapy/Dialect/Kapy/Transform/Combine.cpp.inc"

class CombineSelectOpAndLoadOp : public RewritePattern {
public:
  CombineSelectOpAndLoadOp(MLIRContext *context)
      : RewritePattern(arith::SelectOp::getOperationName(), 2, context,
                       {LoadOp::getOperationName()}) {}

  virtual LogicalResult
  matchAndRewrite(Operation *op, PatternRewriter &rewriter) const override {
    auto selectOp = cast<arith::SelectOp>(op);
    auto condition = selectOp.getCondition();
    auto trueValue = selectOp.getTrueValue();
    auto falseValue = selectOp.getFalseValue();

    auto loadOp = trueValue.getDefiningOp<LoadOp>();
    if (!loadOp)
      return failure();
    auto mask = loadOp.getMask();
    if (!mask)
      return failure();
    auto splatOp = mask.getDefiningOp<SplatOp>();
    if (!splatOp)
      return failure();
    if (splatOp.getOperand() != condition)
      return failure();

    rewriter.replaceOpWithNewOp<LoadOp>(
        op, loadOp.getType(), loadOp.getSource(), mask, falseValue,
        loadOp.getCacheModifier(), loadOp.getEvictPriority(),
        loadOp.getIsVolatile());
    return success();
  }
};

class CombineOpsEqualToDotOp : public RewritePattern {
public:
  CombineOpsEqualToDotOp(MLIRContext *context)
      : RewritePattern(ReduceOp::getOperationName(), 1, context,
                       {DotOp::getOperationName()}) {}

  virtual LogicalResult
  matchAndRewrite(Operation *op, PatternRewriter &rewriter) const override {
    auto reduceOp = cast<ReduceOp>(op);
    auto *body = reduceOp.getBody();
    bool isReduceSum = body->getOperations().size() == 2 &&
                       isAddF32(&*body->getOperations().begin());
    if (!isReduceSum)
      return failure();
    auto mulfOp =
        dyn_cast_or_null<arith::MulFOp>(reduceOp.getOperand().getDefiningOp());
    if (!mulfOp)
      return failure();
    auto lhsBroadcastOp =
        dyn_cast_or_null<BroadcastOp>(mulfOp.getLhs().getDefiningOp());
    if (!lhsBroadcastOp)
      return failure();
    auto rhsBroadcastOp =
        dyn_cast_or_null<BroadcastOp>(mulfOp.getRhs().getDefiningOp());
    if (!rhsBroadcastOp)
      return failure();
    auto lhsUnsqueezeOp = dyn_cast_or_null<UnsqueezeOp>(
        lhsBroadcastOp.getOperand().getDefiningOp());
    if (!lhsUnsqueezeOp)
      return failure();
    auto rhsUnsqueezeOp = dyn_cast_or_null<UnsqueezeOp>(
        rhsBroadcastOp.getOperand().getDefiningOp());
    if (!rhsUnsqueezeOp)
      return failure();
    if (lhsUnsqueezeOp.getAxis() != 2 || rhsUnsqueezeOp.getAxis() != 0)
      return failure();

    auto lhsShapedType = cast<ShapedType>(lhsBroadcastOp.getType());
    auto rhsShapedType = cast<ShapedType>(rhsBroadcastOp.getType());
    auto lhsShape = lhsShapedType.getShape();
    auto rhsShape = rhsShapedType.getShape();
    if (lhsShape[2] < 16 || rhsShape[0] < 16)
      return failure();

    rewriter.setInsertionPoint(op);
    auto accumType = RankedTensorType::get({lhsShape[0], rhsShape[2]},
                                           lhsShapedType.getElementType());
    auto zero = rewriter.create<arith::ConstantOp>(op->getLoc(),
                                                   rewriter.getF32FloatAttr(0));
    auto accum = rewriter.create<SplatOp>(op->getLoc(), accumType, zero);
    auto lhs = lhsUnsqueezeOp.getOperand();
    auto rhs = rhsUnsqueezeOp.getOperand();
    rewriter.replaceOpWithNewOp<DotOp>(op, lhs, rhs, accum);
    return success();
  }

  static bool isAddF32(Operation *op) {
    if (auto addfOp = dyn_cast_or_null<arith::AddFOp>(op))
      return addfOp.getType().getIntOrFloatBitWidth() <= 32;
    return false;
  }
};

#define GEN_PASS_DEF_KAPYCOMBINE
#include "kapy/Dialect/Kapy/Transform/Passes.h.inc"

class KapyCombinePass : public impl::KapyCombineBase<KapyCombinePass> {
public:
  virtual void runOnOperation() override {
    auto module = getOperation();
    auto *context = &getContext();

    combineSelectOpAndIfOp();

    RewritePatternSet patterns(context);
    patterns.add<CombineDotOpAsAddIOpLhs>(context);
    patterns.add<CombineDotOpAsAddIOpRhs>(context);
    patterns.add<CombineDotOpAsAddFOpLhs>(context);
    patterns.add<CombineDotOpAsAddFOpRhs>(context);
    patterns.add<CombineTwoMovMemRefOps>(context);
    patterns.add<CombineSelectOpAndLoadOp>(context);
    patterns.add<CombineOpsEqualToDotOp>(context);

    if (failed(applyPatternsAndFoldGreedily(module, std::move(patterns))))
      signalPassFailure();
  }

private:
  void combineSelectOpAndIfOp() {
    auto module = getOperation();

    DominanceInfo domInfo(module);
    auto dominanceRequires = [&](arith::SelectOp selectOp, scf::IfOp ifOp) {
      // IfOp needs to be dominated by the SelectOp.
      if (!domInfo.dominates(selectOp.getOperation(), ifOp.getOperation()))
        return false;
      // IfOp needs to dominate all the SelectOp's users.
      for (auto *useOp : selectOp.getResult().getUsers())
        if (!domInfo.dominates(ifOp.getOperation(), useOp))
          return false;
      return true;
    };

    // Go over the SelectOps, look if there is an IfOp with the same condition.
    MapVector<scf::IfOp, SmallVector<arith::SelectOp>> ifToSelectOps;
    module.walk([&](arith::SelectOp selectOp) {
      auto *block = selectOp->getBlock();
      auto condition = selectOp.getCondition();
      SetVector<Operation *> useOps(condition.getUsers().begin(),
                                    condition.getUsers().end());
      // Sort the users in topological order.
      useOps = multiRootTopoSort(useOps);
      for (auto *useOp : useOps) {
        auto ifOp = dyn_cast<scf::IfOp>(useOp);
        if (!ifOp || ifOp->getBlock() != block)
          continue;
        if (dominanceRequires(selectOp, ifOp)) {
          ifToSelectOps[ifOp].push_back(selectOp);
          break;
        }
      }
    });

    auto updateYieldOp = [](OpBuilder &builder, Location loc,
                            scf::YieldOp yieldOp,
                            SmallVectorImpl<Value> &operands) {
      builder.setInsertionPoint(yieldOp);
      (void)builder.create<scf::YieldOp>(loc, operands);
      yieldOp.erase();
    };

    for (auto [ifOp, selectOps] : ifToSelectOps) {
      // Add new return value to the IfOp (and create else block if necessary),
      // then yield the select value in the then block and the else block.
      OpBuilder builder(ifOp);
      auto loc = ifOp.getLoc();
      SmallVector<Type> newTypes(ifOp.getResultTypes());
      for (auto selectOp : selectOps)
        newTypes.push_back(selectOp.getResult().getType());
      auto newIfOp =
          builder.create<scf::IfOp>(loc, newTypes, ifOp.getCondition(), true);
      newIfOp.getThenRegion().takeBody(ifOp.getThenRegion());
      if (ifOp.elseBlock())
        newIfOp.getElseRegion().takeBody(ifOp.getElseRegion());
      else
        (void)newIfOp.getElseBodyBuilder().create<scf::YieldOp>(loc);

      SmallVector<Value> thenOperands(newIfOp.thenYield().getOperands());
      SmallVector<Value> elseOperands(newIfOp.elseYield().getOperands());
      for (auto selectOp : selectOps) {
        auto thenValue = selectOp.getTrueValue();
        auto elseValue = selectOp.getFalseValue();
        thenOperands.push_back(thenValue);
        elseOperands.push_back(elseValue);
      }
      updateYieldOp(builder, loc, newIfOp.thenYield(), thenOperands);
      updateYieldOp(builder, loc, newIfOp.elseYield(), elseOperands);

      auto i = 0;
      for (auto result : ifOp.getResults())
        result.replaceAllUsesWith(newIfOp->getResult(i++));
      for (auto selectOp : selectOps) {
        selectOp.replaceAllUsesWith(newIfOp->getResult(i++));
        selectOp.erase();
      }

      ifOp.erase();
    }
  }
};
} // namespace

std::unique_ptr<Pass> kapy::createKapyCombinePass() {
  return std::make_unique<KapyCombinePass>();
}
