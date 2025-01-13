//===- ReorderBroadcast.cpp -------------------------------------*- C++ -*-===//
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
#include "kapy/Dialect/Kapy/Transform/Passes.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

using namespace mlir;
using namespace mlir::kapy;

static bool isSplat(Operation *op) {
  if (isa<SplatOp>(op))
    return true;
  DenseElementsAttr denseAttr;
  return matchPattern(op, m_Constant(&denseAttr)) && denseAttr.isSplat();
}

static bool isSplat(Operation *op, DenseElementsAttr &denseAttr) {
  return matchPattern(op, m_Constant(&denseAttr)) && denseAttr.isSplat();
}

namespace {

class MoveSplatOpAfterElementwiseOp
    : public OpTraitRewritePattern<OpTrait::Elementwise> {
public:
  using OpTraitRewritePattern::OpTraitRewritePattern;

  virtual LogicalResult match(Operation *op) const override {
    if (!isMemoryEffectFree(op))
      return failure();

    for (auto operand : op->getOperands()) {
      auto *defOp = operand.getDefiningOp();
      if (!defOp || !isSplat(defOp))
        return failure();
    }
    return success(op->getNumOperands() > 0);
  }

  virtual void rewrite(Operation *op,
                       PatternRewriter &rewriter) const override {
    auto loc = op->getLoc();
    auto operands = op->getOperands();

    SmallVector<Value> newOperands(operands.size());
    for (unsigned i = 0; i < operands.size(); ++i) {
      auto *defOp = operands[i].getDefiningOp();
      DenseElementsAttr denseAttr;
      if (auto splatOp = dyn_cast<SplatOp>(defOp))
        newOperands[i] = splatOp.getOperand();
      else if (isSplat(defOp, denseAttr))
        newOperands[i] = arith::ConstantOp::materialize(
            rewriter, denseAttr.getSplatValue<Attribute>(),
            denseAttr.getElementType(), loc);
    }

    auto types = op->getResultTypes();
    SmallVector<Type> newTypes;
    for (auto type : types)
      newTypes.push_back(cast<ShapedType>(type).getElementType());

    auto *newOp = rewriter.create(loc, op->getName().getIdentifier(),
                                  newOperands, newTypes, op->getAttrs());
    for (unsigned i = 0; i < types.size(); ++i)
      rewriter.replaceAllUsesWith(
          op->getResult(i),
          rewriter.create<SplatOp>(loc, types[i], newOp->getResult(i)));
  }
};

/// This also generalizes to multiple elementwise operands when the rest are
/// splat like, but multiple broadcast operands with different shapes are not
/// handled.
class MoveBroadcastOpAfterElementwiseOp
    : public OpTraitRewritePattern<OpTrait::Elementwise> {
public:
  using OpTraitRewritePattern::OpTraitRewritePattern;

  virtual LogicalResult match(Operation *op) const override {
    if (!isMemoryEffectFree(op))
      return failure();

    BroadcastOp matchOp;
    auto haveSameOperandShape = [](BroadcastOp op0, BroadcastOp op1) {
      return op0.getOperand().getType().getShape() ==
             op1.getOperand().getType().getShape();
    };
    for (auto operand : op->getOperands()) {
      auto *defOp = operand.getDefiningOp();
      if (!defOp)
        return failure();
      if (auto broadcastOp = dyn_cast<BroadcastOp>(defOp)) {
        if (!matchOp)
          matchOp = broadcastOp;
        else if (!haveSameOperandShape(matchOp, broadcastOp))
          return failure();
      } else if (!isSplat(defOp)) {
        return failure();
      }
    }
    return success(matchOp);
  }

  virtual void rewrite(Operation *op,
                       PatternRewriter &rewriter) const override {
    auto loc = op->getLoc();
    auto operands = op->getOperands();

    BroadcastOp broadcastOp;
    for (auto operand : operands) {
      broadcastOp = operand.getDefiningOp<BroadcastOp>();
      if (broadcastOp)
        break;
    }
    auto tmpType = broadcastOp.getOperand().getType();
    auto getNewType = [tmpType](Type type) {
      auto elementType = cast<ShapedType>(type).getElementType();
      return cloneWith(tmpType, elementType);
    };

    SmallVector<Value> newOperands;
    for (auto operand : operands) {
      auto *defOp = operand.getDefiningOp();
      // Case 1: The BroadcastOp.
      if (isa<BroadcastOp>(defOp)) {
        newOperands.push_back(defOp->getOperand(0));
        continue;
      }
      // Make other operands the same shape with the BroadcastOp's operand.
      auto newType = getNewType(operand.getType());
      // Case 2: SplatOp.
      if (auto splatOp = dyn_cast<SplatOp>(defOp)) {
        newOperands.push_back(
            rewriter.create<SplatOp>(loc, newType, splatOp.getOperand()));
        continue;
      }
      // Case 3: Splat ConstantOp.
      DenseElementsAttr denseAttr;
      if (isSplat(defOp, denseAttr)) {
        newOperands.push_back(rewriter.create<arith::ConstantOp>(
            loc, newType,
            SplatElementsAttr::get(newType,
                                   denseAttr.getSplatValue<Attribute>())));
        continue;
      }
    }

    auto types = op->getResultTypes();
    SmallVector<Type> newTypes;
    for (auto type : types)
      newTypes.push_back(getNewType(type));

    auto *newOp = rewriter.create(loc, op->getName().getIdentifier(),
                                  newOperands, newTypes, op->getAttrs());
    for (unsigned i = 0; i < newTypes.size(); ++i)
      rewriter.replaceAllUsesWith(
          op->getResult(i),
          rewriter.create<BroadcastOp>(loc, types[i], newOp->getResult(i)));
  }
};

#define GEN_PASS_DEF_KAPYREORDERBROADCAST
#include "kapy/Dialect/Kapy/Transform/Passes.h.inc"

class KapyReorderBroadcastPass
    : public impl::KapyReorderBroadcastBase<KapyReorderBroadcastPass> {
public:
  virtual void runOnOperation() override {
    auto module = getOperation();

    auto *context = &getContext();
    RewritePatternSet patterns(context);
    UnsqueezeOp::getCanonicalizationPatterns(patterns, context);
    BroadcastOp::getCanonicalizationPatterns(patterns, context);
    patterns.add<MoveSplatOpAfterElementwiseOp>(context);
    patterns.add<MoveBroadcastOpAfterElementwiseOp>(context);

    if (failed(applyPatternsAndFoldGreedily(module, std::move(patterns))))
      signalPassFailure();
  }
};

} // namespace

std::unique_ptr<Pass> kapy::createKapyReorderBroadcastPass() {
  return std::make_unique<KapyReorderBroadcastPass>();
}
