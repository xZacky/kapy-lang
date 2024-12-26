//===- OptimizeDotOperand.cpp -----------------------------------*- C++ -*-===//
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
#include "kapy/Dialect/Kgpu/IR/Kgpu.h"
#include "kapy/Dialect/Kgpu/Transform/Passes.h"

#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

using namespace mlir;
using namespace mlir::kapy;

static bool isSupportedElementwise(Operation *op) {
  // Only consider custom cast or arith operations.
  // TODO: Is this too restrictive?
  return isa<FPToFPOp>(op) || fromDialect<arith::ArithDialect>(op);
}

namespace {
/// The goal is to put the change right next to the originating load. If we can
/// accomplish this, then we can save a shared memory round-trip.
///
/// Before:
/// - Load from global memory to shared memory using an async copy.
/// - Load from shared memory into a registers layout.
/// - Do elementwise operations over registers layout.
/// - Change to dot op load layout (round-trip through shared memory).
/// - Do dot.
///
/// After:
/// - Load from global memory to shared memory using an async copy.
/// - Load from shared memory into a dot op load layout.
/// - Do elementwise operations over dot op load layout.
/// - Do dot.
///
/// Eliminating the shared memory round-trip is such a big win, we're willing to
/// do it even if this duplicates work because some elementwise operations have
/// uses that don't flow into the dot. On the other hand, we only do this if we
/// can in fact reduce shared memory round-trip - for example, simply moving a
/// change up above an add now means we have two changes. That's worse, unless
/// we can continue moving the change upwards and eventually merge it with a
/// load. So we try to check this will be beneficial before making any changes.
class SaveSharedMemRoundTrip : public OpRewritePattern<ChangeOp> {
public:
  using OpRewritePattern::OpRewritePattern;

  virtual LogicalResult
  matchAndRewrite(ChangeOp op, PatternRewriter &rewriter) const override {
    auto operandType = op.getOperand().getType();
    auto resultType = op.getType();
    if (!isa<DotOpLoadLayoutAttr>(resultType.getEncoding()))
      return failure();
    auto dotldLayout = cast<DotOpLoadLayoutAttr>(resultType.getEncoding());
    auto *defOp = op.getOperand().getDefiningOp();
    if (!defOp || defOp->getNumOperands() == 0 || defOp->getNumResults() != 1)
      return failure();
    if (!all_of(defOp->getOperandTypes(),
                [](Type type) { return isa<RankedTensorType>(type); }))
      return failure();
    if (!isSupportedElementwise(defOp))
      return failure();
    // Currently, these operations are not supported during lowering from shared
    // memory layout to dot op load layout.
    if (isa<arith::TruncIOp, arith::TruncFOp, arith::SelectOp>(defOp))
      return failure();

    // Check that the change is transitively dependent on a load, and all the
    // operations between the load and the change are layout preserving.
    // TODO: This is accidentally quadratic, we iterate over the whole slice
    // but then at the end we only modify one operation!
    SetVector<Operation *> slice;
    BackwardSliceOptions options;
    options.omitBlockArguments = true;
    getBackwardSlice(op.getOperation(), &slice, options);

    bool hasPathFromLoad;
    for (auto *curOp : slice)
      if (auto loadOp = dyn_cast<LoadOp>(curOp))
        if (hasRestrictedPath(loadOp, op, slice, isSupportedElementwise))
          hasPathFromLoad = true;
    if (!hasPathFromLoad)
      return failure();

    SmallVector<Value> newOperands;
    for (auto operand : defOp->getOperands()) {
      auto newType =
          cloneWith(cast<RankedTensorType>(operand.getType()), dotldLayout);
      newOperands.push_back(
          rewriter.create<ChangeOp>(op.getLoc(), newType, operand));
    }

    auto *newOp = rewriter.clone(*defOp);
    for (int i = 0; i < newOperands.size(); ++i)
      newOp->setOperand(i, newOperands[i]);
    auto newType = cloneWith(operandType, dotldLayout);
    newOp->getResult(0).setType(newType);

    rewriter.replaceOp(op, newOp);
    return success();
  }
};

#define GEN_PASS_DEF_KGPUOPTIMIZEDOTOPERAND
#include "kapy/Dialect/Kgpu/Transform/Passes.h.inc"

class KgpuOptimizeDotOperandPass
    : public impl::KgpuOptimizeDotOperandBase<KgpuOptimizeDotOperandPass> {
public:
  virtual void runOnOperation() override {
    auto module = getOperation();
    auto *context = &getContext();

    RewritePatternSet patterns(context);
    patterns.add<SaveSharedMemRoundTrip>(context);
    ChangeOp::getCanonicalizationPatterns(patterns, context);

    if (failed(applyPatternsAndFoldGreedily(module, std::move(patterns))))
      signalPassFailure();
  }
};
} // namespace

std::unique_ptr<Pass> kapy::createKgpuOptimizeDotOperandPass() {
  return std::make_unique<KgpuOptimizeDotOperandPass>();
}
