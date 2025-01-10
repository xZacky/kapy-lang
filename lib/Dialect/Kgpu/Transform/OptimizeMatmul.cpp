//===- OptimizeMatmul.cpp ---------------------------------------*- C++ -*-===//
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
#include "kapy/Analysis/Utils.h"
#include "kapy/Dialect/Kapy/IR/Kapy.h"
#include "kapy/Dialect/Kgpu/IR/Kgpu.h"
#include "kapy/Dialect/Kgpu/Transform/Passes.h"

#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

using namespace mlir;
using namespace mlir::kapy;

/// Find the first different element bit-width in the chain of shape preserving
/// unary operations that `value` depends on.
///
/// There are two primary scenarios:
/// 1. Upcasting:
///    Sequence such as loading a f16, followed by arithmetic operations, then
///    bitcasting to f32, and finally computed in f32.
/// 2. Downcasting:
///    Sequence such as loading a f32, followed by arithmetic operations, then
///    bitcasting to f16, and finally computed in f16.
///
/// In the upcasting scenarios, element reordering converts the original
/// elements distribution to the order of higher precision primitives, as a
/// result, bit-width can follow the lower precision primitive.
/// Conversely, in the downcasting scenarios, no reordering is performed, making
/// it directly use the lower precision primitive.
static unsigned getOriginalBitWidth(Value value) {
  SetVector<Operation *> slice;
  BackwardSliceOptions options;
  options.omitBlockArguments = true;
  auto filter = [](Operation *op) {
    if (op->getNumOperands() != 1)
      return false;
    return isa<FpToFpOp, ChangeOp>(op) || fromDialect<arith::ArithDialect>(op);
  };
  options.filter = filter;
  getBackwardSlice(value, &slice, options);

  auto bitWidth = getIntOrFloatBitWidth(value.getType());
  for (auto *op : slice)
    if (auto operand = op->getOperand(0))
      if (auto tensorType = dyn_cast<RankedTensorType>(operand.getType()))
        bitWidth = std::min(bitWidth, getIntOrFloatBitWidth(tensorType));
  return bitWidth;
}

static Value castOperand(OpBuilder &builder, Value operand, Type elementType) {
  auto newType =
      cloneWith(cast<RankedTensorType>(operand.getType()), elementType);
  return builder.create<FpToFpOp>(operand.getLoc(), newType, operand);
}

static void decomposeMixedModeMatmulOp(ModuleOp module, int64_t nvidiaCC) {
  module.walk([=](MatmulOp matmulOp) {
    OpBuilder builder(matmulOp);
    auto lhsType = matmulOp.getLhs().getType();
    auto rhsType = matmulOp.getRhs().getType();
    auto resultType = matmulOp.getType();
    Type elementType;
    auto nvmmaLayout = dyn_cast<NvidiaMmaLayoutAttr>(resultType.getEncoding());
    if (nvmmaLayout) {
      bool isNativeF8 = lhsType.getElementType().isFloat8E5M2() ||
                        lhsType.getElementType().isFloat8E5M2FNUZ() ||
                        lhsType.getElementType().isFloat8E4M3FNUZ() ||
                        rhsType.getElementType().isFloat8E5M2() ||
                        rhsType.getElementType().isFloat8E5M2FNUZ() ||
                        rhsType.getElementType().isFloat8E4M3FNUZ();
      if (!isNativeF8 || nvidiaCC == 89)
        return;
      elementType = builder.getF16Type();
    } else {
      if (lhsType.getElementType() == resultType.getElementType() &&
          rhsType.getElementType() == resultType.getElementType())
        return;
      elementType = resultType.getElementType();
    }
    auto newLhs = castOperand(builder, matmulOp.getLhs(), elementType);
    auto newRhs = castOperand(builder, matmulOp.getRhs(), elementType);
    matmulOp.setOperand(0, newLhs);
    matmulOp.setOperand(1, newRhs);
  });
}

static bool isSupportedElementwise(Operation *op) {
  // Only consider custom cast or arith operations.
  // TODO: Is this too restrictive?
  return isa<FpToFpOp>(op) || fromDialect<arith::ArithDialect>(op);
}

namespace {

class ChangeFmaMatmulOpToMmaMatmulOp : public OpRewritePattern<MatmulOp> {
public:
  using OpRewritePattern::OpRewritePattern;

  virtual LogicalResult
  matchAndRewrite(MatmulOp op, PatternRewriter &rewriter) const override {
    auto resultType = op.getType();
    if (!resultType.getEncoding() || hasLayout<NvidiaMmaLayoutAttr>(resultType))
      return failure();
    if (!supportNvidiaMma(op))
      return failure();

    auto numWarps = getNumWarps(op->getParentOfType<ModuleOp>());
    auto accumLayout = getNvidiaMmaLayout(op, numWarps);
    auto accumType = cloneWith(resultType, accumLayout);
    auto accum = op.getAccum();
    accum = rewriter.create<ChangeOp>(accum.getLoc(), accumType, accum);

    auto lhs = op.getLhs();
    auto rhs = op.getRhs();
    auto lhsType = lhs.getType();
    auto rhsType = rhs.getType();
    auto bitWidth =
        std::min(getOriginalBitWidth(lhs), getOriginalBitWidth(rhs));

    auto lhsLayout =
        MmOperandLayoutAttr::get(op.getContext(), accumLayout, 0, bitWidth);
    lhsType = cloneWith(lhsType, lhsLayout);
    lhs = rewriter.create<ChangeOp>(lhs.getLoc(), lhsType, lhs);

    auto rhsLayout =
        MmOperandLayoutAttr::get(op.getContext(), accumLayout, 1, bitWidth);
    rhsType = cloneWith(rhsType, rhsLayout);
    rhs = rewriter.create<ChangeOp>(rhs.getLoc(), rhsType, rhs);

    auto newOp = rewriter.create<MatmulOp>(op.getLoc(), lhs, rhs, accum,
                                           op.getMatmulFormat());
    rewriter.replaceOpWithNewOp<ChangeOp>(op, resultType, newOp);
    return success();
  }
};

/// The goal is to put the change right next to the originating load. If we can
/// accomplish this, then we can save a shared memory round-trip.
///
/// Before:
/// 1. Load from global memory to shared memory using asynchronously copy.
/// 2. Load from shared memory into a registers layout.
/// 3. Do elementwise operations over registers layout.
/// 4. Change to matmul operand layout (round-trip through shared memory).
/// 5. Do matmul.
///
/// After:
/// 1. Load from global memory to shared memory using asynchronously copy.
/// 2. Load from shared memory into a matmul operand layout.
/// 3. Do elementwise operations matmul operand layout.
/// 4. Do matmul.
///
/// Eliminating the shared memory round-trip is such a big win, we are willing
/// to do it even if this duplicates work because some elementwise operations
/// have uses that don't flow into the matmul.
/// On the other hand, we only do this if we can in fact save the shared memory
/// round-trip - for example, simply moving a change up above an add now means
/// we have two changes, That's worse, unless we can continue moving the change
/// upwards and eventually merge it with a load. So we try to check this will be
/// beneficial before making any rewrites.
class TryToSaveSharedMemoryRoundTrip : public OpRewritePattern<ChangeOp> {
public:
  using OpRewritePattern::OpRewritePattern;

  virtual LogicalResult
  matchAndRewrite(ChangeOp op, PatternRewriter &rewriter) const override {
    auto resultType = op.getType();
    auto mmopdLayout = dyn_cast<MmOperandLayoutAttr>(resultType.getEncoding());
    if (!mmopdLayout)
      return failure();

    auto *defOp = op.getOperand().getDefiningOp();
    if (!defOp || defOp->getNumOperands() == 0 || defOp->getNumResults() != 1)
      return failure();
    if (!llvm::all_of(defOp->getOperandTypes(),
                      [](Type type) { return isa<RankedTensorType>(type); }))
      return failure();
    if (!isSupportedElementwise(defOp))
      return failure();
    // Currently, these operations are not supported during lowering from shared
    // memory layout to matmul operand layout.
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
          cloneWith(cast<RankedTensorType>(operand.getType()), mmopdLayout);
      newOperands.push_back(
          rewriter.create<ChangeOp>(operand.getLoc(), newType, operand));
    }

    auto *newOp = rewriter.clone(*defOp);
    for (unsigned i = 0; i < newOperands.size(); ++i)
      newOp->setOperand(i, newOperands[i]);
    newOp->getResult(0).setType(resultType);

    rewriter.replaceOp(op, newOp);
    return success();
  }
};

#define GEN_PASS_DEF_KGPUOPTIMIZEMATMUL
#include "kapy/Dialect/Kgpu/Transform/Passes.h.inc"

class KgpuOptimizeMatmulPass
    : public impl::KgpuOptimizeMatmulBase<KgpuOptimizeMatmulPass> {
public:
  virtual void runOnOperation() override {
    auto module = getOperation();

    auto *context = &getContext();
    RewritePatternSet patterns(context);
    patterns.add<ChangeFmaMatmulOpToMmaMatmulOp>(context);
    patterns.add<TryToSaveSharedMemoryRoundTrip>(context);
    ChangeOp::getCanonicalizationPatterns(patterns, context);

    if (failed(applyPatternsAndFoldGreedily(module, std::move(patterns))))
      signalPassFailure();

    decomposeMixedModeMatmulOp(module, getNvidiaCC(module));
  }
};

} // namespace

std::unique_ptr<Pass> kapy::createKgpuOptimizeMatmulPass() {
  return std::make_unique<KgpuOptimizeMatmulPass>();
}
