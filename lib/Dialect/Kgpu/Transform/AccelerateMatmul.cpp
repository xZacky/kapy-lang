//===- AccelerateMatmul.cpp -------------------------------------*- C++ -*-===//
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

static SmallVector<int, 4> getWarpPerCTA(DotOp dotOp, int numWarps) {
  auto type = dotOp.getType();
  auto rank = type.getRank();
  auto shape = type.getShape();

  // Early exit for batched matmul case.
  if (rank == 3)
    return {numWarps, 1, 1};

  auto inSameRegion = [dotOp](Operation *curOp) {
    return dotOp->getParentRegion() == curOp->getParentRegion();
  };
  auto slice = multiRootGetSlice(dotOp, {inSameRegion}, {inSameRegion});

  bool hasChainedDotOps = false;
  for (auto *curOp : slice) {
    if (isa<DotOp>(curOp) && curOp != dotOp) {
      auto curType = cast<DotOp>(curOp).getType();
      if (curType.getRank() != rank)
        continue;
      if (auto nvmmaLayout =
              dyn_cast<NvidiaMmaLayoutAttr>(curType.getEncoding()))
        return nvmmaLayout.getWarpPerCTA();
      hasChainedDotOps = true;
    }
  }
  if (hasChainedDotOps) {
    if (shape[0] >= shape[1])
      return {numWarps, 1};
    else
      return {1, numWarps};
  }

  SmallVector<int, 4> warpPerCTA(rank, 1);
  while (warpPerCTA[0] * warpPerCTA[1] < numWarps) {
    if (shape[0] / warpPerCTA[0] / 16 >= shape[1] / warpPerCTA[1] / 16) {
      if (warpPerCTA[0] < shape[0] / 16) {
        warpPerCTA[0] *= 2;
        continue;
      }
    }
    warpPerCTA[1] *= 2;
  }
  return warpPerCTA;
}

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
static int getOriginalBitWidth(Value value) {
  auto origBitWidth = getIntOrFloatBitWidth(value.getType());

  SetVector<Operation *> slice;
  BackwardSliceOptions options;
  options.omitBlockArguments = true;
  auto filter = [](Operation *op) {
    if (op->getNumOperands() != 1)
      return false;
    return isa<FPToFPOp, ChangeOp>(op) || fromDialect<arith::ArithDialect>(op);
  };
  options.filter = filter;
  getBackwardSlice(value, &slice, options);

  for (auto *op : slice) {
    if (auto operand = op->getOperand(0)) {
      if (auto tensorType = dyn_cast<RankedTensorType>(operand.getType())) {
        auto bitWidth = getIntOrFloatBitWidth(tensorType);
        if (bitWidth != origBitWidth) {
          origBitWidth = std::min(origBitWidth, bitWidth);
          break;
        }
      }
    }
  }
  return origBitWidth;
}

static Value promoteOperand(OpBuilder &builder, Location loc, Value operand,
                            Type elementType) {
  return builder.create<FPToFPOp>(
      loc, cloneWith(cast<RankedTensorType>(operand.getType()), elementType),
      operand);
}

static void decomposeMixedModeDotOp(ModuleOp module, int nvidiaCC) {
  module.walk([=](DotOp dotOp) {
    OpBuilder builder(dotOp);
    auto loc = dotOp.getLoc();
    auto lhsType = dotOp.getLhs().getType();
    auto resultType = dotOp.getType();
    Type elementType;
    auto nvmmaLayout = dyn_cast<NvidiaMmaLayoutAttr>(resultType.getEncoding());
    if (nvmmaLayout) {
      bool isNativeF8 = lhsType.getElementType().isFloat8E5M2() ||
                        lhsType.getElementType().isFloat8E5M2FNUZ() ||
                        lhsType.getElementType().isFloat8E4M3FNUZ();
      if (!isNativeF8 || nvidiaCC == 89)
        return;
      elementType = builder.getF16Type();
    } else {
      if (lhsType.getElementType() == resultType.getElementType())
        return;
      elementType = resultType.getElementType();
    }
    auto newLhs = promoteOperand(builder, loc, dotOp.getLhs(), elementType);
    auto newRhs = promoteOperand(builder, loc, dotOp.getRhs(), elementType);
    dotOp.setOperand(0, newLhs);
    dotOp.setOperand(1, newRhs);
  });
}

namespace {
class ChangeFmaDotOpToMmaDotOp : public OpRewritePattern<DotOp> {
public:
  using OpRewritePattern::OpRewritePattern;

  virtual LogicalResult
  matchAndRewrite(DotOp op, PatternRewriter &rewriter) const override {
    auto lhsType = op.getLhs().getType();
    auto rhsType = op.getRhs().getType();
    auto resultType = op.getType();
    auto precision = op.getDotPrecision();
    if (!resultType.getEncoding() || hasLayout<NvidiaMmaLayoutAttr>(resultType))
      return failure();
    if (lhsType.getElementType().isF32() && rhsType.getElementType().isF32() &&
        precision == DotPrecision::IEEE)
      return failure();

    auto numWarps = KgpuDialect::getNumWarps(op->getParentOfType<ModuleOp>());
    auto warpPerCTA = getWarpPerCTA(op, numWarps);
    auto nvmmaLayout =
        NvidiaMmaLayoutAttr::get(resultType.getContext(), warpPerCTA);
    auto accumType = cloneWith(resultType, nvmmaLayout);
    auto accum = op.getAccum();
    accum = rewriter.create<ChangeOp>(accum.getLoc(), accumType, accum);

    auto lhs = op.getLhs();
    auto rhs = op.getRhs();
    auto bitWidth =
        std::min(getOriginalBitWidth(lhs), getOriginalBitWidth(rhs));
    auto lhsLayout = DotOpLoadLayoutAttr::get(lhsType.getContext(), nvmmaLayout,
                                              0, bitWidth);
    lhsType = cloneWith(lhsType, lhsLayout);
    lhs = rewriter.create<ChangeOp>(lhs.getLoc(), lhsType, lhs);
    auto rhsLayout = DotOpLoadLayoutAttr::get(rhsType.getContext(), nvmmaLayout,
                                              1, bitWidth);
    rhsType = cloneWith(rhsType, rhsLayout);
    rhs = rewriter.create<ChangeOp>(rhs.getLoc(), rhsType, rhs);

    auto newOp =
        rewriter.create<DotOp>(op.getLoc(), lhs, rhs, accum, precision);
    rewriter.replaceOpWithNewOp<ChangeOp>(op, resultType, newOp);
    return success();
  }
};

#define GEN_PASS_DEF_KGPUACCELERATEMATMUL
#include "kapy/Dialect/Kgpu/Transform/Passes.h.inc"

class KgpuAccelerateMatmulPass
    : public impl::KgpuAccelerateMatmulBase<KgpuAccelerateMatmulPass> {
public:
  virtual void runOnOperation() override {
    auto module = getOperation();
    auto *context = &getContext();

    RewritePatternSet patterns(context);
    patterns.add<ChangeFmaDotOpToMmaDotOp>(context);

    if (failed(applyPatternsAndFoldGreedily(module, std::move(patterns))))
      signalPassFailure();

    decomposeMixedModeDotOp(module, KgpuDialect::getNvidiaCC(module));
  }
};
} // namespace

std::unique_ptr<Pass> kapy::createKgpuAccelerateMatmulPass() {
  return std::make_unique<KgpuAccelerateMatmulPass>();
}
