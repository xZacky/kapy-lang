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

#include "kapy/Analysis/AnalysisUtils.h"
#include "kapy/Dialect/Kapy/IR/Kapy.h"
#include "kapy/Dialect/Kapy/Transforms/Passes.h"
#include "mlir/IR/Dominance.h"
#include "mlir/IR/Matchers.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

using namespace mlir;
using namespace mlir::kapy;

static bool isZero(Value value) {
  if (matchPattern(value, m_Zero()) || matchPattern(value, m_AnyZeroFloat()))
    return true;
  if (auto broadcastOp = value.getDefiningOp<BroadcastOp>())
    if (matchPattern(broadcastOp.getOperand(), m_Zero()) ||
        matchPattern(broadcastOp.getOperand(), m_AnyZeroFloat()))
      return true;
  return false;
}

static void combineSelectOpAndIfOp(ModuleOp module) {
  DominanceInfo domInfo(module);
  auto dominanceRequires = [&](arith::SelectOp selectOp, scf::IfOp ifOp) {
    // IfOp needs to be dominated by the SelectOp.
    if (!domInfo.dominates(selectOp.getOperation(), ifOp.getOperation()))
      return false;
    // IfOp needs to dominate all the SelectOp's users.
    for (auto *user : selectOp.getResult().getUsers())
      if (!domInfo.dominates(ifOp.getOperation(), user))
        return false;
    return true;
  };

  // Go over the SelectOps, look if there is an IfOp with the same condition.
  llvm::MapVector<scf::IfOp, SmallVector<arith::SelectOp>> ifToSelectOps;
  module.walk([&](arith::SelectOp selectOp) {
    auto *block = selectOp->getBlock();
    auto condition = selectOp.getCondition();
    SetVector<Operation *> users(condition.getUsers().begin(),
                                 condition.getUsers().end());
    // Sort the users in topological order.
    users = multiRootTopoSort(users);
    for (auto *user : users) {
      auto ifOp = dyn_cast<scf::IfOp>(user);
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
    builder.create<scf::YieldOp>(loc, operands);
    yieldOp.erase();
  };

  for (auto [ifOp, selectOps] : ifToSelectOps) {
    // Add new return value to the IfOp (and create else block if necessary),
    // then yield the select value in the then block and the else block.
    OpBuilder builder(ifOp);
    auto loc = ifOp.getLoc();
    SmallVector<Type> resultTypes(ifOp.getResultTypes());
    for (auto selectOp : selectOps)
      resultTypes.push_back(selectOp.getResult().getType());
    auto newIfOp =
        builder.create<scf::IfOp>(loc, resultTypes, ifOp.getCondition(), true);
    newIfOp.getThenRegion().takeBody(ifOp.getThenRegion());
    if (ifOp.elseBlock())
      newIfOp.getElseRegion().takeBody(ifOp.getElseRegion());
    else
      newIfOp.getElseBodyBuilder().create<scf::YieldOp>(loc);

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

    unsigned i = 0;
    for (auto result : ifOp.getResults())
      result.replaceAllUsesWith(newIfOp->getResult(i++));
    for (auto selectOp : selectOps) {
      selectOp.replaceAllUsesWith(newIfOp->getResult(i++));
      selectOp.erase();
    }

    ifOp.erase();
  }
}

namespace {

#include "kapy/Dialect/Kapy/Transforms/Combine.cpp.inc"

#define GEN_PASS_DEF_KAPYCOMBINE
#include "kapy/Dialect/Kapy/Transforms/Passes.h.inc"

class KapyCombinePass : public impl::KapyCombineBase<KapyCombinePass> {
public:
  virtual void runOnOperation() override {

    auto *context = &getContext();
    RewritePatternSet patterns(context);
    patterns.add<CombineMatmulOpAsAddIOpLhs>(context);
    patterns.add<CombineMatmulOpAsAddIOpRhs>(context);
    patterns.add<CombineMatmulOpAsAddFOpLhs>(context);
    patterns.add<CombineMatmulOpAsAddFOpRhs>(context);

    auto module = getOperation();
    if (failed(applyPatternsAndFoldGreedily(module, std::move(patterns))))
      signalPassFailure();
  }
};

} // namespace

std::unique_ptr<Pass> kapy::createKapyCombinePass() {
  return std::make_unique<KapyCombinePass>();
}
