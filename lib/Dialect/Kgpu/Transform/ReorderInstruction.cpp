//===- ReorderInstruction.cpp -----------------------------------*- C++ -*-===//
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
#include "kapy/Dialect/Kgpu/IR/Kgpu.h"
#include "kapy/Dialect/Kgpu/Transform/Passes.h"
#include "mlir/IR/Dominance.h"

using namespace mlir;
using namespace mlir::kapy;

static Operation *getFirstUser(Operation *op) {
  std::vector<Operation *> useOps;
  for (auto *useOp : op->getUsers())
    if (auto *ancestor = op->getBlock()->findAncestorOpInBlock(*useOp))
      useOps.push_back(ancestor);
  auto isBefore = [](Operation *op0, Operation *op1) {
    return op0->isBeforeInBlock(op1);
  };
  auto minOpIt = std::min_element(useOps.begin(), useOps.end(), isBefore);
  return minOpIt != useOps.end() ? *minOpIt : nullptr;
}

static bool willIncreaseRegisterPressure(Operation *op) {
  if (isa<LocalLoadOp>(op))
    return true;
  auto changeOp = dyn_cast<ChangeOp>(op);
  if (!changeOp)
    return false;
  if (isa<MmOperandLayoutAttr>(changeOp.getType().getEncoding()))
    return true;
  return false;
}

namespace {

#define GEN_PASS_DEF_KGPUREORDERINSTRUCTION
#include "kapy/Dialect/Kgpu/Transform/Passes.h.inc"

class KgpuReorderInstructionPass
    : public impl::KgpuReorderInstructionBase<KgpuReorderInstructionPass> {
public:
  virtual void runOnOperation() override {
    auto module = getOperation();

    // Sink changes after the last local_free before the first use ancestor in
    // its block.
    module.walk([&](ChangeOp changeOp) {
      auto *firstUser = getFirstUser(changeOp);
      for (auto it = Block::iterator(changeOp); &*it != firstUser; ++it)
        if (isa<LocalFreeOp>(&*it))
          changeOp->moveAfter(&*it);
    });

    // Sink changes into loops when they will increase register pressure.
    DenseMap<Operation *, Operation *> opsToMove;
    module.walk([&](Operation *op) {
      if (!willIncreaseRegisterPressure(op))
        return;
      auto userBegin = op->user_begin();
      auto userEnd = op->user_end();
      if (std::distance(userBegin, userEnd) != 1)
        return;
      if (userBegin->getParentOfType<scf::ForOp>() ==
          op->getParentOfType<scf::ForOp>())
        return;
      opsToMove.insert({op, *userBegin});
    });
    for (auto it : opsToMove)
      it.first->moveBefore(it.second);
    opsToMove.clear();

    // Move local_alloc(load) immediately after dependent load.
    module.walk([&](LocalAllocOp allocOp) {
      if (!allocOp.getOperand())
        return;
      auto *defOp = allocOp.getOperand().getDefiningOp();
      if (!defOp)
        return;
      allocOp->moveAfter(defOp);
    });

    // Move Transpose just after their defining operation.
    module.walk([&](TransposeOp transposeOp) {
      auto *defOp = transposeOp.getOperand().getDefiningOp();
      if (!defOp)
        return;
      transposeOp->moveAfter(defOp);
    });

    // Move matmul lhs load after rhs load.
    DominanceInfo domInfo(module);
    module.walk([&](LocalLoadOp loadOp) {
      auto mmopdLayout =
          dyn_cast<MmOperandLayoutAttr>(loadOp.getType().getEncoding());
      if (!mmopdLayout)
        return;
      if (mmopdLayout.getOperandIndex() != 1)
        return;
      auto rhsLoadOp = loadOp;
      if (!rhsLoadOp->hasOneUse())
        return;
      auto matmulOp = dyn_cast<MatmulOp>(*loadOp->getUsers().begin());
      if (!matmulOp)
        return;
      auto lhsLoadOp = matmulOp.getLhs().getDefiningOp<LocalLoadOp>();
      if (!lhsLoadOp)
        return;
      if (!domInfo.dominates(rhsLoadOp.getOperation(),
                             lhsLoadOp.getOperation()))
        return;
      rhsLoadOp->moveAfter(lhsLoadOp);
    });
  }
};

} // namespace

std::unique_ptr<Pass> kapy::createKgpuReorderInstructionPass() {
  return std::make_unique<KgpuReorderInstructionPass>();
}
