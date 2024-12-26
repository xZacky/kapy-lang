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

namespace {
#define GEN_PASS_DEF_KGPUREORDERINSTRUCTION
#include "kapy/Dialect/Kgpu/Transform/Passes.h.inc"

class KgpuReorderInstructionPass
    : public impl::KgpuReorderInstructionBase<KgpuReorderInstructionPass> {
public:
  virtual void runOnOperation() override {
    auto module = getOperation();
    DominanceInfo domInfo(module);
    // Sink changes after the last free before the first use ancestor in its
    // block.
    module.walk([&](ChangeOp changeOp) {
      auto *firstUseOp = getFirstUse(changeOp);
      for (auto it = Block::iterator(changeOp); &*it != firstUseOp; ++it)
        if (isa<LocalFreeOp>(&*it))
          changeOp->moveAfter(&*it);
    });

    // Sink changes into loops when they will increase register pressure.
    DenseMap<Operation *, Operation *> opsToMove;
    module.walk([&](Operation *op) {
      if (!willIncreaseRegisterPressure(op))
        return;
      if (op->getUsers().begin()->getParentOfType<scf::ForOp>() ==
          op->getParentOfType<scf::ForOp>())
        return;
      opsToMove.insert({op, *op->getUsers().begin()});
    });
    for (auto it : opsToMove)
      it.first->moveBefore(it.second);

    // Move local_alloc(load) immediately after dependent load.
    module.walk([&](LocalAllocOp localAllocOp) {
      if (!localAllocOp.getOperand())
        return;
      auto *defOp = localAllocOp.getOperand().getDefiningOp();
      if (!defOp)
        return;
      localAllocOp->moveAfter(defOp);
    });

    // Move dot op load lhs after changes to dot op load rhs.
    module.walk([&](LocalLoadOp localLoadOp) {
      auto dotldLayout =
          dyn_cast<DotOpLoadLayoutAttr>(localLoadOp.getType().getEncoding());
      if (!dotldLayout)
        return;
      auto index = dotldLayout.getOperandIndex();
      if (index != 1)
        return;
      if (!localLoadOp->hasOneUse())
        return;
      auto dotOp = dyn_cast<DotOp>(*localLoadOp->getUsers().begin());
      if (!dotOp)
        return;
      auto lhsLocalLoadOp = dotOp.getLhs().getDefiningOp<LocalLoadOp>();
      if (!lhsLocalLoadOp)
        return;
      if (!domInfo.dominates(localLoadOp.getOperation(),
                             lhsLocalLoadOp.getOperation()))
        return;
      localLoadOp->moveAfter(lhsLocalLoadOp);
    });
  }

private:
  static Operation *getFirstUse(Operation *op) {
    std::vector<Operation *> useOps;
    for (auto *useOp : op->getUsers())
      if (auto *ancestor = op->getBlock()->findAncestorOpInBlock(*useOp))
        useOps.push_back(ancestor);
    auto minOpIt = std::min_element(useOps.begin(), useOps.end(),
                                    [](Operation *op0, Operation *op1) {
                                      return op0->isBeforeInBlock(op1);
                                    });
    return minOpIt != useOps.end() ? *minOpIt : nullptr;
  }

  static bool willIncreaseRegisterPressure(Operation *op) {
    if (isa<LocalLoadOp>(op))
      return true;
    auto changeOp = dyn_cast<ChangeOp>(op);
    if (!changeOp)
      return false;
    if (isa<DotOpLoadLayoutAttr>(changeOp.getType().getEncoding()))
      return true;
    return false;
  }
};
} // namespace

std::unique_ptr<Pass> kapy::createKgpuReorderInstructionPass() {
  return std::make_unique<KgpuReorderInstructionPass>();
}
