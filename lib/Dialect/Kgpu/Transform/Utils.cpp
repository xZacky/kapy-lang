//===- Utils.cpp ------------------------------------------------*- C++ -*-===//
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

#include "kapy/Dialect/Kgpu/Transform/Utils.h"
#include "kapy/Analysis/Layout.h"
#include "kapy/Dialect/Kapy/IR/Kapy.h"
#include "kapy/Dialect/Kgpu/IR/Kgpu.h"

using namespace mlir;
using namespace mlir::kapy;

static bool hasMoreThreadsThanElements(Operation *op) {
  auto memrefType = cast<KapyMemRefType>(op->getOperandTypes()[0]);
  return memrefType.getNumElements() <
         numLanes * getNumWarps(op->getParentOfType<ModuleOp>());
}

bool kapy::isExpensiveMemoryRead(Operation *op) {
  if (!isa<LoadOp, AtomicRMWOp, AtomicCASOp>(op))
    return false;
  return !hasMoreThreadsThanElements(op);
}

bool kapy::isExpensiveMemoryWrite(Operation *op) {
  if (!isa<StoreOp, AtomicRMWOp, AtomicCASOp>(op))
    return false;
  return !hasMoreThreadsThanElements(op);
}

static Attribute inferUseLayout(ReduceOp op, Attribute defLayout) {
  return SliceAxisLayoutAttr::get(op.getContext(), defLayout, op.getAxis());
}

static Attribute inferUseLayout(UnsqueezeOp op, Attribute defLayout) {
  auto sliceLayout = dyn_cast<SliceAxisLayoutAttr>(defLayout);
  if (!sliceLayout || op.getAxis() != sliceLayout.getAxis())
    return Attribute();
  return sliceLayout.getParent();
}

Attribute kapy::inferUseLayout(Operation *op, Attribute defLayout) {
  if (op->hasTrait<OpTrait::SameOperandsAndResultLayout>() ||
      op->hasTrait<OpTrait::Elementwise>() ||
      isa<scf::ForOp, scf::WhileOp, scf::ConditionOp, scf::YieldOp>(op))
    return defLayout;
  if (auto reduceOp = dyn_cast<ReduceOp>(op))
    return ::inferUseLayout(reduceOp, defLayout);
  if (auto unsqueezeOp = dyn_cast<UnsqueezeOp>(op))
    return ::inferUseLayout(unsqueezeOp, defLayout);
  // TODO: Support PermuteOp.
  return Attribute();
}

static Attribute inferDefLayout(ReduceOp op, Attribute useLayout) {
  auto sliceLayout = dyn_cast<SliceAxisLayoutAttr>(useLayout);
  if (!sliceLayout || op.getAxis() != sliceLayout.getAxis())
    return Attribute();
  return sliceLayout.getParent();
}

static Attribute inferDefLayout(UnsqueezeOp op, Attribute useLayout) {
  return SliceAxisLayoutAttr::get(op.getContext(), useLayout, op.getAxis());
}

Attribute kapy::inferDefLayout(Operation *op, Attribute useLayout) {
  if (op->hasTrait<OpTrait::SameOperandsAndResultLayout>() ||
      op->hasTrait<OpTrait::Elementwise>() ||
      isa<scf::ForOp, scf::WhileOp, scf::IfOp>(op))
    return useLayout;
  if (auto reduceOp = dyn_cast<ReduceOp>(op))
    return ::inferDefLayout(reduceOp, useLayout);
  if (auto unsqueezeOp = dyn_cast<UnsqueezeOp>(op))
    return ::inferDefLayout(unsqueezeOp, useLayout);
  // TODO: Support PermuteOp.
  return Attribute();
}

bool kapy::isFreeChangeOp(Operation *op) {
  auto changeOp = dyn_cast<ChangeOp>(op);
  if (!changeOp)
    return false;
  auto operandType = changeOp.getOperand().getType();
  if (auto nvmmaLayout =
          dyn_cast<NvidiaMmaLayoutAttr>(operandType.getEncoding())) {
    auto mmopdLayout =
        dyn_cast<MmOperandLayoutAttr>(changeOp.getType().getEncoding());
    auto regisLayout =
        dyn_cast<RegistersLayoutAttr>(changeOp.getType().getEncoding());
    return isNvidiaMmaToMmOperandShortcut(nvmmaLayout, mmopdLayout) ||
           isNvidiaMmaToRegistersShortcut(nvmmaLayout, regisLayout);
  }
  return false;
}
