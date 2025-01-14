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

static Attribute inferResultLayout(ReduceOp op, Attribute operandLayout) {
  return AxisSliceLayoutAttr::get(op.getContext(), operandLayout, op.getAxis());
}

static Attribute inferResultLayout(UnsqueezeOp op, Attribute operandLayout) {
  auto sliceLayout = dyn_cast<AxisSliceLayoutAttr>(operandLayout);
  if (!sliceLayout || op.getAxis() != sliceLayout.getAxis())
    return Attribute();
  return sliceLayout.getParent();
}

Attribute kapy::inferResultLayout(Operation *op, Attribute operandLayout) {
  if (op->hasTrait<OpTrait::SameOperandsAndResultLayout>() ||
      op->hasTrait<OpTrait::Elementwise>() ||
      isa<scf::ForOp, scf::WhileOp, scf::YieldOp, scf::ConditionOp>(op))
    return operandLayout;
  if (auto reduceOp = dyn_cast<ReduceOp>(op))
    return ::inferResultLayout(reduceOp, operandLayout);
  if (auto unsqueezeOp = dyn_cast<UnsqueezeOp>(op))
    return ::inferResultLayout(unsqueezeOp, operandLayout);
  // TODO: Support PermuteOp.
  return Attribute();
}

static Attribute inferOperandLayout(ReduceOp op, Attribute resultLayout) {
  auto sliceLayout = dyn_cast<AxisSliceLayoutAttr>(resultLayout);
  if (!sliceLayout || op.getAxis() != sliceLayout.getAxis())
    return Attribute();
  return sliceLayout.getParent();
}

static Attribute inferOperandLayout(UnsqueezeOp op, Attribute resultLayout) {
  return AxisSliceLayoutAttr::get(op.getContext(), resultLayout, op.getAxis());
}

Attribute kapy::inferOperandLayout(Operation *op, Attribute resultLayout) {
  if (op->hasTrait<OpTrait::SameOperandsAndResultLayout>() ||
      op->hasTrait<OpTrait::Elementwise>() ||
      isa<scf::ForOp, scf::WhileOp, scf::IfOp>(op))
    return resultLayout;
  if (auto reduceOp = dyn_cast<ReduceOp>(op))
    return ::inferOperandLayout(reduceOp, resultLayout);
  if (auto unsqueezeOp = dyn_cast<UnsqueezeOp>(op))
    return ::inferOperandLayout(unsqueezeOp, resultLayout);
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
    auto fragsLayout =
        dyn_cast<FragmentsLayoutAttr>(changeOp.getType().getEncoding());
    return isNvidiaMmaToMmOperandShortcut(nvmmaLayout, mmopdLayout) ||
           isNvidiaMmaToFragmentsShortcut(nvmmaLayout, fragsLayout);
  }
  return false;
}
