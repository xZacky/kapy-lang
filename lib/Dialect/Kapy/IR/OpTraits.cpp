//===- OpTraits.cpp ---------------------------------------------*- C++ -*-===//
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

#include "kapy/Dialect/Kapy/IR/OpTraits.h"
#include "kapy/Dialect/Kapy/IR/Utils.h"

#include "mlir/IR/BuiltinTypes.h"

using namespace mlir;
using namespace mlir::kapy;

LogicalResult OpTrait::impl::verifyTensorShape(Operation *op) {
  for (auto type : op->getOperandTypes()) {
    if (auto tensorType = dyn_cast<RankedTensorType>(type)) {
      if (tensorType.getRank() == 0)
        return op->emitError("0d tensor is not allowed");
      auto numElements = product(tensorType.getShape());
      if (numElements > maxElements)
        return op->emitError("maximum allowed number of elements is")
               << maxElements << ", but " << *op << " has more than that";
      if ((numElements & (numElements - 1)) != 0)
        return op->emitError("number of elements must be power of 2, but ")
               << *op << " has " << numElements << " doesn't follow the rule";
    }
  }
  for (auto type : op->getResultTypes()) {
    if (auto tensorType = dyn_cast<RankedTensorType>(type)) {
      if (tensorType.getRank() == 0)
        return op->emitError("0d tensor is not allowed");
      auto numElements = product(tensorType.getShape());
      if (numElements > maxElements)
        return op->emitError("maximum allowed number of elements is")
               << maxElements << ", but " << *op << " has more than that";
      if ((numElements & (numElements - 1)) != 0)
        return op->emitError("number of elements must be power of 2, but ")
               << *op << " has " << numElements << " doesn't follow the rule";
    }
  }
  return success();
}

static LogicalResult verifySameLayout(Type type0, Type type1) {
  auto getLayout = [](Type type) {
    if (auto tensorType = dyn_cast<RankedTensorType>(type))
      return tensorType.getEncoding();
    return Attribute();
  };
  auto layout0 = getLayout(type0);
  auto layout1 = getLayout(type1);
  if (!layout0 || !layout1)
    return success();
  return success(layout0 == layout1);
}

LogicalResult OpTrait::impl::verifySameOperandsLayout(Operation *op) {
  if (failed(verifyAtLeastNOperands(op, 1)))
    return failure();

  auto firstType = op->getOperand(0).getType();
  for (auto type : op->getOperandTypes())
    if (failed(verifySameLayout(firstType, type)))
      return op->emitOpError("requires same layout for all operands");

  return success();
}

LogicalResult OpTrait::impl::verifySameOperandsAndResultLayout(Operation *op) {
  if (op->getNumOperands() == 0)
    return success();

  if (failed(verifyAtLeastNResults(op, 1)))
    return failure();

  auto firstType = op->getOperand(0).getType();
  for (auto type : op->getResultTypes())
    if (failed(verifySameLayout(firstType, type)))
      return op->emitOpError(
          "requires same layout for all operands and results");

  return verifySameOperandsLayout(op);
}
