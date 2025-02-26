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
#include "mlir/IR/BuiltinTypes.h"

#define GET_TYPEDEF_CLASSES
#include "kapy/Dialect/Kapy/IR/Types.h.inc"

using namespace mlir;
using namespace mlir::kapy;

static SmallVector<Type> getOperandsAndResultType(Operation *op) {
  auto types = llvm::to_vector(op->getOperandTypes());
  types.append(op->getResultTypes().begin(), op->getResultTypes().end());
  return types;
}

static LogicalResult verifyValidShapeImpl(Operation *op) {
  for (auto type : getOperandsAndResultType(op)) {
    if (auto tensorType = dyn_cast<RankedTensorType>(type)) {
      if (tensorType.getRank() != 2)
        return op->emitError("ranked tensor can only have rank 2");
      auto numElements = tensorType.getNumElements();
      if ((numElements & (numElements - 1)) != 0)
        return op->emitError("number of elements must be power of 2, but ")
               << *op << " has " << numElements << " doesn't follow the rule";
      continue;
    }
    if (auto sharedType = dyn_cast<SharedMemRefType>(type)) {
      if (sharedType.getRank() != 2)
        return op->emitError("shared memref can only have rank 2");
      auto numElements = sharedType.getNumElements();
      if ((numElements & (numElements - 1)) != 0)
        return op->emitError("number of elements must be power of 2, but ")
               << *op << " has " << numElements << " doesn't follow the rule";
      continue;
    }
    if (auto globalType = dyn_cast<GlobalMemRefType>(type)) {
      if (globalType.getRank() != 2)
        return op->emitError("global memref can only have rank 2");
      continue;
    }
  }
  return success();
}

LogicalResult OpTrait::impl::verifyValidShape(Operation *op) {
  bool noInvalid = true;
  op->walk([&](Operation *op) {
    if (failed(verifyValidShapeImpl(op)))
      noInvalid = false;
  });
  return success(noInvalid);
}

static LogicalResult verifySameLayoutImpl(Type typeA, Type typeB) {
  auto getLayout = [](Type type) -> Attribute {
    if (auto tensorType = dyn_cast<RankedTensorType>(type))
      return tensorType.getEncoding();
    return nullptr;
  };
  auto layoutA = getLayout(typeA);
  auto layoutB = getLayout(typeB);
  if (!layoutA || !layoutB)
    return success();
  return success(layoutA == layoutB);
}

LogicalResult OpTrait::impl::verifySameOperandsLayout(Operation *op) {
  if (op->getNumOperands() == 0)
    return success();

  auto types = op->getOperandTypes();
  for (auto type : types)
    if (failed(verifySameLayoutImpl(types[0], type)))
      return op->emitError("requires same layout for all operands");

  return success();
}

LogicalResult OpTrait::impl::verifySameOperandsAndResultLayout(Operation *op) {
  if (op->getNumOperands() + op->getNumResults() == 0)
    return success();

  auto types = getOperandsAndResultType(op);
  for (auto type : types)
    if (failed(verifySameLayoutImpl(types[0], type)))
      return op->emitOpError(
          "requires same layout for all operands and result");

  return success();
}
