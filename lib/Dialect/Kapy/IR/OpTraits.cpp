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
#include "kapy/Dialect/Kapy/IR/Kapy.h"

using namespace mlir;
using namespace mlir::kapy;

static SmallVector<Type> getOperandsAndResultType(Operation *op) {
  auto types = llvm::to_vector(op->getOperandTypes());
  types.append(op->getResultTypes().begin(), op->getResultTypes().end());
  return types;
}

static LogicalResult verifyValidTensorShapeImpl(Operation *op) {
  for (auto type : getOperandsAndResultType(op)) {
    if (auto tensorType = dyn_cast<RankedTensorType>(type)) {
      if (tensorType.getRank() != 2)
        return op->emitOpError("ranked tensor can only have rank 2");
      if (tensorType.isDynamicDim(0) || tensorType.isDynamicDim(1))
        continue;
      auto numElements = tensorType.getNumElements();
      if ((numElements & (numElements - 1)) != 0)
        return op->emitOpError("number of elements must be power of 2, but ")
               << *op << " has " << numElements << " doesn't follow the rule";
      continue;
    }
  }
  return success();
}

LogicalResult OpTrait::impl::verifyValidTensorShape(Operation *op) {
  bool noInvalid = true;
  op->walk([&](Operation *op) {
    if (failed(verifyValidTensorShapeImpl(op)))
      noInvalid = false;
  });
  return success(noInvalid);
}

static LogicalResult verifyValidMemorySpaceImpl(Operation *op) {
  if (isa<CallOp, ReturnOp>(op) || isa<scf::SCFDialect>(op->getDialect()))
    return success();
  if (auto selectOp = dyn_cast<arith::SelectOp>(op))
    if (!isa<RankedTensorType>(selectOp.getCondition().getType()))
      return success();
  for (auto &operand : op->getOpOperands()) {
    if (operand.getOperandNumber() == 0 &&
        (op->hasTrait<OpTrait::SourceInGlobalMemory>() ||
         op->hasTrait<OpTrait::SourceInSharedMemory>()))
      continue;
    if (operand.getOperandNumber() == 1 &&
        (op->hasTrait<OpTrait::TargetInGlobalMemory>() ||
         op->hasTrait<OpTrait::TargetInSharedMemory>()))
      continue;
    if (auto tensorType = dyn_cast<RankedTensorType>(operand.get().getType())) {
      if (!inRegisterFile(tensorType))
        return op->emitOpError("operand ")
               << operand.getOperandNumber() << " must in register file";
    }
  }
  for (auto result : op->getResults()) {
    if (result.getResultNumber() == 0 &&
        (op->hasTrait<OpTrait::ResultInGlobalMemory>() ||
         op->hasTrait<OpTrait::ResultInSharedMemory>()))
      continue;
    if (auto tensorType = dyn_cast<RankedTensorType>(result.getType())) {
      if (!inRegisterFile(tensorType))
        return op->emitOpError("result ")
               << result.getResultNumber() << " must in register file";
    }
  }
  return success();
}

LogicalResult OpTrait::impl::verifyValidMemorySpace(Operation *op) {
  bool noInvalid = true;
  op->walk([&](Operation *op) {
    if (failed(verifyValidMemorySpaceImpl(op)))
      noInvalid = false;
  });
  return success(noInvalid);
}

LogicalResult OpTrait::impl::verifySourceInGlobalMemory(Operation *op) {
  auto source = op->getOperand(0);
  auto sourceType = dyn_cast<RankedTensorType>(source.getType());
  if (!sourceType)
    return op->emitOpError("source must be a tensor");
  if (!inGlobalMemory(sourceType))
    return op->emitOpError("source must in global memory");
  return success();
}

LogicalResult OpTrait::impl::verifyTargetInGlobalMemory(Operation *op) {
  auto target = op->getOperand(1);
  auto targetType = dyn_cast<RankedTensorType>(target.getType());
  if (!targetType)
    return op->emitOpError("target must be a tensor");
  if (!inGlobalMemory(targetType))
    return op->emitOpError("target must in global memory");
  return success();
}

LogicalResult OpTrait::impl::verifyResultInGlobalMemory(Operation *op) {
  auto result = op->getResult(0);
  auto resultType = dyn_cast<RankedTensorType>(result.getType());
  if (!resultType)
    return op->emitOpError("result must be a tensor");
  if (!inGlobalMemory(resultType))
    return op->emitOpError("result must in global memory");
  return success();
}

LogicalResult OpTrait::impl::verifySourceInSharedMemory(Operation *op) {
  auto source = op->getOperand(0);
  auto sourceType = dyn_cast<RankedTensorType>(source.getType());
  if (!sourceType)
    return op->emitOpError("source must be a tensor");
  if (!inSharedMemory(sourceType))
    return op->emitOpError("source must in shared memory");
  return success();
}

LogicalResult OpTrait::impl::verifyTargetInSharedMemory(Operation *op) {
  auto target = op->getOperand(1);
  auto targetType = dyn_cast<RankedTensorType>(target.getType());
  if (!targetType)
    return op->emitOpError("target must be a tensor");
  if (!inSharedMemory(targetType))
    return op->emitOpError("target must in shared memory");
  return success();
}

LogicalResult OpTrait::impl::verifyResultInSharedMemory(Operation *op) {
  auto result = op->getResult(0);
  auto resultType = dyn_cast<RankedTensorType>(result.getType());
  if (!resultType)
    return op->emitOpError("result must be a tensor");
  if (!inSharedMemory(resultType))
    return op->emitOpError("result must in shared memory");
  return success();
}

static LogicalResult verifySameLayoutImpl(Type type0, Type type1) {
  auto getLayout = [](Type type) {
    if (auto tensorType = dyn_cast<RankedTensorType>(type))
      return cast<EncodingAttr>(tensorType.getEncoding()).getLayout();
    return Attribute();
  };
  auto layout0 = getLayout(type0);
  auto layout1 = getLayout(type1);
  if (!layout0 || !layout1)
    return success();
  return success(layout0 == layout1);
}

LogicalResult OpTrait::impl::verifySameOperandsLayout(Operation *op) {
  if (op->getNumOperands() == 0)
    return success();

  auto types = op->getOperandTypes();
  for (auto type : types)
    if (failed(verifySameLayoutImpl(types[0], type)))
      return op->emitOpError("requires same layout for all operands");

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
