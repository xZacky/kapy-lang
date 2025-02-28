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
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/BuiltinTypes.h"

#include "kapy/Dialect/Kapy/IR/Enums.h.inc"

#define GET_ATTRDEF_CLASSES
#include "kapy/Dialect/Kapy/IR/Attrs.h.inc"

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
        return op->emitError("ranked tensor can only have rank 2");
      if (tensorType.isDynamicDim(0) || tensorType.isDynamicDim(1))
        continue;
      auto numElements = tensorType.getNumElements();
      if ((numElements & (numElements - 1)) != 0)
        return op->emitError("number of elements must be power of 2, but ")
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
  if (isa<arith::SelectOp>(op))
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
      auto encoding = cast<EncodingAttr>(tensorType.getEncoding());
      if (encoding.getMemory() != MemorySpace::REGISTER_FILE)
        return op->emitOpError("operand ")
               << operand.getOperandNumber() << " must in register file";
    }
  }
  for (auto result : op->getOpResults()) {
    if (result.getResultNumber() == 0 &&
        (op->hasTrait<OpTrait::ResultInGlobalMemory>() ||
         op->hasTrait<OpTrait::ResultInSharedMemory>()))
      continue;
    if (auto tensorType = dyn_cast<RankedTensorType>(result.getType())) {
      auto encoding = cast<EncodingAttr>(tensorType.getEncoding());
      if (encoding.getMemory() != MemorySpace::REGISTER_FILE)
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
  auto encoding = cast<EncodingAttr>(sourceType.getEncoding());
  if (encoding.getMemory() != MemorySpace::GLOBAL_MEMORY)
    return op->emitOpError("source must in global memory");
  return success();
}

LogicalResult OpTrait::impl::verifyTargetInGlobalMemory(Operation *op) {
  auto target = op->getOperand(1);
  auto targetType = dyn_cast<RankedTensorType>(target.getType());
  if (!targetType)
    return op->emitOpError("target must be a tensor");
  auto encoding = cast<EncodingAttr>(targetType.getEncoding());
  if (encoding.getMemory() != MemorySpace::GLOBAL_MEMORY)
    return op->emitOpError("target must in global memory");
  return success();
}

LogicalResult OpTrait::impl::verifyResultInGlobalMemory(Operation *op) {
  auto result = op->getResult(0);
  auto resultType = dyn_cast<RankedTensorType>(result.getType());
  if (!resultType)
    return op->emitOpError("result must be a tensor");
  auto encoding = cast<EncodingAttr>(resultType.getEncoding());
  if (encoding.getMemory() != MemorySpace::GLOBAL_MEMORY)
    return op->emitOpError("result must in global memory");
  return success();
}

LogicalResult OpTrait::impl::verifySourceInSharedMemory(Operation *op) {
  auto source = op->getOperand(0);
  auto sourceType = dyn_cast<RankedTensorType>(source.getType());
  if (!sourceType)
    return op->emitOpError("source must be a tensor");
  auto encoding = cast<EncodingAttr>(sourceType.getEncoding());
  if (encoding.getMemory() != MemorySpace::SHARED_MEMORY)
    return op->emitOpError("source must in shared memory");
  return success();
}

LogicalResult OpTrait::impl::verifyTargetInSharedMemory(Operation *op) {
  auto target = op->getOperand(1);
  auto targetType = dyn_cast<RankedTensorType>(target.getType());
  if (!targetType)
    return op->emitOpError("target must be a tensor");
  auto encoding = cast<EncodingAttr>(targetType.getEncoding());
  if (encoding.getMemory() != MemorySpace::SHARED_MEMORY)
    return op->emitOpError("target must in shared memory");
  return success();
}

LogicalResult OpTrait::impl::verifyResultInSharedMemory(Operation *op) {
  auto result = op->getResult(0);
  auto resultType = dyn_cast<RankedTensorType>(result.getType());
  if (!resultType)
    return op->emitOpError("result must be a tensor");
  auto encoding = cast<EncodingAttr>(resultType.getEncoding());
  if (encoding.getMemory() != MemorySpace::SHARED_MEMORY)
    return op->emitOpError("result must in shared memory");
  return success();
}

static LogicalResult verifySameLayoutImpl(Type type1, Type type2) {
  auto getLayout = [](Type type) -> Attribute {
    if (auto tensorType = dyn_cast<RankedTensorType>(type))
      return cast<EncodingAttr>(tensorType.getEncoding()).getLayout();
    return nullptr;
  };
  auto layout1 = getLayout(type1);
  auto layout2 = getLayout(type2);
  if (!layout1 || !layout2)
    return success();
  return success(layout1 == layout2);
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
