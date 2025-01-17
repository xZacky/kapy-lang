//===- OpHelpers.cpp --------------------------------------------*- C++ -*-===//
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

#include "kapy/Analysis/OpHelpers.h"
#include "kapy/Dialect/Kapy/IR/Kapy.h"
#include "kapy/Dialect/Kapy/IR/Utils.h"
#include "kapy/Dialect/Kgpu/IR/Kgpu.h"

using namespace mlir;
using namespace mlir::kapy;

unsigned ReduceOpHelper::getAxis() const {
  return cast<ReduceOp>(operation).getAxis();
}

RankedTensorType ReduceOpHelper::getOperandType() const {
  return cast<ReduceOp>(operation).getOperand().getType();
}

bool ReduceOpHelper::isSupportedLayout() const {
  auto operandLayout = getOperandType().getEncoding();
  return isa<FragmentsLayoutAttr, NvidiaMmaLayoutAttr>(operandLayout);
}

bool ReduceOpHelper::isLaneSynchronous() const {
  // TODO: Implement this.
  return false;
}

bool ReduceOpHelper::isWarpSynchronous() const {
  // TODO: Implement this.
  return false;
}

int64_t ReduceOpHelper::getScratchSizeInBytes() const {
  if (isLaneSynchronous() || isWarpSynchronous())
    return 0;
  auto bitWidth = getIntOrFloatBitWidth(getOperandType());
  // TODO: Compute inter-warp data size.
  return ceilDiv<unsigned>(bitWidth, 8) * product(getOperandType().getShape());
}

RankedTensorType ChangeOpHelper::getOperandType() const {
  return cast<ChangeOp>(operation).getOperand().getType();
}

RankedTensorType ChangeOpHelper::getResultType() const {
  return cast<ChangeOp>(operation).getType();
}

bool ChangeOpHelper::isLaneSynchronous() const {
  // TODO: Implement this.
  return false;
}

bool ChangeOpHelper::isWarpSynchronous() const {
  // TODO: Implement this.
  return false;
}

int64_t ChangeOpHelper::getScratchSizeInBytes() const {
  if (isLaneSynchronous() || isWarpSynchronous())
    return 0;

  auto bitWidth = getIntOrFloatBitWidth(getOperandType());
  // TODO: Compute inter-warp data size.
  return ceilDiv<unsigned>(bitWidth, 8) * product(getOperandType().getShape());
}
