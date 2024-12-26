//===- OpHelpers.h ----------------------------------------------*- C++ -*-===//
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

#ifndef KAPY_ANALYSIS_OPHELPERS_H
#define KAPY_ANALYSIS_OPHELPERS_H

#include "mlir/IR/Operation.h"

namespace mlir {
namespace kapy {
class ReduceOpHelper {
public:
  explicit ReduceOpHelper(Operation *op) : operation(op) {}

  int getAxis() const;
  RankedTensorType getOperandType() const;

  bool isSupportedLayout() const;
  bool isWarpSynchronous() const;

  SmallVector<int, 4> getScratchShape() const;
  int getScratchSizeInBytes() const;

private:
  Operation *operation;
};

class ScanOpHelper {
public:
  explicit ScanOpHelper(Operation *op) : operation(op) {}

  int getAxis() const;
  bool getReverse() const;
  RankedTensorType getOperandType() const;

  bool isSupportedLayout() const;
  bool isWarpSynchronous() const;

  SmallVector<int, 4> getScratchShape() const;
  int getScratchSizeInBytes() const;

private:
  Operation *operation;
};

class ChangeOpHelper {
public:
  explicit ChangeOpHelper(Operation *op) : operation(op) {}

  RankedTensorType getOperandType() const;
  RankedTensorType getResultType() const;

  bool isWarpSynchronous() const;

  SmallVector<int, 4> getScratchShape() const;
  int getScratchSizeInBytes() const;

private:
  Operation *operation;
};
} // namespace kapy
} // namespace mlir

#endif // KAPY_ANALYSIS_OPHELPERS_H
