//===- TypeConverter.cpp ----------------------------------------*- C++ -*-===//
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

#include "kapy/Conversion/KapyToKgpu/TypeConverter.h"
#include "kapy/Analysis/Layout.h"
#include "kapy/Dialect/Kapy/IR/Kapy.h"
#include "kapy/Dialect/Kgpu/IR/Kgpu.h"

using namespace mlir;
using namespace mlir::kapy;

KgpuTypeConverter::KgpuTypeConverter(MLIRContext *context, int64_t numWarps)
    : context(context), numWarps(numWarps) {
  addConversion([](Type type) { return type; });

  addConversion([this](RankedTensorType tensorType) {
    // RankedTensorType with layout are already in the right format.
    if (tensorType.getEncoding())
      return tensorType;
    auto shape = tensorType.getShape();
    SmallVector<int64_t, 4> loops(shape.size(), 1);
    auto regisLayout = getRegistersLayout(this->context, shape, this->numWarps);
    return cloneWith(tensorType, regisLayout);
  });

  addTargetMaterialization([](OpBuilder &builder, RankedTensorType tensorType,
                              ValueRange values, Location loc) {
    return std::optional<Value>(
        builder.create<ChangeOp>(loc, tensorType, values));
  });
}
