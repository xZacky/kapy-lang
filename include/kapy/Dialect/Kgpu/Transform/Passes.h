//===- Passes.h -------------------------------------------------*- C++ -*-===//
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

#ifndef KAPY_DIALECT_KGPU_TRANSFORM_PASSES_H
#define KAPY_DIALECT_KGPU_TRANSFORM_PASSES_H

#include "mlir/Pass/Pass.h"

namespace mlir {
namespace kapy {

std::unique_ptr<Pass> createKgpuCoalescePass();
std::unique_ptr<Pass> createKgpuOptimizeLayoutPass();
std::unique_ptr<Pass> createKgpuOptimizeMatmulPass();
std::unique_ptr<Pass> createKgpuCacheMatmulOperandPass();
std::unique_ptr<Pass> createKgpuReorderInstructionPass();

} // namespace kapy

#define GEN_PASS_REGISTRATION
#include "kapy/Dialect/Kgpu/Transform/Passes.h.inc"

} // namespace mlir

#endif // KAPY_DIALECT_KGPU_TRANSFORM_PASSES_H
