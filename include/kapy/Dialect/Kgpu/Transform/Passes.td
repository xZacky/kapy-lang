//===- Passes.td -------------------------------------------*- tablegen -*-===//
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

#ifndef KAPY_DIALECT_KGPU_TRANSFORM_PASSES
#define KAPY_DIALECT_KGPU_TRANSFORM_PASSES

include "mlir/Pass/PassBase.td"

def KgpuCoalesce : Pass<"kgpu-coalesce", "ModuleOp"> {
  let constructor = "kapy::createKgpuCoalescePass()";

  let dependentDialects = ["kapy::KapyDialect", "kapy::KgpuDialect"];
}

// def KgpuOptimizeLayout : Pass<"kgpu-optimize-layout", "ModuleOp"> {
//   let constructor = "kapy::createKgpuOptimizeLayoutPass()";
//
//   let dependentDialects = ["kapy::KapyDialect", "kapy::KgpuDialect"];
// }

def KgpuOptimizeMatmul : Pass<"kgpu-optimize-matmul", "ModuleOp"> {
  let constructor = "kapy::createKgpuOptimizeMatmulPass()";

  let dependentDialects = ["kapy::KapyDialect", "kapy::KgpuDialect"];
}

def KgpuCacheMatmulOperand : Pass<"kgpu-cache-matmul-operand", "ModuleOp"> {
  let constructor = "kapy::createKgpuCacheMatmulOperandPass()";

  let dependentDialects = ["kapy::KapyDialect", "kapy::KgpuDialect"];
}

def KgpuReorderInstruction : Pass<"kgpu-reorder-instruction", "ModuleOp"> {
  let constructor = "kapy::createKgpuReorderInstructionPass()";

  let dependentDialects = ["kapy::KapyDialect", "kapy::KgpuDialect"];
}

#endif // KAPY_DIALECT_KGPU_TRANSFORM_PASSES
