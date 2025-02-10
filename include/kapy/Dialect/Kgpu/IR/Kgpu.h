//===- Kgpu.h ---------------------------------------------------*- C++ -*-===//
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

#ifndef KAPY_DIALECT_KGPU_IR_KGPU_H
#define KAPY_DIALECT_KGPU_IR_KGPU_H

#include "kapy/Dialect/Kapy/IR/Kapy.h"

#include "kapy/Dialect/Kgpu/IR/Dialect.h.inc"

#define GET_ATTRDEF_CLASSES
#include "kapy/Dialect/Kgpu/IR/Attrs.h.inc"

#define GET_TYPEDEF_CLASSES
#include "kapy/Dialect/Kgpu/IR/Types.h.inc"

#define GET_OP_CLASSES
#include "kapy/Dialect/Kgpu/IR/Ops.h.inc"

namespace mlir {
namespace kapy {

/// This class represents the shared memory resource.
class SharedMemory : public SideEffects::Resource::Base<SharedMemory> {
public:
  virtual StringRef getName() override { return "<SharedMemory>"; }
};

constexpr int64_t numLanes = 32;
constexpr int64_t maxWarps = 32;

constexpr char nvidiaCCAttrName[] = "kgpu.nvidia_cc";
constexpr char numWarpsAttrName[] = "kgpu.num_warps";

constexpr char sharedNeededAttrName[] = "kgpu.shared_needed";
constexpr char sharedOffsetAttrName[] = "kgpu.shared_offset";

/// Get nvidia compute capability from module attributes, this should be used
/// after running ConvertKapyToKgpuPass.
int64_t getNvidiaCC(ModuleOp module);

/// Get the number of warps from module attributes, this should be used after
/// running ConvertKapyToKgpuPass.
int64_t getNumWarps(ModuleOp module);

/// Get shared memory needed from module attributes, this should be used after
/// running KgpuAllocateSharedMemoryPass.
int64_t getSharedMemoryNeeded(ModuleOp module);

/// Get shared memory offset for an operation, this should be used after running
/// KgpuAllocateSharedMemoryPass.
/// For operations do not use shared memory, always returns 0.
int64_t getSharedMemoryOffset(Operation *op);

/// Get a string to show how we distribute elements to threads.
std::string getTensorLayoutString(RankedTensorType tensorType);

} // namespace kapy
} // namespace mlir

#endif // KAPY_DIALECT_KGPU_IR_KGPU_H
