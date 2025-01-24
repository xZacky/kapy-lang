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

class SharedMemory : public SideEffects::Resource::Base<SharedMemory> {
public:
  virtual StringRef getName() override { return "<SharedMemory>"; }
};

constexpr char nvidiaCCAttrName[] = "kgpu.nvidia_cc";
constexpr char numWarpsAttrName[] = "kgpu.num_warps";
constexpr char shmemNeededAttrName[] = "kgpu.shmem_needed";
constexpr char shmemOffsetAttrName[] = "kgpu.shmem_offset";
constexpr int64_t numLanes = 32;

int64_t getNvidiaCC(ModuleOp module);
int64_t getNumWarps(ModuleOp module);
int64_t getSharedMemNeeded(ModuleOp module);
int64_t getSharedMemOffset(Operation *op);

bool supportNvidiaMma(MatmulOp matmulOp);
bool supportNvidiaMma(Type elementType);

bool isNvidiaMmaToMmOperandShortcut(NvidiaMmaLayoutAttr nvmmaLayout,
                                    MmOperandLayoutAttr mmopdLayout);
bool isNvidiaMmaToFragmentsShortcut(NvidiaMmaLayoutAttr nvmmaLayout,
                                    FragmentsLayoutAttr fragsLayout);

bool isLayoutShortcut(Attribute srcLayout, Attribute dstLayout);

bool isExpensiveMemoryRead(Operation *op);
bool isExpensiveMemoryWrite(Operation *op);

/// Get an AffineMap from tensor indices to the minimum thread id hold it.
AffineMap getTensorMap(ArrayRef<int64_t> shape, Attribute layout);

std::string getLayoutString(RankedTensorType tensorType);

} // namespace kapy
} // namespace mlir

#endif // KAPY_DIALECT_KGPU_IR_KGPU_H
