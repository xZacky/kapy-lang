//===- Kapy.h ---------------------------------------------------*- C++ -*-===//
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

#ifndef KAPY_DIALECT_KAPY_IR_KAPY_H
#define KAPY_DIALECT_KAPY_IR_KAPY_H

#include "kapy/Dialect/Kapy/IR/OpTraits.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlow.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Interfaces/FunctionInterfaces.h"

#include "kapy/Dialect/Kapy/IR/Dialect.h.inc"
#include "kapy/Dialect/Kapy/IR/Enums.h.inc"

#define GET_ATTRDEF_CLASSES
#include "kapy/Dialect/Kapy/IR/Attrs.h.inc"

#define GET_OP_CLASSES
#include "kapy/Dialect/Kapy/IR/Ops.h.inc"

namespace mlir {
namespace kapy {

constexpr char nvidiaCCAttrName[] = "kapy.nvidia_cc";
constexpr char numWarpsAttrName[] = "kapy.num_warps";

constexpr int64_t warpSize = 32;

constexpr char alignmentAttrName[] = "kapy.alignment";

class GlobalMemory : public SideEffects::Resource::Base<GlobalMemory> {
public:
  virtual StringRef getName() override { return "<GlobalMemory>"; }
};

class SharedMemory : public SideEffects::Resource::Base<SharedMemory> {
public:
  virtual StringRef getName() override { return "<SharedMemory>"; }
};

class KapyLayoutInterface : public DialectInterface::Base<KapyLayoutInterface> {
public:
  KapyLayoutInterface(Dialect *dialect) : Base(dialect) {}

  virtual LogicalResult verifyLdMatrixOpLayouts(LdMatrixOp op) const = 0;

  virtual Attribute inferTransposeOpLayout(Attribute sourceLayout) const = 0;

  virtual LogicalResult verifyMatmulOpLayouts(MatmulOp op) const = 0;
};

/// Get the integer or floating-point bit width for the given Type, if it is a
/// ShapedType, get its element bit width.
unsigned getIntOrFloatBitWidth(Type type);

/// Get nvidia compute capability from module attributes.
int64_t getNvidiaCC(ModuleOp module);

/// Get number of warps from module attributes.
int64_t getNumWarps(ModuleOp module);

int64_t getAlignment(Operation *op);

bool isGlobalMemoryRead(Operation *op);

bool isGlobalMemoryWrite(Operation *op);

bool isSharedMemoryRead(Operation *op);

bool isSharedMemoryWrite(Operation *op);

bool inGlobalMemory(RankedTensorType rankedType);

bool inSharedMemory(RankedTensorType rankedType);

bool inRegisterFile(RankedTensorType rankedType);

bool hasLayout(RankedTensorType rankedType);

Attribute getLayout(RankedTensorType rankedType);

template <typename LayoutT> LayoutT getLayout(RankedTensorType rankedType) {
  return dyn_cast<LayoutT>(getLayout(rankedType));
}

RankedTensorType cloneWithShape(RankedTensorType rankedType,
                                ArrayRef<int64_t> shape);

RankedTensorType cloneWithLayout(RankedTensorType rankedType, Attribute layout);

} // namespace kapy
} // namespace mlir

#endif // KAPY_DIALECT_KAPY_IR_KAPY_H
