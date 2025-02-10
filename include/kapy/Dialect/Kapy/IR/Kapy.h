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

#define GET_TYPEDEF_CLASSES
#include "kapy/Dialect/Kapy/IR/Types.h.inc"

#define GET_OP_CLASSES
#include "kapy/Dialect/Kapy/IR/Ops.h.inc"

namespace mlir {
namespace kapy {

/// This class representes the global memory resource.
class GlobalMemory : public SideEffects::Resource::Base<GlobalMemory> {
public:
  virtual StringRef getName() override { return "<GlobalMemory>"; }
};

class KapyLayoutInterface : public DialectInterface::Base<KapyLayoutInterface> {
public:
  KapyLayoutInterface(Dialect *dialect) : Base(dialect) {}

  virtual Attribute inferTransposeOpLayout(Attribute sourceLayout) const = 0;

  virtual LogicalResult verifyMatmulOpLayouts(MatmulOp matmulOp) const = 0;
};

/// Get the integer or floating-point bit width for the given Type, if it is a
/// ShapedType, get its element bit width.
unsigned getIntOrFloatBitWidth(Type type);

/// Return true if this operation is global memory read.
bool isGlobalMemoryRead(Operation *op);

/// Return true if this operation is global memory write.
bool isGlobalMemoryWrite(Operation *op);

constexpr char alignmentAttrName[] = "kapy.alignment";

/// Get the alignemnt of the given memory access operation, this should be used
/// after running KapyAnalyzeAlignmentPass.
unsigned getAlignment(Operation *op);

} // namespace kapy
} // namespace mlir

#endif // KAPY_DIALECT_KAPY_IR_KAPY_H
