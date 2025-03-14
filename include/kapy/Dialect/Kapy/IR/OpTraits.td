//===- OpTraits.td -----------------------------------------*- tablegen -*-===//
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

#ifndef KAPY_DIALECT_KAPY_IR_OPTRAITS
#define KAPY_DIALECT_KAPY_IR_OPTRAITS

include "mlir/IR/OpBase.td"

// Tensor in kapy can only be 2d (have rank 2), besides, the number of elements
// should be power of 2.
// Tensor in global memory with unknown shape only need to be 2d.
def ValidTensorShape : NativeOpTrait<"ValidTensorShape">;

// Tensor should in register file unless there are other trait indicates it.
// Control flow operations is ignored.
// SelectOp is checked only if its condition is tensor. 
def ValidMemorySpace : NativeOpTrait<"ValidMemorySpace">;

// Source (operand 0) is a tensor in global memory.
def SourceInGlobalMemory : NativeOpTrait<"SourceInGlobalMemory">;

// Target (operand 1) is a tensor in global memory.
def TargetInGlobalMemory : NativeOpTrait<"TargetInGlobalMemory">;

// Result is a tensor in global memory.
def ResultInGlobalMemory : NativeOpTrait<"ResultInGlobalMemory">;

// Source (operand 0) is a tensor in shared memory.
def SourceInSharedMemory : NativeOpTrait<"SourceInSharedMemory">;

// Target (operand 1) is a tensor in shared memory.
def TargetInSharedMemory : NativeOpTrait<"TargetInSharedMemory">;

// Result is a tensor in shared memory.
def ResultInSharedMemory : NativeOpTrait<"ResultInSharedMemory">;

// Operations with this trait requires all its operands have same layout.
def SameOperandsLayout : NativeOpTrait<"SameOperandsLayout">;

// Operations with this trait requires all its operands and result have same
// layout.
def SameOperandsAndResultLayout : NativeOpTrait<"SameOperandsAndResultLayout">;

#endif // DIALECT_KAPY_IR_OPTRAITS
