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

// Tensor in Kapy dialect can only be 1d or 2d (with rank 1, 2), and number of
// elements must be power of 2.
def ValidTensorShape : NativeOpTrait<"ValidTensorShape">;

// Operation with this trait requires all its tensor operands having the same
// layout.
def SameOperandsLayout : NativeOpTrait<"SameOperandsLayout">;

// Operation with this trait requires all its tensor operands and result having
// the same layout.
def SameOperandsAndResultLayout : NativeOpTrait<"SameOperandsAndResultLayout">;

// Operations with this trait have single block with a kapy::YieldOp as block
// terminator.
def KapySingleBlockTerminator : SingleBlockImplicitTerminator<"YieldOp">;

#endif // DIALECT_KAPY_IR_OPTRAITS
