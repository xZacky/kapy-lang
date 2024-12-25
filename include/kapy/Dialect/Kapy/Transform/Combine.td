//===- Combine.td ------------------------------------------*- tablegen -*-===//
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

#ifndef KAPY_DIALECT_KAPY_TRANSFORM_COMBINE
#define KAPY_DIALECT_KAPY_TRANSFORM_COMBINE

include "mlir/Dialect/Arith/IR/ArithOps.td"
include "mlir/IR/PatternBase.td"

include "kapy/Dialect/Kapy/IR/Ops.td"

def CombineDotOpAsAddIOpLhs : Pat<
  (Arith_AddIOp
    (Kapy_DotOp:$result $lhs, $rhs, $zero, $precision),
    $accum,
    $overflow
  ),
  (Kapy_DotOp $lhs, $rhs, $accum, $precision, (location $result)),
  [
    (Constraint<CPred<"isZero($0)">> $zero),
    (Constraint<CPred<"$0.hasOneUse()">> $result)
  ]
>;

def CombineDotOpAsAddIOpRhs : Pat<
  (Arith_AddIOp
    $accum,
    (Kapy_DotOp:$result $lhs, $rhs, $zero, $precision),
    $overflow
  ),
  (Kapy_DotOp $lhs, $rhs, $accum, $precision, (location $result)),
  [
    (Constraint<CPred<"isZero($0)">> $zero),
    (Constraint<CPred<"$0.hasOneUse()">> $result)
  ]
>;

def CombineDotOpAsAddFOpLhs : Pat<
  (Arith_AddFOp
    (Kapy_DotOp:$result $lhs, $rhs, $zero, $precision),
    $accum,
    $fastmath
  ),
  (Kapy_DotOp $lhs, $rhs, $accum, $precision, (location $result)),
  [
    (Constraint<CPred<"isZero($0)">> $zero),
    (Constraint<CPred<"$0.hasOneUse()">> $result)
  ]
>;

def CombineDotOpAsAddFOpRhs : Pat<
  (Arith_AddFOp
    $accum,
    (Kapy_DotOp:$result $lhs, $rhs, $zero, $precision),
    $fastmath
  ),
  (Kapy_DotOp $lhs, $rhs, $accum, $precision, (location $result)),
  [
    (Constraint<CPred<"isZero($0)">> $zero),
    (Constraint<CPred<"$0.hasOneUse()">> $result)
  ]
>;

def CombineTwoMovMemRefOps : Pat<
  (Kapy_MovMemRefOp (Kapy_MovMemRefOp $source, $offset0), $offset1),
  (Kapy_MovMemRefOp $source, (Arith_AddIOp $offset0, $offset1, Arith_IOFnone)),
  [
    (Constraint<CPred<"isCombinable($0, $1)">> $offset0, $offset1)
  ]
>;

#endif // KAPY_DIALECT_KAPY_TRANSFORM_COMBINE
