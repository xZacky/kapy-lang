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

def CombineMatmulOpAsAddIOpLhs : Pat<
  (Arith_AddIOp
    (Kapy_MatmulOp:$result $lhs, $rhs, $zero, $format),
    $accum,
    $overflow
  ),
  (Kapy_MatmulOp $lhs, $rhs, $accum, $format, (location $result)),
  [
    (Constraint<CPred<"isZero($0)">> $zero),
    (Constraint<CPred<"$0.hasOneUse()">> $result)
  ]
>;

def CombineMatmulOpAsAddIOpRhs : Pat<
  (Arith_AddIOp
    $accum,
    (Kapy_MatmulOp:$result $lhs, $rhs, $zero, $format),
    $overflow
  ),
  (Kapy_MatmulOp $lhs, $rhs, $accum, $format, (location $result)),
  [
    (Constraint<CPred<"isZero($0)">> $zero),
    (Constraint<CPred<"$0.hasOneUse()">> $result)
  ]
>;

def CombineMatmulOpAsAddFOpLhs : Pat<
  (Arith_AddFOp
    (Kapy_MatmulOp:$result $lhs, $rhs, $zero, $format),
    $accum,
    $fastmath
  ),
  (Kapy_MatmulOp $lhs, $rhs, $accum, $format, (location $result)),
  [
    (Constraint<CPred<"isZero($0)">> $zero),
    (Constraint<CPred<"$0.hasOneUse()">> $result)
  ]
>;

def CombineMatmulOpAsAddFOpRhs : Pat<
  (Arith_AddFOp
    $accum,
    (Kapy_MatmulOp:$result $lhs, $rhs, $zero, $format),
    $fastmath
  ),
  (Kapy_MatmulOp $lhs, $rhs, $accum, $format, (location $result)),
  [
    (Constraint<CPred<"isZero($0)">> $zero),
    (Constraint<CPred<"$0.hasOneUse()">> $result)
  ]
>;

def CombineTwoMovMemRefOps : Pat<
  (Kapy_MovMemRefOp:$result1
    (Kapy_MovMemRefOp:$result0 $source, $offset0),
    $offset1
  ),
  (Kapy_MovMemRefOp
    $source,
    (Arith_AddIOp $offset0, $offset1, Arith_IOFnone, (location $result0)),
    (location $result1)
  ),
  [
    (Constraint<CPred<"isCombinable($0, $1)">> $offset0, $offset1)
  ]
>;

#endif // KAPY_DIALECT_KAPY_TRANSFORM_COMBINE
