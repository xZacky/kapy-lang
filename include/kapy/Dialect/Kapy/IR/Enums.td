//===- Enums.td --------------------------------------------*- tablegen -*-===//
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

#ifndef KAPY_DIALECT_KAPY_IR_ENUMS
#define KAPY_DIALECT_KAPY_IR_ENUMS

include "mlir/IR/EnumAttr.td"

class Kapy_EnumAttr<string name, list<I32EnumAttrCase> cases>
    : I32EnumAttr<name, "", cases> { let cppNamespace = "mlir::kapy"; }

def Kapy_CacheModifierAttr : Kapy_EnumAttr<
  "CacheModifier",
  [
    I32EnumAttrCase<"None", 0, "none">,
    I32EnumAttrCase<"CA", 1, "ca">,
    I32EnumAttrCase<"CG", 2, "cg">,
    I32EnumAttrCase<"WB", 3, "wb">,
    I32EnumAttrCase<"CS", 4, "cs">,
    I32EnumAttrCase<"WT", 5, "wt">
  ]
>;

def Kapy_EvictPriorityAttr : Kapy_EnumAttr<
  "EvictPriority",
  [
    I32EnumAttrCase<"Normal", 0, "normal">,
    I32EnumAttrCase<"EvictFirst", 1, "evict_first">,
    I32EnumAttrCase<"EvictLast", 2, "ecict_last">
  ]
>;

def Kapy_AtomicRMWKindAttr : Kapy_EnumAttr<
  "AtomicRMWKind",
  [
    I32EnumAttrCase<"And", 1, "and">,
    I32EnumAttrCase<"Or", 2, "or">,
    I32EnumAttrCase<"Xor", 3, "xor">,
    I32EnumAttrCase<"Add", 4, "add">,
    I32EnumAttrCase<"Fadd", 5, "fadd">,
    I32EnumAttrCase<"Max", 6, "max">,
    I32EnumAttrCase<"Min", 7, "min">,
    I32EnumAttrCase<"Umax", 8, "umax">,
    I32EnumAttrCase<"Umin", 9, "umin">,
    I32EnumAttrCase<"Xchg", 10, "xchg">
  ]
>;

def Kapy_MemSemanticAttr : Kapy_EnumAttr<
  "MemSemantic",
  [
    I32EnumAttrCase<"Relaxed", 1, "relaxed">,
    I32EnumAttrCase<"Acquire", 2, "acquire">,
    I32EnumAttrCase<"Release", 3, "release">,
    I32EnumAttrCase<"AcquireRelease", 4, "acquire_release">
  ]
>;

def Kapy_ProgramAxisAttr : Kapy_EnumAttr<
  "ProgramAxis",
  [
    I32EnumAttrCase<"X", 0, "x">,
    I32EnumAttrCase<"Y", 1, "y">,
    I32EnumAttrCase<"Z", 2, "z">
  ]
>;

def Kapy_RoundingModeAttr : Kapy_EnumAttr<
  "RoundingMode",
  [
    I32EnumAttrCase<"RTZ", 0, "rtz">,
    I32EnumAttrCase<"RTNE", 1, "rtne">
  ]
>;

def Kapy_PropagateNanAttr : Kapy_EnumAttr<
  "PropagateNan",
  [
    I32EnumAttrCase<"None", 0, "none">,
    I32EnumAttrCase<"All", 1, "all">
  ]
>;

def Kapy_DotPrecisionAttr : Kapy_EnumAttr<
  "DotPrecision",
  [
    I32EnumAttrCase<"IEEE", 0, "ieee">,
    I32EnumAttrCase<"TF32", 1, "tf32">,
    I32EnumAttrCase<"TF32x3", 2, "tf32x3">
  ]
>;

#endif // KAPY_DIALECT_KAPY_IR_ENUMS
