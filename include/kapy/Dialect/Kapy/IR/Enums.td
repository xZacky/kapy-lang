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

class Kapy_EnumAttr<
  string name, string description, list<I32EnumAttrCase> cases
> : I32EnumAttr<name, description, cases> {
  let cppNamespace = "mlir::kapy";
}

def Kapy_MemorySpaceAttr : Kapy_EnumAttr<
  "MemorySpace",
  "memory space where the tensor is stored",
  [
    I32EnumAttrCase<"GLOBAL_MEMORY", 0, "global_memory">,
    I32EnumAttrCase<"SHARED_MEMORY", 1, "shared_memory">,
    I32EnumAttrCase<"REGISTER_FILE", 2, "register_file">
  ]
>;

def Kapy_PaddingOptionAttr : Kapy_EnumAttr<
  "PaddingOption",
  "padding option for out of bound memory read",
  [
    I32EnumAttrCase<"ZERO", 0, "zero">,
    I32EnumAttrCase<"QNAN", 1, "qnan">,
    I32EnumAttrCase<"PINF", 2, "pinf">,
    I32EnumAttrCase<"NINF", 3, "ninf">
  ]
>;

def Kapy_CacheModifierAttr : Kapy_EnumAttr<
  "CacheModifier",
  "cache operator on load and store instrutions",
  [
    I32EnumAttrCase<"NONE", 0, "none">,
    I32EnumAttrCase<"CA", 1, "ca">,
    I32EnumAttrCase<"CG", 2, "cg">,
    I32EnumAttrCase<"CS", 3, "cs">,
    I32EnumAttrCase<"WB", 4, "wb">,
    I32EnumAttrCase<"WT", 5, "wt">,
    I32EnumAttrCase<"LU", 6, "lu">,
    I32EnumAttrCase<"CV", 7, "cv">
  ]
>;

def Kapy_EvictPriorityAttr : Kapy_EnumAttr<
  "EvictPriority",
  "cache eviction priority hints on load and store instructions",
  [
    I32EnumAttrCase<"EVICT_NORMAL", 0, "evict_normal">,
    I32EnumAttrCase<"EVICT_FIRST", 1, "evict_first">,
    I32EnumAttrCase<"EVICT_LAST", 2, "evict_last">,
    I32EnumAttrCase<"EVICT_UNCHANGED", 3, "evict_unchanged">,
    I32EnumAttrCase<"NO_ALLOCATE", 4, "no_allocate">
  ]
>;

def Kapy_RoundingModeAttr : Kapy_EnumAttr<
  "RoundingMode",
  "rounding mode for floating-point cast instructions",
  [
    I32EnumAttrCase<"RZ", 0, "rz">,
    I32EnumAttrCase<"RN", 1, "rn">
  ]
>;

def Kapy_MatmulImplWayAttr : Kapy_EnumAttr<
  "MatmulImplWay",
  "how to implement matmul",
  [
    I32EnumAttrCase<"MMA_M16N8K8_F16", 1, "mma_m16n8k8_f16">,
    I32EnumAttrCase<"MMA_M16N8K16_F16", 2, "mma_m16n8k16_f16">,
    I32EnumAttrCase<"MMA_M16N8K8_TF32", 3, "mma_m16n8k8_tf32">,
    I32EnumAttrCase<"MMA_M16N8K16_F8", 4, "mma_m16n8k16_f8">
    // TODO: Support more.
  ]
>;

def Kapy_DefaultValuedPaddingOptionAttr
    : DefaultValuedAttr<Kapy_PaddingOptionAttr, "PaddingOption::ZERO">;

def Kapy_DefaultValuedCacheModifierAttr
    : DefaultValuedAttr<Kapy_CacheModifierAttr, "CacheModifier::NONE">;

def Kapy_DefaultValuedEvictPriorityAttr
    : DefaultValuedAttr<Kapy_EvictPriorityAttr, "EvictPriority::EVICT_NORMAL">;

#endif // KAPY_DIALECT_KAPY_IR_ENUMS
