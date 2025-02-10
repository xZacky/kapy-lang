//===- Types.td --------------------------------------------*- tablegen -*-===//
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

#ifndef KAPY_DIALECT_KGPU_IR_TYPES
#define KAPY_DIALECT_KGPU_IR_TYPES

include "mlir/IR/BuiltinTypeInterfaces.td"
include "kapy/Dialect/Kgpu/IR/Dialect.td"

class Kgpu_TypeDef<string name, string keyword, list<Trait> traits = []>
    : TypeDef<Kgpu_Dialect, name, traits> {
  let mnemonic = keyword;
}

def Kgpu_Int : AnyTypeOf<[I1, I8, I16, I32, I64], "integer">;
def Kgpu_Float : AnyTypeOf<
    [F8E4M3FNUZ, F8E5M2FNUZ, F8E5M2, F16, BF16, F32, F64], "floating-point">;
def Kgpu_Number : AnyTypeOf<[Kgpu_Int, Kgpu_Float]>;

def SharedMemRefType
    : Kgpu_TypeDef<"SharedMemRef", "shared", [ShapedTypeInterface]> {
  let summary = "shared memref (memory reference) type";

  let parameters = (ins
    ArrayRefParameter<"int64_t">:$shape,
    "Type":$elementType,
    "Attribute":$encoding
  );

  let builders = [
    TypeBuilderWithInferredContext<
      (ins
        "ArrayRef<int64_t>":$shape,
        "Type":$elementType,
        "Attribute":$encoding
      ),
      "return $_get(elementType.getContext(), shape, elementType, encoding);"
    >
  ];

  let hasCustomAssemblyFormat = 1;

  let extraClassDeclaration = [{
    SharedMemRefType
    cloneWith(std::optional<ArrayRef<int64_t>> shape, Type elementType) const {
      return get(shape.value_or(getShape()), elementType, getEncoding());
    }

    bool hasRank() const { return true; }
  }];
}

class SharedMemRefOf<list<Type> types> : DialectType<
  Kgpu_Dialect,
  And<
    [
      CPred<"isa<SharedMemRefType>($_self)">,
      Concat<"[](Type elementType) { return ",
             SubstLeaves<"$_self", "elementType", AnyTypeOf<types>.predicate>,
             "; }(cast<SharedMemRefType>($_self).getElementType())">
    ]
  >,
  "shared memref",
  "SharedMemRefType"
>;

def Kgpu_Shared : SharedMemRefOf<[Kgpu_Number]>;

#endif // KAPY_DIALECT_KGPU_IR_TYPES
