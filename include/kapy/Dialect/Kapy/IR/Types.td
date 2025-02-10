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

#ifndef KAPY_DIALECT_KAPY_IR_TYPES
#define KAPY_DIALECT_KAPY_IR_TYPES

include "mlir/IR/BuiltinTypeInterfaces.td"
include "kapy/Dialect/Kapy/IR/Dialect.td"

class Kapy_TypeDef<string name, string keyword, list<Trait> traits = []>
    : TypeDef<Kapy_Dialect, name, traits> {
  let mnemonic = keyword;
}

def Kapy_Int : AnyTypeOf<[I1, I8, I16, I32, I64], "integer">;
def Kapy_IntLike : AnyTypeOf<[Kapy_Int, RankedTensorOf<[Kapy_Int]>]>;

def Kapy_Float : AnyTypeOf<
    [F8E4M3FNUZ, F8E5M2FNUZ, F8E5M2, F16, BF16, F32, F64], "floating-point">;
def Kapy_FloatLike : AnyTypeOf<[Kapy_Float, RankedTensorOf<[Kapy_Float]>]>;

def Kapy_Number : AnyTypeOf<[Kapy_Int, Kapy_Float]>;
def Kapy_NumberLike : AnyTypeOf<[Kapy_Number, RankedTensorOf<[Kapy_Number]>]>;

def Kapy_Tensor : RankedTensorOf<[Kapy_Number]>;

def GlobalMemRefType
    : Kapy_TypeDef<"GlobalMemRef", "global", [ShapedTypeInterface]> {
  let summary = "global memref (memory reference) type";

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
    GlobalMemRefType
    cloneWith(std::optional<ArrayRef<int64_t>> shape, Type elementType) const {
      return get(shape.value_or(getShape()), elementType, getEncoding());
    }

    bool hasRank() const { return true; }
  }];
}

class GlobalMemRefOf<list<Type> types> : DialectType<
  Kapy_Dialect,
  And<
    [
      CPred<"isa<GlobalMemRefType>($_self)">,
      Concat<"[](Type elementType) { return ",
             SubstLeaves<"$_self", "elementType", AnyTypeOf<types>.predicate>,
             "; }(cast<GlobalMemRefType>($_self).getElementType())">
    ]
  >,
  "global memref",
  "GlobalMemRefType"
>;

def Kapy_Global : GlobalMemRefOf<[Kapy_Number]>;

#endif // KAPY_DIALECT_KAPY_IR_TYPES
