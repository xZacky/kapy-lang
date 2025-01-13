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
    : TypeDef<Kapy_Dialect, name, traits> { let mnemonic = keyword; }

def Kapy_Int : AnyTypeOf<[I1, I8, I16, I32, I64], "integer">;
def Kapy_IntTensor : RankedTensorOf<[Kapy_Int]>;
def Kapy_IntLike : AnyTypeOf<[Kapy_Int, Kapy_IntTensor]>;

def Kapy_I1Tensor : RankedTensorOf<[I1]>;
def Kapy_I1Like : AnyTypeOf<[I1, Kapy_I1Tensor]>;

def Kapy_I32Tensor : RankedTensorOf<[I32]>;
def Kapy_I32Like : AnyTypeOf<[I32, Kapy_I32Tensor]>;

def Kapy_Fp : AnyTypeOf<
  [F8E4M3FNUZ, F8E5M2, F8E5M2FNUZ, F16, BF16, F32, F64], "floating-point"
>;
def Kapy_FpTensor : RankedTensorOf<[Kapy_Fp]>;
def Kapy_FpLike : AnyTypeOf<[Kapy_Fp, Kapy_FpTensor]>;

def Kapy_IntFp : AnyTypeOf<[Kapy_Int, Kapy_Fp]>;
def Kapy_IntFpTensor : RankedTensorOf<[Kapy_IntFp]>;
def Kapy_IntFpLike : AnyTypeOf<[Kapy_IntFp, Kapy_IntFpTensor]>;

def Kapy_PointerType : Kapy_TypeDef<"KapyPointer", "ptr"> {
  let parameters = (ins "unsigned":$addressSpace);

  let skipDefaultBuilders = 1;
  let builders = [
    TypeBuilder<(ins "unsigned":$space), "return $_get($_ctxt, space);">
  ];

  let assemblyFormat = "`<` $addressSpace `>`";
}

def Kapy_Ptr : DialectType<
  Kapy_Dialect,
  CPred<"isa<KapyPointerType>($_self)">,
  "pointer",
  "KapyPointerType"
>;

def Kapy_MemRefType
    : Kapy_TypeDef<"KapyMemRef", "memref", [ShapedTypeInterface]> {
  let parameters = (ins
    ArrayRefParameter<"int64_t">:$shape,
    "Type":$element_type,
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
    KapyMemRefType
    cloneWith(std::optional<ArrayRef<int64_t>> shape, Type elementType) const {
      return KapyMemRefType::get(
          shape.value_or(getShape()), elementType, getEncoding());
    }

    bool hasRank() const { return true; }
  }];
}

class Kapy_MemRefOf<list<Type> types> : DialectType<
  Kapy_Dialect,
  And<
    [
      CPred<"isa<KapyMemRefType>($_self)">,
      Concat<"[](Type elementType) { return ",
             SubstLeaves<"$_self", "elementType", AnyTypeOf<types>.predicate>,
             "; }(cast<KapyMemRefType>($_self).getElementType())">
    ]
  >,
  "memref",
  "KapyMemRefType"
>;

def Kapy_IntFpMemRef : Kapy_MemRefOf<[Kapy_IntFp]>;

#endif // KAPY_DIALECT_KAPY_IR_TYPES
