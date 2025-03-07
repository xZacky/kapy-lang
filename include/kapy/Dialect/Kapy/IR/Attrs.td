//===- Attrs.td --------------------------------------------*- tablegen -*-===//
//
// This file defines the attributes used by the kapy dialect.
//
//===----------------------------------------------------------------------===//

#ifndef KAPY_DIALECT_KAPY_IR_ATTRS
#define KAPY_DIALECT_KAPY_IR_ATTRS

include "kapy/Dialect/Kapy/IR/Dialect.td"

class Kapy_AttrDef<string name, string keyword, list<Trait> traits = []>
    : AttrDef<Kapy_Dialect, name, traits, "Attribute"> {
  let attrName = "kapy." # keyword;
}

def EncodingAttr : Kapy_AttrDef<"Encoding", "encoding"> {
  let summary = "tensor encoding";

  let mnemonic = "encoding";
  let parameters = (ins
    "MemorySpace":$memory, OptionalParameter<"Attribute">:$layout
  );

  let assemblyFormat = "`<` $memory (`,` $layout^)? `>`";
}

def Strided2dLayoutAttr : Kapy_AttrDef<"Strided2dLayout", "strided2d"> {
  let summary = "strided 2d layout";

  let mnemonic = "strided2d";
  let parameters = (ins "int64_t":$strideX, "int64_t":$strideY);

  let hasCustomAssemblyFormat = 1;

  let extraClassDeclaration = [{
    inline bool isRowMajor() const { return getStrideY() == 1; }
    inline bool isColMajor() const { return getStrideX() == 1; }
  }];
}

def SwizzlingLayoutAttr : Kapy_AttrDef<"SwizzlingLayout", "swizzling"> {
  let summary = "swizzling layout";

  let mnemonic = "swizzling";
  let parameters = (ins
    "int64_t":$strideX,
    "int64_t":$strideY,
    "int64_t":$bankParam,
    "int64_t":$lineParam
  );

  let builders = [
    AttrBuilder<
      (ins "int64_t":$strideX, "int64_t":$strideY),
      [{
        auto dynamic = ShapedType::kDynamic;
        return get(context, strideX, strideY, dynamic, dynamic);
      }]
    >
  ];

  let hasCustomAssemblyFormat = 1;

  let extraClassDeclaration = [{
    inline bool isRowMajor() const { return getStrideY() == 1; }
    inline bool isColMajor() const { return getStrideX() == 1; }

    bool isDynamicParams() const;

    SwizzlingLayoutAttr setParams(int64_t bankParam, int64_t lineParam) const;
  }];
}

#endif // KAPY_DIALECT_KAPY_IR_ATTRS
