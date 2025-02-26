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

def Strided2dLayoutAttr : Kapy_AttrDef<"Strided2dLayout", "strided2d"> {
  let summary = "strided 2d memory layout";

  let mnemonic = "strided2d";
  let parameters = (ins "int64_t":$strideX, "int64_t":$strideY);

  let hasCustomAssemblyFormat = 1;

  let extraClassDeclaration = [{
    inline bool isRowMajor() const { return getStrideY() == 1; }
    inline bool isColMajor() const { return getStrideX() == 1; }

    AffineMap getAffineMap() const;
  }];
}

#endif // KAPY_DIALECT_KAPY_IR_ATTRS
