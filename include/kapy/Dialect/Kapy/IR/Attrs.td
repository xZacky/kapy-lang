//===- Attrs.td --------------------------------------------*- tablegen -*-===//
//
// This file defines the attributes used by the KapyDialect.
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
  let parameters = (ins "int64_t":$stride0, "int64_t":$stride1);

  let hasCustomAssemblyFormat = 1;

  let extraClassDeclaration = [{
    SmallVector<int64_t, 2> getStrides() const {
      return {getStride0(), getStride1()};
    }

    AffineMap getAffineMap() const;
  }];
}

#endif // KAPY_DIALECT_KAPY_IR_ATTRS
