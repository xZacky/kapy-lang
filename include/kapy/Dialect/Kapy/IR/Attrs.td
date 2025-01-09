//===- Attrs.td --------------------------------------------*- tablegen -*-===//
//
// This file defines the attributes used by the Kapy dialect.
//
//===----------------------------------------------------------------------===//

#ifndef KAPY_DIALECT_KAPY_IR_ATTRS
#define KAPY_DIALECT_KAPY_IR_ATTRS

include "kapy/Dialect/Kapy/IR/Dialect.td"

class Kapy_AttrDef<string name, string keyword, list<Trait> traits = []>
    : AttrDef<Kapy_Dialect, name, traits, "Attribute"> {
  let attrName = "kapy." # keyword;
}

def Kapy_GlobalMemLayoutAttr : Kapy_AttrDef<"GlobalMemLayout", "glmem"> {
  let mnemonic = "glmem";
  let parameters = (ins ArrayRefParameter<"int64_t">:$strides);

  let hasCustomAssemblyFormat = 1;

  let extraClassDeclaration = [{
    /// Get an AffineMap from memref indices to the relative offset from the
    /// first element in memory. Use symbols for dynamic strides.
    AffineMap getMemRefMap() const;
  }];
}

#endif // KAPY_DIALECT_KAPY_IR_ATTRS
