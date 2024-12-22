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

def Kapy_GlobalMemLayout : Kapy_AttrDef<"GlobalMemLayout", "gmem"> {
  let description = [{
    GlobalMemLayout is an AffineMap that maps from memref indices and captured
    symbols to `offset`, where `offset` is the element offset from the base
    pointer.
  }];

  let mnemonic = "gmem";
  let parameters = (ins "AffineMap":$map);

  let assemblyFormat = "`<` struct(params) `>`";
}

#endif // KAPY_DIALECT_KAPY_IR_ATTRS
