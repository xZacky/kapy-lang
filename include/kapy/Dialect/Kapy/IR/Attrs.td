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

#endif // KAPY_DIALECT_KAPY_IR_ATTRS
