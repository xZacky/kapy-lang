//===- Attrs.td --------------------------------------------*- tablegen -*-===//
//
// This file defines the attributes used by the kgpu dialect.
//
//===----------------------------------------------------------------------===//

#ifndef KAPY_DIALECT_KGPU_IR_ATTRS
#define KAPY_DIALECT_KGPU_IR_ATTRS

include "kapy/Dialect/Kgpu/IR/Dialect.td"

class Kgpu_AttrDef<string name, string keyword, list<Trait> traits = []>
    : AttrDef<Kgpu_Dialect, name, traits, "Attribute"> {
  let attrName = "kgpu." # keyword;
}

def Kgpu_RegistersLayout : Kgpu_AttrDef<"RegistersLayout", "regs"> {
  let description = [{
    RegistersLayout is an AffineMap that maps from a tuple of integers
    `(elemId, laneId, warpId)` to tensor indices, where `elemId` is the id of
    element within each thread.
  }];

  let mnemonic = "regs";
  let parameters = (ins "AffineMap":$map);

  let builders = [
    AttrBuilder<
      (ins
        "ArrayRef<int>":$tilePerLane,
        "ArrayRef<int>":$lanePerWarp,
        "ArrayRef<int>":$warpPerCTA,
        "ArrayRef<int>":$replication
      )
    >,
    AttrBuilder<
      (ins
        "ArrayRef<int64_t>":$shape,
        "ArrayRef<int>":$tilePerLane,
        "ArrayRef<int>":$lanePerWarp,
        "ArrayRef<int>":$warpPerCTA
      )
    >,
    AttrBuilder<
      (ins
        "ArrayRef<int64_t>":$shape,
        "ArrayRef<int>":$tilePerLane,
        "int":$numWarps
      )
    >
  ];

  let assemblyFormat = "`<` struct(params) `>`";

  let extraClassDeclaration = [{
    // Get the maximum element id `maxElemId`, range of element id used by this
    // layout is `[0, maxElemId)`.
    int getMaxElementId(ArrayRef<int64_t> shape, int numWarps) const;

    // Get the maximum lane id `maxLaneId`, range of lane id used by this layout
    // is `[0, maxLaneId)`.
    int getMaxLaneId(ArrayRef<int64_t> shape, int numWarps) const;

    // Get the maximum warp id `maxWarpId`, range of warp id used by this layout
    // is `[0, maxWarpId)`.
    int getMaxWarpId(ArrayRef<int64_t> shape, int numWarps) const;
  }];
}

def Kgpu_SharedMemLayout : Kgpu_AttrDef<"SharedMemLayout", "smem"> {
  let description = [{
    There are two parameters of SharedMemLayout:
    1. An AffineMap that maps from memref indices to `offset`, where `offset`
       is the element offset from the base pointer before swizzling.
    2. An integer `numModes` represents the number of modes we do swizzling,
       default `numModes` is 1, which means no swizzling.

    The complete logic to get the offset after swizzling contains three steps:
    1) Input memref indices to the map and get the `offset` before swizzling.
    2) Compute `modeId = (offset * bitWidth) / (32 * 32)`, where `bitWidth` is
       the element bit-width of this memref. Here the first 32 is the bit-width
       of a bank word (4 bytes), the second 32 is the number of banks.
    3) Compute `offset = offset ^ modeId` to get the offset after swizzling.

    Since swizzling won't affect the way we do vectorization, so we can only do
    the step 1) when we analyze the contiguity for vectorization.
  }];

  let mnemonic = "smem";
  let parameters = (ins
    "AffineMap":$map, DefaultValuedParameter<"int", "1">:$num_modes
  );

  let assemblyFormat = "`<` struct(params) `>`";
}

def Kgpu_NvidiaMmaLayout : Kgpu_AttrDef<"NvidiaMmaLayout", "nvmma"> {
  let mnemonic = "nvmma";
  let parameters = (ins ArrayRefParameter<"int">:$warpPerCTARef);

  let assemblyFormat = "`<` `warp_per_cta` `=` `[` $warpPerCTARef `]` `>`";

  let extraClassDeclaration = [{
    int getRank() const { return getWarpPerCTARef().size(); }
    RegistersLayoutAttr toRegistersLayout(ArrayRef<int64_t> shape) const;

    SmallVector<int, 4> getTilePerLane() const;
    SmallVector<int, 4> getLanePerWarp() const;
    SmallVector<int, 4> getWarpPerCTA() const;
    SmallVector<int, 4> getReplication(ArrayRef<int64_t> shape) const;

    SmallVector<int, 4> getTilePerLaneForChild(int bitWidth) const;
    SmallVector<int, 4> getLanePerWarpForChild() const;
    SmallVector<int, 4> getReplicationForChild(
        int index, int bitWidth, ArrayRef<int64_t> shape) const;
  }];
}

def Kgpu_DotOpLoadLayout : Kgpu_AttrDef<"DotOpLoadLayout", "dotld"> {
  let mnemonic = "dotld";
  let parameters = (ins
    "Attribute":$parent, "int":$operand_index, "int":$bit_width
  );

  let builders = [
    AttrBuilder<
      (ins "Attribute":$parent, "int":$index, "Type":$elementType),
      "return get(context, parent, index, getIntOrFloatBitWidth(elementType));"
    >
  ];

  let assemblyFormat = "`<` struct(params) `>`";

  let extraClassDeclaration = [{
    int getRank() const;
    RegistersLayoutAttr toRegistersLayout(ArrayRef<int64_t> shape) const;
  }];
}

#endif // KAPY_DIALECT_KGPU_IR_ATTRS
