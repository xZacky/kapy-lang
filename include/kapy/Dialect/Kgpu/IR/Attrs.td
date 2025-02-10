//===- Attrs.td --------------------------------------------*- tablegen -*-===//
//
// This file defines the attributes used by the KgpuDialect.
//
//===----------------------------------------------------------------------===//

#ifndef KAPY_DIALECT_KGPU_IR_ATTRS
#define KAPY_DIALECT_KGPU_IR_ATTRS

include "kapy/Dialect/Kgpu/IR/Dialect.td"

class Kgpu_AttrDef<string name, string keyword, list<Trait> traits = []>
    : AttrDef<Kgpu_Dialect, name, traits, "Attribute"> {
  let attrName = "kgpu." # keyword;
}

def SwizzlingLayoutAttr : Kgpu_AttrDef<"SwizzlingLayout", "swizzling"> {
  let summary = "memory swizzling layout";

  let mnemonic = "swizzling";
  let parameters = (ins "int64_t":$unitSize, "int64_t":$maxPhase);

  let assemblyFormat = "`<` `(` $unitSize `,` $maxPhase `)` `>`";
}

def FragmentsLayoutAttr : Kgpu_AttrDef<"FragmentsLayout", "fragments"> {
  let summary = "fragments layout";

  let description = [{
    FragmentsLayoutAttr has 6 parameters:
    1. `$warp_array`: The number of warps on each axis.
    3. `$warp_loops`: Loop executed by each warp on each axis.
    2. `$lane_array`: The number of lanes on each axis.
    4. `$lane_loops`: Loop executed by each lane on each axis.
    5. `$minor_axis`: The axis on which thread is not contiguous.
    5. `$major_axis`: The axis on which thread is continuous.

    We will build a layout from these parameters.
    For example:
    ```
    #kgpu.blocked<[2, 2], [2, 1], [2, 4], [1, 2], (0, 1)>
    ```
    We first build the layout of threads from `$warp_array, $lane_array`, and
    `$minor_axis, $major_axis`:
    ```
    [ 0,  1,  2,  3,  8,  9, 10, 11]
    [ 4,  5,  6,  7, 12, 13, 14, 15]
    [16, 17, 18, 19, 24, 25, 26, 27]
    [20, 21, 22, 23, 28, 29, 30, 31]
    ```
    then we execute loop on each axis by `$warp_loops` and `$lane_loops`:
    ```
    [ 0,  0,  1,  1,  2,  2,  3,  3,  8,  8,  9,  9, 10, 10, 11, 11]
    [ 4,  4,  5,  5,  6,  6,  7,  7, 12, 12, 13, 13, 14, 14, 15, 15]
    [ 0,  0,  1,  1,  2,  2,  3,  3,  8,  8,  9,  9, 10, 10, 11, 11]
    [ 4,  4,  5,  5,  6,  6,  7,  7, 12, 12, 13, 13, 14, 14, 15, 15]
    [16, 16, 17, 17, 18, 18, 19, 19, 24, 24, 25, 25, 26, 26, 27, 27]
    [20, 20, 21, 21, 22, 22, 23, 23, 28, 28, 29, 29, 30, 30, 31, 31]
    [16, 16, 17, 17, 18, 18, 19, 19, 24, 24, 25, 25, 26, 26, 27, 27]
    [20, 20, 21, 21, 22, 22, 23, 23, 28, 28, 29, 29, 30, 30, 31, 31]
    ```

    Apply a layout to a tensor. Since layout and tensor may not have same shape,
    we first broadcast them into same shape, then we can get the relationship of
    elements and threads.
    For example, we have a tensor:
    ```
    [A, B, C, D, E, F, G, H]
    [I, J, K, L, M, N, O, P]
    ```
    and a layout:
    ```
    [0, 0, 1, 1]
    [2, 2, 3, 3]
    [4, 4, 5, 5]
    [6, 6, 7, 7]
    ```
    We first broadcast both of them to shape [4, 8], and then we will distribute
    elements to threads as:
    ```
    A -> (0, 4); B -> (0, 4); C -> (1, 5); D -> (1, 5);
    E -> (0, 4); F -> (0, 4); G -> (1, 5); H -> (1, 5);
    I -> (2, 6); J -> (2, 6); K -> (3, 7); L -> (3, 7);
    M -> (2, 6); N -> (2, 6); O -> (3, 7); P -> (3, 7);
    ```
  }];

  let mnemonic = "fragments";
  let parameters = (ins
    ArrayRefParameter<"int64_t">:$warpArrayRef,
    ArrayRefParameter<"int64_t">:$warpLoopsRef,
    ArrayRefParameter<"int64_t">:$laneArrayRef,
    ArrayRefParameter<"int64_t">:$laneLoopsRef,
    "unsigned":$minorAxis,
    "unsigned":$majorAxis
  );

  let builders = [
    AttrBuilder<
      (ins
        "ArrayRef<int64_t>":$warpArray,
        "ArrayRef<int64_t>":$warpLoops,
        "ArrayRef<int64_t>":$laneArray,
        "ArrayRef<int64_t>":$laneLoops,
        CArg<"bool", "true">:$rowMajor
      ),
      [{
        unsigned i = rowMajor ? 0 : 1;
        unsigned j = rowMajor ? 1 : 0;
        return get(context, warpArray, warpLoops, laneArray, laneLoops, i, j);
      }]
    >
  ];

  let assemblyFormat = [{
    `<` `[` $warpArrayRef `]` `,` `[` $warpLoopsRef `]` `,`
        `[` $laneArrayRef `]` `,` `[` $laneLoopsRef `]` `,`
        `(` $minorAxis `,` $majorAxis `)` `>`
  }];

  let extraClassDeclaration = [{
    SmallVector<int64_t, 2> getWarpArray() const {
      return llvm::to_vector<2>(getWarpArrayRef());
    }
    SmallVector<int64_t, 2> getWarpLoops() const {
      return llvm::to_vector<2>(getWarpLoopsRef());
    }
    SmallVector<int64_t, 2> getLaneArray() const {
      return llvm::to_vector<2>(getLaneArrayRef());
    }
    SmallVector<int64_t, 2> getLaneLoops() const {
      return llvm::to_vector<2>(getLaneLoopsRef());
    }

    bool isRowMajor() const { return getMajorAxis() == 1; }

    /// Get loop executed by each thread on each axis when apply to a tensor.
    SmallVector<int64_t, 2> getLoopSpace(ArrayRef<int64_t> shape) const;

    /// Get an AffineMap `(warpId, laneId, loopIv) -> (rowIdx, colIdx)`.
    /// If `unique == true`, instead of map to same position, we will map to the
    /// position in the broadcasted tensor.
    AffineMap getAffineMap(ArrayRef<int64_t> shape, bool unique = false) const;

    /// Transpose this layout to get a new layout.
    FragmentsLayoutAttr transpose() const;
  }];
}

#endif // KAPY_DIALECT_KGPU_IR_ATTRS
