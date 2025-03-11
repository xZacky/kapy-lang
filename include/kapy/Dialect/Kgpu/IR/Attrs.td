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

def FragmentsLayoutAttr : Kgpu_AttrDef<"FragmentsLayout", "fragments"> {
  let summary = "fragments layout";

  let description = [{
    Fragments layout has 4 parameters:
    1. `$lane_array`: Number of lanes on each axis.
    2. `$lane_loops`: Loop executed by each lane on each axis.
    3. `$minor_axis`: Axis on which lane id is not contiguous.
    4. `$major_axis`: Axis on which lane id is contiguous.

    We will build a layout from these parameters.
    For example:
    ```
    #kgpu.fragments<[8, 4], [1, 2], (0, 1)>
    ```
    We first build the layout of lanes:
    ```
    [ 0,  1,  2,  3]
    [ 4,  5,  6,  7]
    [ 8,  9, 10, 11]
    [12, 13, 14, 15]
    [16, 17, 18, 19]
    [20, 21, 22, 23]
    [24, 25, 26, 27]
    [28, 29, 30, 31]
    ```
    then we execute loop for every lane:
    ```
    [ 0,  0,  1,  1,  2,  2,  3,  3]
    [ 4,  4,  5,  5,  6,  6,  7,  7]
    [ 8,  8,  9,  9, 10, 10, 11, 11]
    [12, 12, 13, 13, 14, 14, 15, 15]
    [16, 16, 17, 17, 18, 18, 19, 19]
    [20, 20, 21, 21, 22, 22, 23, 23]
    [24, 24, 25, 25, 26, 26, 27, 27]
    [28, 28, 29, 29, 30, 30, 31, 31]
    ```

    Apply a layout to a tensor. Since layout and tensor may not have same shape,
    we first broadcast them into same shape, then we can get the relationship of
    elements and lanes.
    
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
    elements to lanes as:
    ```
    A -> (0, 4); B -> (0, 4); C -> (1, 5); D -> (1, 5);
    E -> (0, 4); F -> (0, 4); G -> (1, 5); H -> (1, 5);
    I -> (2, 6); J -> (2, 6); K -> (3, 7); L -> (3, 7);
    M -> (2, 6); N -> (2, 6); O -> (3, 7); P -> (3, 7);
    ```
  }];

  let mnemonic = "fragments";
  let parameters = (ins
    ArrayRefParameter<"int64_t">:$laneArrayRef,
    ArrayRefParameter<"int64_t">:$laneLoopsRef,
    "unsigned":$minorAxis,
    "unsigned":$majorAxis
  );

  let builders = [
    AttrBuilder<
      (ins
        "ArrayRef<int64_t>":$laneArray,
        "ArrayRef<int64_t>":$laneLoops,
        CArg<"bool", "true">:$rowMajor
      ),
      [{
        unsigned i = rowMajor ? 0 : 1;
        unsigned j = rowMajor ? 1 : 0;
        return get(context, laneArray, laneLoops, i, j);
      }]
    >
  ];

  let assemblyFormat = [{
    `<` `[` $laneArrayRef `]` `,` `[` $laneLoopsRef `]` `,`
        `(` $minorAxis `,` $majorAxis `)` `>`
  }];

  let extraClassDeclaration = [{
    inline SmallVector<int64_t, 2> getLaneArray() const {
      return llvm::to_vector<2>(getLaneArrayRef());
    }
    inline SmallVector<int64_t, 2> getLaneLoops() const {
      return llvm::to_vector<2>(getLaneLoopsRef());
    }

    inline bool isRowMajor() const { return getMajorAxis() == 1; }
    inline bool isColMajor() const { return getMajorAxis() == 0; }

    SmallVector<int64_t, 2> getWarpLoops(ArrayRef<int64_t> shape) const;

    SmallVector<int64_t, 2> getLoopSpace(ArrayRef<int64_t> shape) const;

    /// Options:
    /// 1. From lane id and loop iv to tensor indices.
    /// 2. From lane id and loop iv to values indices.
    /// 3. From values indices to lane id and loop iv.
    AffineMap getAffineMap(ArrayRef<int64_t> shape, unsigned option) const;

    /// Transpose this layout to get a new layout.
    FragmentsLayoutAttr transpose() const;

    /// Exchange minor axis and major axis to get a new layout.
    FragmentsLayoutAttr exchangeAxes() const;
  }];
}

#endif // KAPY_DIALECT_KGPU_IR_ATTRS
