//===- Attrs.td --------------------------------------------*- tablegen -*-===//
//
// This file defines the attributes used by the Kgpu dialect.
//
//===----------------------------------------------------------------------===//

#ifndef KAPY_DIALECT_KGPU_IR_ATTRS
#define KAPY_DIALECT_KGPU_IR_ATTRS

include "kapy/Dialect/Kgpu/IR/Dialect.td"

class Kgpu_AttrDef<string name, string keyword, list<Trait> traits = []>
    : AttrDef<Kgpu_Dialect, name, traits, "Attribute"> {
  let attrName = "kgpu." # keyword;
}

def Kgpu_SharedMemLayoutAttr : Kgpu_AttrDef<"SharedMemLayout", "shmem"> {
  let mnemonic = "shmem";
  let parameters = (ins
    ArrayRefParameter<"int64_t">:$strides,
    "unsigned":$bitWidth,
    "unsigned":$maxPhase
  );

  let assemblyFormat = [{
    `<` `[` $strides `]` `,` $bitWidth `,` $maxPhase `>`
  }];

  let extraClassDeclaration = [{
    /// Get an AffineMap from memref indices to the relative offset from the
    /// first element of this memref before swizzling.
    AffineMap getMemRefMap() const;
  }];
}

def Kgpu_FragmentsLayoutAttr : Kgpu_AttrDef<"FragmentsLayout", "frags"> {
  let description = [{
    FragmentsLayout has five parameters:
    1. `$shape_of_warps`: Shape of warps.
    3. `$warp_loops`: Loops executed by each warp.
    2. `$shape_of_lanes`: Shape of lanes.
    4. `$lane_loops`: Loops executed by each lane.
    5. `$order`: Order of axes from minor to major.

    We will build a layout from these parameters.
    For example:
    ```
    #frags<[2, 2], [2, 1], [2, 4], [1, 2], (0, 1)>
    ```
    We first compute the layout of threads from `$order`, `$shape_of_lanes` and
    `$shape_of_warps`:
    ```
    [ 0,  1,  2,  3,  8,  9, 10, 11]
    [ 4,  5,  6,  7, 12, 13, 14, 15]
    [16, 17, 18, 19, 24, 25, 26, 27]
    [20, 21, 22, 23, 28, 29, 30, 31]
    ```
    then we execute loop on each axis by `$lane_loops` and `$warp_loops` to get
    the final layout:
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
    A -> {0, 4}; B -> {0, 4}; C -> {1, 5}; D -> {1, 5};
    E -> {0, 4}; F -> {0, 4}; G -> {1, 5}; H -> {1, 5};
    I -> {2, 6}; J -> {2, 6}; K -> {3, 7}; L -> {3, 7};
    M -> {2, 6}; N -> {2, 6}; O -> {3, 7}; P -> {3, 7};
    ```
  }];

  let mnemonic = "frags";
  let parameters = (ins
    ArrayRefParameter<"int64_t">:$shapeOfWarpsRef,
    ArrayRefParameter<"int64_t">:$warpLoopsRef,
    ArrayRefParameter<"int64_t">:$shapeOfLanesRef,
    ArrayRefParameter<"int64_t">:$laneLoopsRef,
    ArrayRefParameter<"unsigned">:$orderRef
  );

  let builders = [
    AttrBuilder<
      (ins
        "ArrayRef<int64_t>":$shapeOfWarps,
        "ArrayRef<int64_t>":$warpLoops,
        "ArrayRef<int64_t>":$shapeOfLanes,
        "ArrayRef<int64_t>":$laneLoops
      )
    >
  ];

  let assemblyFormat = [{
    `<` `[` $shapeOfWarpsRef `]` `,` `[` $warpLoopsRef `]` `,`
        `[` $shapeOfLanesRef `]` `,` `[` $laneLoopsRef `]` `,`
        `(` $orderRef `)` `>`
  }];

  let extraClassDeclaration = [{
    unsigned getRank() const { return getOrderRef().size(); }

    SmallVector<int64_t, 4> getShapeOfWarps() const {
      return llvm::to_vector<4>(getShapeOfWarpsRef());
    }
    SmallVector<int64_t, 4> getWarpLoops() const {
      return llvm::to_vector<4>(getWarpLoopsRef());
    }
    SmallVector<int64_t, 4> getShapeOfLanes() const {
      return llvm::to_vector<4>(getShapeOfLanesRef());
    }
    SmallVector<int64_t, 4> getLaneLoops() const {
      return llvm::to_vector<4>(getLaneLoopsRef());
    }
    SmallVector<unsigned, 4> getOrder() const {
      return llvm::to_vector<4>(getOrderRef());
    }

    /// Get an AffineMap from layout indices to the thread id for this layout.
    AffineMap getLayoutMap() const;

    /// Get an AffineMap from tensor indices to the minimum thread id hold it.
    AffineMap getTensorMap(ArrayRef<int64_t> shape) const;
  }];
}

def Kgpu_AxisSliceLayoutAttr : Kgpu_AttrDef<"AxisSliceLayout", "slice"> {
  let description = [{
    AxisSliceLayout is build from slicing a layout `$parent` along `$axis`.
    For example, we have a layout:
    ```
    [ 0,  1,  2,  3]
    [ 4,  5,  6,  7]
    [ 8,  9, 10, 11]
    [12, 13, 14, 15]
    ```
    slice it along axis 0 will get four layouts:
    ```
    [ 0]  [ 1]  [ 2]  [ 3]
    [ 4]  [ 5]  [ 6]  [ 7]
    [ 8]  [ 9]  [10]  [11]
    [12], [13], [14], [15]
    ```
  }];

  let mnemonic = "slice";
  let parameters = (ins "Attribute":$parent, "unsigned":$axis);

  let assemblyFormat = "`<` $parent `,` $axis `>`";

  let extraClassDeclaration = [{
    unsigned getRank() const;

    SmallVector<int64_t, 4> unsqueeze(ArrayRef<int64_t> shape) const;
  }];
}

def Kgpu_NvidiaMmaLayoutAttr : Kgpu_AttrDef<"NvidiaMmaLayout", "nvmma"> {
  let mnemonic = "nvmma";
  let parameters = (ins
    ArrayRefParameter<"int64_t">:$shapeOfWarpsRef,
    ArrayRefParameter<"int64_t">:$warpLoopsRef
  );

  let assemblyFormat = [{
    `<` `[` $shapeOfWarpsRef `]` `,` `[` $warpLoopsRef `]` `>`
  }];

  let extraClassDeclaration = [{
    unsigned getRank() const { return getShapeOfWarpsRef().size(); }

    SmallVector<int64_t, 4> getShapeOfWarps() const {
      return llvm::to_vector<4>(getShapeOfWarpsRef());
    }
    SmallVector<int64_t, 4> getWarpLoops() const {
      return llvm::to_vector<4>(getWarpLoopsRef());
    }
    SmallVector<int64_t, 4> getShapeOfLanes() const;
    SmallVector<int64_t, 4> getLaneLoops() const;
    SmallVector<unsigned, 4> getOrder() const;

    FragmentsLayoutAttr toFragmentsLayout() const;
  }];
}

def Kgpu_MmOperandLayoutAttr : Kgpu_AttrDef<"MmOperandLayout", "mmopd"> {
  let mnemonic = "mmopd";
  let parameters = (ins
    "Attribute":$parent, "unsigned":$operandIndex, "unsigned":$bitWidth
  );

  let builders = [
    AttrBuilder<
      (ins "Attribute":$parent, "unsigned":$index, "Type":$elementType),
      "return get(context, parent, index, getIntOrFloatBitWidth(elementType));"
    >
  ];

  let assemblyFormat = "`<` $parent `,` $operandIndex `,` $bitWidth `>`";

  let extraClassDeclaration = [{
    unsigned getRank() const;
  }];
}

#endif // KAPY_DIALECT_KGPU_IR_ATTRS
