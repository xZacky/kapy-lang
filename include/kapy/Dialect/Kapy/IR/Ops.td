//===- Ops.td ----------------------------------------------*- tablegen -*-===//
//
// Copyright 2018-2020 Philippe Tillet
// Copyright 2020-2022 OpenAI
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.
//
//===----------------------------------------------------------------------===//
// 
// This file is modified from the triton project.
// https://github.com/triton-lang/triton
//
//===----------------------------------------------------------------------===//

#ifndef KAPY_DIALECT_KAPY_IR_OPS
#define KAPY_DIALECT_KAPY_IR_OPS

include "mlir/IR/OpAsmInterface.td"
include "mlir/IR/SymbolInterfaces.td"
include "mlir/Interfaces/CallInterfaces.td"
include "mlir/Interfaces/ControlFlowInterfaces.td"
include "mlir/Interfaces/FunctionInterfaces.td"
include "mlir/Interfaces/InferIntRangeInterface.td"
include "mlir/Interfaces/InferTypeOpInterface.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "kapy/Dialect/Kapy/IR/Dialect.td"
include "kapy/Dialect/Kapy/IR/Enums.td"
include "kapy/Dialect/Kapy/IR/Types.td"
include "kapy/Dialect/Kapy/IR/OpTraits.td"

class Kapy_Op<string keyword, list<Trait> traits = []>
    : Op<Kapy_Dialect, keyword, traits>;

def Kapy_FPToFPOp : Kapy_Op<
  "fptofp",
  [Elementwise, Pure, SameOperandsAndResultShape, SameOperandsAndResultLayout]
> {
  let description = [{
    Cast between floating-point types, attribute `$rounding_mode` must be set
    for down cast.
  }];

  let arguments = (ins
    Kapy_FloatLike:$source, OptionalAttr<Kapy_RoundingModeAttr>:$rounding_mode
  );
  let results = (outs Kapy_FloatLike:$result);

  let assemblyFormat = "$source attr-dict `:` type($source) `to` type($result)";

  let hasVerifier = 1;

  let extraClassDeclaration = [{
    bool isUpCast();
    bool isDownCast();
  }];
}

def Kapy_ClampFOp : Kapy_Op<
  "clampf", [Elementwise, Pure, SameOperandsAndResultType]
> {
  let description = [{
    Clamps `$source` within the range `[$low, $high]`.
    If `$propagate_nan` is set to `all`, then all the NaN in `$source` will
    propagate to `$result`.
    If either `$low` or `$high` is NaN, the `$result` is undefined.
  }];

  let arguments = (ins
    Kapy_FloatLike:$source,
    Kapy_FloatLike:$low,
    Kapy_FloatLike:$high,
    Kapy_PropagateNanAttr:$propagate_nan
  );
  let results = (outs Kapy_FloatLike:$result);

  let assemblyFormat = "$source `,` $low `,` $high attr-dict `:` type($source)";
}

def Kapy_MkGlobalOp : Kapy_Op<"mk_global", [Pure, ResultInGlobalMemory]> {
  let arguments = (ins
    I64:$address, I32:$size_x, I32:$size_y, I32:$stride_x, I32:$stride_y
  );
  let results = (outs Kapy_Tensor:$result);

  let assemblyFormat = [{
    $address `,` `[` $size_x `,` $size_y `]` `,` `[` $stride_x `,` $stride_y `]`
    attr-dict `:` type($result)
  }];
}

def Kapy_SvGlobalOp : Kapy_Op<
  "sv_global", [Pure, SourceInGlobalMemory, ResultInGlobalMemory]
> {
  let description = [{
    Takes a subview of tensor in global memory.
    Result will be padded to power of 2 size, padded element is nullptr.
  }];

  let arguments = (ins
    Kapy_Tensor:$source, I32:$start_x, I32:$end_x, I32:$start_y, I32:$end_y
  );
  let results = (outs Kapy_Tensor:$result);

  let assemblyFormat = [{
    $source `[` $start_x `:` $end_x `,` $start_y `:` $end_y `]` attr-dict `:`
    type($source) `->` type($result)
  }];

  let hasCanonicalizeMethod = 1;
  let hasVerifier = 1;
}

def Kapy_LdGlobalOp : Kapy_Op<
  "ld_global", 
  [
    SameOperandsAndResultShape,
    SameOperandsAndResultElementType,
    SourceInGlobalMemory,
    DeclareOpInterfaceMethods<MemoryEffectsOpInterface>,
    TypesMatchWith<"from filler type infer result type",
                   "filler", "result", "$_self">
  ]
> {
  let description = [{
    For each element, loads `$result` from `$source` if it is not nullptr,
    otherwise, pads `$result` with `$filler`.
  }];

  let arguments = (ins
    Kapy_Tensor:$source,
    Kapy_Tensor:$filler,
    Kapy_DefaultValuedCacheModifierAttr:$cache_modifier,
    Kapy_DefaultValuedEvictPriorityAttr:$evict_priority,
    DefaultValuedAttr<BoolAttr, "false">:$if_volatile
  );
  let results = (outs Kapy_Tensor:$result);

  let assemblyFormat = [{
    $source `,` $filler attr-dict `:` type($source) `->` type($filler)
  }];

  let extraClassDeclaration = [{
    inline bool isVolatile() { return getIfVolatile(); }
  }];
}

def Kapy_StGlobalOp : Kapy_Op<
  "st_global",
  [
    SameOperandsShape,
    SameOperandsElementType,
    TargetInGlobalMemory,
    DeclareOpInterfaceMethods<MemoryEffectsOpInterface>
  ]
> {
  let description = [{
    For each element, stores `$source` to `$target` if it is not nullptr.
  }];

  let arguments = (ins
    Kapy_Tensor:$source,
    Kapy_Tensor:$target,
    Kapy_DefaultValuedCacheModifierAttr:$cache_modifier,
    Kapy_DefaultValuedEvictPriorityAttr:$evict_priority
  );

  let assemblyFormat = [{
    $source `,` $target attr-dict `:` type($source) `,` type($target)
  }];
}

def Kapy_AtomicRMWOp : Kapy_Op<
  "atomic_rmw",
  [
    SameOperandsAndResultShape,
    SameOperandsAndResultElementType,
    SourceInGlobalMemory,
    DeclareOpInterfaceMethods<MemoryEffectsOpInterface>,
    TypesMatchWith<"from value type infer filler type",
                   "value", "filler", "$_self">,
    TypesMatchWith<"from value type infer result type",
                   "value", "result", "$_self">
  ]
> {
  let description = [{
    For each element, loads `$result` from `$source`, computes with `$value` and
    stores the computation result back if it is not nullptr, otherwise, pads
    `$result` with `$filler`.
  }];

  let arguments = (ins
    Kapy_AtomicRMWKindAttr:$kind,
    Kapy_Tensor:$source,
    Kapy_Tensor:$value,
    Kapy_Tensor:$filler,
    Kapy_MemSemanticAttr:$mem_semantic
  );
  let results = (outs Kapy_Tensor:$result);

  let assemblyFormat = [{
    $kind $source `,` $value `,` $filler attr-dict `:` type($source) `->`
    type($value)
  }];

  let hasVerifier = 1;
}

def Kapy_MkSharedOp : Kapy_Op<
  "mk_shared",
  [ResultInSharedMemory, DeclareOpInterfaceMethods<MemoryEffectsOpInterface>]
> {
  let results = (outs Kapy_Tensor:$result);

  let assemblyFormat = "attr-dict `:` type($result)";
}

def Kapy_SvSharedOp : Kapy_Op<
  "sv_shared", [Pure, SourceInSharedMemory, ResultInSharedMemory]
> {
  let description = [{
    Takes a subview of tensor in shared memory.
    Result will be padded to power of 2 size, padded element is nullptr.
  }];

  let arguments = (ins
    Kapy_Tensor:$source, I32:$start_x, I32:$end_x, I32:$start_y, I32:$end_y
  );
  let results = (outs Kapy_Tensor:$result);

  let assemblyFormat = [{
    $source `[` $start_x `:` $end_x `,` $start_y `:` $end_y `]` attr-dict `:`
    type($source) `->` type($result)
  }];

  let hasCanonicalizeMethod = 1;
  let hasVerifier = 1;
}

def Kapy_LdSharedOp : Kapy_Op<
  "ld_shared",
  [
    SameOperandsAndResultShape,
    SameOperandsAndResultElementType,
    SourceInSharedMemory,
    DeclareOpInterfaceMethods<MemoryEffectsOpInterface>
  ]
> {
  let description = [{
    For each element, loads `$result` from `$source` if it is not nullptr,
    otherwise, pads `$result` with zero.
  }];

  let arguments = (ins Kapy_Tensor:$source);
  let results = (outs Kapy_Tensor:$result);

  let assemblyFormat = "$source attr-dict `:` type($source) `->` type($result)";
}

def Kapy_StSharedOp : Kapy_Op<
  "st_shared",
  [
    SameOperandsAndResultShape,
    SameOperandsAndResultElementType,
    TargetInSharedMemory,
    DeclareOpInterfaceMethods<MemoryEffectsOpInterface>
  ]
> {
  let description = [{
    For each element, stores `$source` to `$target` if it is not nullptr.
  }];

  let arguments = (ins Kapy_Tensor:$source, Kapy_Tensor:$target);
  
  let assemblyFormat = [{
    $source `,` $target attr-dict `:` type($source) `,` type($target)
  }];
}

def Kapy_LdMatrixOp : Kapy_Op<
  "ld_matrix",
  [
    SameOperandsAndResultShape,
    SameOperandsAndResultElementType,
    SourceInSharedMemory,
    DeclareOpInterfaceMethods<MemoryEffectsOpInterface>
  ]
> {
  let description = [{
    For each element, loads `$result` from `$source` if it is not nullptr,
    otherwise, pads `$result` with `$zero`.
    The `$loader` is a virtual operand, it is used to indicate the layout of
    pointers when lowering this operation.
  }];

  let arguments = (ins Kapy_Tensor:$source, Kapy_Tensor:$loader);
  let results = (outs Kapy_Tensor:$result);

  let assemblyFormat = [{
    $source `,` $loader attr-dict `:` type($source) `,` type($loader) `->`
    type($result)
  }];

  let hasVerifier = 1;
}

def Kapy_CpAsyncGlobalToSharedOp : Kapy_Op<
  "cp_async_global_to_shared",
  [
    SameOperandsShape,
    SameOperandsElementType,
    SourceInGlobalMemory,
    TargetInSharedMemory,
    DeclareOpInterfaceMethods<MemoryEffectsOpInterface>
  ]
> {
  let description = [{
    For each elements, copies from `$source` to `$result` if it is not nullptr,
    otherwise, pads `$result` with zero.
    The `$loader` is a virtual operand, it is used to indicate the layout of
    pointers when lowering this operation.
  }];

  let arguments = (ins
    Kapy_Tensor:$source,
    Kapy_Tensor:$target,
    Kapy_Tensor:$loader,
    Kapy_DefaultValuedCacheModifierAttr:$cache_modifier,
    Kapy_DefaultValuedEvictPriorityAttr:$evict_priority
  );

  let assemblyFormat = [{
    $source `,` $target `,` $loader attr-dict `:` type($source) `,`
    type($target) `,` type($loader)
  }];
}

def Kapy_CpAsyncCommitGroupOp : Kapy_Op<"cp_async_commit_group"> {
  let assemblyFormat = "attr-dict";
}

def Kapy_CpAsyncWaitGroupOp : Kapy_Op<"cp_async_wait_group"> {
  let arguments = (ins I32Attr:$num_pending);

  let assemblyFormat = "attr-dict";
}

def Kapy_SplatOp : Kapy_Op<"splat", [Pure, SameOperandsAndResultElementType]> {
  let arguments = (ins Kapy_Number:$source);
  let results = (outs Kapy_Tensor:$result);

  let assemblyFormat = [{
    $source attr-dict `:` type($source) `->` type($result)
  }];

  let hasFolder = 1;
}

def Kapy_BroadcastOp : Kapy_Op<
  "broadcast",
  [Pure, SameOperandsAndResultElementType, SameOperandsAndResultLayout]
> {
  let arguments = (ins Kapy_Tensor:$source);
  let results = (outs Kapy_Tensor:$result);

  let assemblyFormat = [{
    $source attr-dict `:` type($source) `->` type($result)
  }];

  let hasCanonicalizeMethod = 1;
  let hasFolder = 1;
  let hasVerifier = 1;
}

def Kapy_TransposeOp : Kapy_Op<
  "transpose",
  [
    Pure,
    SameOperandsAndResultElementType,
    DeclareOpInterfaceMethods<InferTypeOpInterface>
  ]
> {
  let arguments = (ins Kapy_Tensor:$source);
  let results = (outs Kapy_Tensor:$result);

  let assemblyFormat = [{
    $source attr-dict `:` type($source) `->` type($result)
  }];

  let hasCanonicalizeMethod = 1;
  let hasFolder = 1;
}

def Kapy_ProgramIdOp : Kapy_Op<"program_id", [Pure]> {
  let arguments = (ins I32Attr:$axis);
  let results = (outs I32:$result);

  let assemblyFormat = "attr-dict `:` type($result)";
}

def Kapy_WarpIdOp : Kapy_Op<
  "warp_id",
  [
    Pure,
    DeclareOpInterfaceMethods<InferIntRangeInterface, ["inferResultRanges"]>
  ]
> {
  let results = (outs I32:$result);

  let assemblyFormat = "attr-dict `:` type($result)";
}

def Kapy_ArangeOp : Kapy_Op<"arange", [Pure]> {
  let arguments = (ins I32Attr:$axis, I32Attr:$start, I32Attr:$end);
  let results = (outs RankedTensorOf<[I32]>:$result);

  let assemblyFormat = "attr-dict `:` type($result)";

  let hasFolder = 1;
  let hasVerifier = 1;
}

def Kapy_MatmulOp : Kapy_Op<
  "matmul",
  [
    Pure,
    TypesMatchWith<"from acc type infer result type", "acc", "result", "$_self">
  ]
> {
  let description = "Computes `$result = matmul($lhs, $rhs) + $acc`.";

  let arguments = (ins
    Kapy_Tensor:$lhs,
    Kapy_Tensor:$rhs,
    Kapy_Tensor:$acc,
    Kapy_MatmulImplWayAttr:$matmul_impl_way
  );
  let results = (outs Kapy_Tensor:$result);

  let assemblyFormat = [{
    $matmul_impl_way $lhs `,` $rhs `,` $acc attr-dict `:` type($lhs) `,`
    type($rhs) `->` type($acc)
  }];

  let hasVerifier = 1;
}

def Kapy_ReduceOp : Kapy_Op<
  "reduce",
  [
    Pure,
    SameOperandsAndResultElementType,
    SameOperandsAndResultLayout,
    SingleBlock,
    DeclareOpInterfaceMethods<InferTypeOpInterface>
  ]
> {
  let arguments = (ins Kapy_Tensor:$source, I32Attr:$axis);
  let regions = (region SizedRegion<1>:$lambda);
  let results = (outs Kapy_Tensor:$result);

  let hasCustomAssemblyFormat = 1;
  let hasRegionVerifier = 1;
}

def Kapy_ElementwiseExternLibOp : Kapy_Op<
  "elementwise_extern_lib",
  [
    Elementwise,
    SameOperandsAndResultShape,
    SameOperandsAndResultLayout,
    DeclareOpInterfaceMethods<MemoryEffectsOpInterface>
  ]
> {
  let arguments = (ins
    Variadic<Kapy_NumberLike>:$operands,
    StrAttr:$lib_name,
    StrAttr:$lib_path,
    StrAttr:$sym_name,
    BoolAttr:$pure
  );
  let results = (outs Kapy_NumberLike:$result);

  let assemblyFormat = [{
    $operands attr-dict `:` functional-type($operands, $result)
  }];

  let extraClassDeclaration = [{
    inline bool isPure() { return getPure(); }
  }];
}

def Kapy_ElementwiseInlineAsmOp : Kapy_Op<
  "elementwise_inline_asm",
  [
    Elementwise,
    SameOperandsAndResultShape,
    SameOperandsAndResultLayout,
    DeclareOpInterfaceMethods<MemoryEffectsOpInterface>
  ]
> {
  let arguments = (ins
    Variadic<Kapy_NumberLike>:$operands,
    StrAttr:$asm_string,
    StrAttr:$constraints,
    BoolAttr:$pure
  );
  let results = (outs Kapy_NumberLike:$result);

  let assemblyFormat = [{
    $operands attr-dict `:` functional-type($operands, $result)
  }];

  let extraClassDeclaration = [{
    inline bool isPure() { return getPure(); }
  }];
}

def Kapy_CallOp : Kapy_Op<
  "call",
  [
    CallOpInterface,
    DeclareOpInterfaceMethods<SymbolUserOpInterface>
  ]
> {
  let arguments = (ins FlatSymbolRefAttr:$callee, Variadic<AnyType>:$operands);
  let results = (outs Variadic<AnyType>);

  let builders = [
    OpBuilder<
      (ins "FuncOp":$callee, CArg<"ValueRange", "{}">:$operands),
      [{
        $_state.addOperands(operands);
        $_state.addAttribute("callee", SymbolRefAttr::get(callee));
        $_state.addTypes(callee.getFunctionType().getResults());
      }]
    >,
    OpBuilder<
      (ins
        "SymbolRefAttr":$callee,
        "TypeRange":$types,
        CArg<"ValueRange", "{}">:$operands
      ),
      [{
        $_state.addOperands(operands);
        $_state.addAttribute("callee", callee);
        $_state.addTypes(types);
      }]
    >,
    OpBuilder<
      (ins
        "StringAttr":$callee,
        "TypeRange":$types,
        CArg<"ValueRange", "{}">:$operands
      ),
      "build($_builder, $_state, SymbolRefAttr::get(callee), types, operands);"
    >,
    OpBuilder<
      (ins
        "StringRef":$callee,
        "TypeRange":$types,
        CArg<"ValueRange", "{}">:$operands
      ),
      [{
        build($_builder, $_state,
              StringAttr::get($_builder.getContext(), callee), types, operands);
      }]
    >
  ];

  let assemblyFormat = [{
    $callee `(` $operands `)` attr-dict `:` functional-type($operands, results)
  }];

  let extraClassDeclaration = [{
    FunctionType getCalleeType() {
      return FunctionType::get(getContext(), getOperandTypes(),
                               getResultTypes());
    }

    operand_range getArgOperands() {
      return {operand_begin(), operand_end()};
    }

    CallInterfaceCallable getCallableForCallee() {
      return (*this)->getAttrOfType<SymbolRefAttr>("callee");
    }

    void setCalleeFromCallable(CallInterfaceCallable callee) {
      (*this)->setAttr("callee", callee.get<SymbolRefAttr>());
    }

    /// Required by CallOpInterface.
    MutableOperandRange getArgOperandsMutable() {
      return getOperandsMutable();
    }
  }];
}

def Kapy_FuncOp : Kapy_Op<
  "func",
  [
    AffineScope,
    AutomaticAllocationScope,
    CallableOpInterface,
    FunctionOpInterface,
    IsolatedFromAbove,
    OpAsmOpInterface,
    ValidTensorShape,
    ValidMemorySpace
  ]
> {
  let arguments = (ins
    SymbolNameAttr:$sym_name,
    TypeAttrOf<FunctionType>:$function_type,
    OptionalAttr<StrAttr>:$sym_visibility,
    OptionalAttr<DictArrayAttr>:$arg_attrs,
    OptionalAttr<DictArrayAttr>:$res_attrs
  );
  let regions = (region AnyRegion:$body);

  let builders = [
    OpBuilder<
      (ins
        "StringRef":$name,
        "FunctionType":$funcType,
        CArg<"bool", "true">:$inlineable,
        CArg<"ArrayRef<DictionaryAttr>", "{}">:$argAttrs
      )
    >
  ];

  let hasCustomAssemblyFormat = 1;

  let extraClassDeclaration = [{
    Region *getCallableRegion() { return isExternal() ? nullptr : &getBody(); }

    ArrayRef<Type> getCallableResults() {
      return getFunctionType().getResults();
    }

    ArrayAttr getCallableArgAttrs() {
      return getArgAttrs().value_or(nullptr);
    }

    ArrayAttr getCallableResAttrs() {
      return getResAttrs().value_or(nullptr);
    }

    ArrayRef<Type> getArgumentTypes() { return getFunctionType().getInputs(); }

    ArrayRef<Type> getResultTypes() { return getFunctionType().getResults(); }

    /// Required by SymbolOpInterface.
    bool isDeclaration() { return isExternal(); }
  }];
}

def Kapy_ReturnOp : Kapy_Op<
  "return", [ParentOneOf<["FuncOp", "ReduceOp"]>, Pure, ReturnLike, Terminator]
> {
  let arguments = (ins Variadic<AnyType>:$operands);

  let builders = [
    OpBuilder<(ins), "build($_builder, $_state, std::nullopt);">
  ];

  let assemblyFormat = "attr-dict ($operands^ `:` type($operands))?";

  let hasVerifier = 1;
}

#endif // KAPY_DIALECT_KAPY_IR_OPS
