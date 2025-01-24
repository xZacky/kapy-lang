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
include "mlir/Interfaces/InferTypeOpInterface.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "kapy/Dialect/Kapy/IR/Dialect.td"
include "kapy/Dialect/Kapy/IR/Enums.td"
include "kapy/Dialect/Kapy/IR/OpTraits.td"
include "kapy/Dialect/Kapy/IR/Types.td"

class Kapy_Op<string keyword, list<Trait> traits = []>
    : Op<Kapy_Dialect, keyword, traits # [ValidTensorShape]>;

// Using cast operations in the Arith dialect:
//   fptoui, fptosi, uitofp, sitofp,
//   extf, tructf,
//   extui, extsi, tructi
//   bitcast

def Kapy_FpToFpOp : Kapy_Op<
  "fptofp", [Pure, SameOperandsAndResultShape, SameOperandsAndResultLayout]
> {
  let description = [{
    Cast between floating-point types, attribute `$rounding_mode` must be set
    for downcast case.
  }];

  let arguments = (ins
    Kapy_FpLike:$operand,
    OptionalAttr<Kapy_RoundingModeAttr>:$rounding_mode
  );
  let results = (outs Kapy_FpLike:$result);

  let assemblyFormat = [{
    $operand attr-dict `:` type($operand) `to` type($result)
  }];

  let hasVerifier = 1;
}

// Using arithmetic operations in the Arith dialect.
// Using mathematic operations in the Math dialect.

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
    Kapy_FpLike:$source,
    Kapy_FpLike:$low,
    Kapy_FpLike:$high,
    Kapy_PropagateNanAttr:$propagate_nan
  );
  let results = (outs Kapy_FpLike:$result);

  let assemblyFormat = "$source `,` $low `,` $high attr-dict `:` type($result)";
}

def Kapy_MulhiUIOp : Kapy_Op<
  "mulhiui", [Elementwise, Pure, SameOperandsAndResultType]
> {
  let description = [{
    Computes the most significant N bits of the 2N bits product of `$lhs` and
    `$rhs`.
  }];

  let arguments = (ins Kapy_IntLike:$lhs, Kapy_IntLike:$rhs);
  let results = (outs Kapy_IntLike:$result);

  let assemblyFormat = "$lhs `,` $rhs attr-dict `:` type($result)";
}

def Kapy_ArangeOp : Kapy_Op<"arange", [Pure]> {
  let arguments = (ins I32Attr:$start, I32Attr:$end);
  let results = (outs Kapy_I32Tensor:$result);

  let assemblyFormat = "attr-dict `:` type($result)";

  let hasFolder = 1;
  let hasVerifier = 1;
}

def Kapy_MakeMemRefOp
    : Kapy_Op<"make_memref", [Pure, SameVariadicOperandSize]> {
  let description = [{
    Returns a global memref with the given `$pointer`, `$parent_shape` and
    `$strides`.
  }];

  let arguments = (ins
    Kapy_Ptr:$pointer, Variadic<I32>:$parent_shape, Variadic<I32>:$strides
  );
  let results = (outs Kapy_IntFpMemRef:$result);

  let assemblyFormat = [{
    $pointer `,` `[` $parent_shape `]` `,` `[` $strides `]` attr-dict `:`
    qualified(type($pointer)) `->` qualified(type($result))
  }];

  let hasVerifier = 1;
}

def Kapy_MoveMemRefOp : Kapy_Op<
  "move_memref",
  [
    Pure,
    TypesMatchWith<"from source type infer result type",
                   "source", "result", "$_self">
  ]
> {
  let description = [{
    Moves the memref `$source` by add its pointer with `$offset`.
  }];

  let arguments = (ins Kapy_IntFpMemRef:$source, I32:$offset);
  let results = (outs Kapy_IntFpMemRef:$result);

  let assemblyFormat = [{
    $source `,` $offset attr-dict `:` qualified(type($source))
  }];

  let hasCanonicalizeMethod = 1;
  let hasVerifier = 1;
}

def Kapy_LoadOp : Kapy_Op<
  "load", [DeclareOpInterfaceMethods<MemoryEffectsOpInterface>]
> {
  let description = [{
    For each element, loads `$result` from `$source`, for memory access out of
    range, pad `$result` with zero or NaN according to `$padding_option`.
  }];

  let arguments = (ins
    Kapy_IntFpMemRef:$source,
    Kapy_DefaultValuedPaddingOptionAttr:$padding_option,
    Kapy_DefaultValuedCacheModifierAttr:$cache_modifier,
    Kapy_DefaultValuedEvictPriorityAttr:$evict_priority,
    DefaultValuedAttr<BoolAttr, "false">:$is_volatile
  );
  let results = (outs Kapy_IntFpLike:$result);

  let assemblyFormat = [{
    $source attr-dict `:` qualified(type($source)) `->` type($result)
  }];

  let hasVerifier = 1;
}

def Kapy_StoreOp : Kapy_Op<
  "store", [DeclareOpInterfaceMethods<MemoryEffectsOpInterface>]
> {
  let description = [{
    For each element, stores `$value` to `$target`, for memory access out of
    range, do nothing.
  }];

  let arguments = (ins
    Kapy_IntFpMemRef:$target,
    Kapy_IntFpLike:$value,
    Kapy_DefaultValuedCacheModifierAttr:$cache_modifier,
    Kapy_DefaultValuedEvictPriorityAttr:$evict_priority
  );

  let assemblyFormat = [{
    $target `,` $value attr-dict `:` qualified(type($target)) `,` type($value)
  }];

  let hasVerifier = 1;
}

def Kapy_AtomicRMWOp : Kapy_Op<
  "atomic_rmw",
  [
    SameOperandsAndResultLayout,
    DeclareOpInterfaceMethods<MemoryEffectsOpInterface>,
    TypesMatchWith<"from value type infer result type",
                   "value", "result", "$_self">
  ]
> {
  let description = [{
    For each element, loads `$result` from `$source`, computes with `$value` and
    stores the computation result back, for memory access out of range, just pad
    `$result` with zero.
  }];

  let arguments = (ins
    Kapy_AtomicRMWKindAttr:$kind,
    Kapy_IntFpMemRef:$source,
    Kapy_IntFpLike:$value,
    Kapy_MemSemanticAttr:$mem_semantic
  );
  let results = (outs Kapy_IntFpLike:$result);

  let assemblyFormat = [{
    $kind $source `,` $value attr-dict `:` qualified(type($source)) `->`
    type($value)
  }];

  let hasVerifier = 1;
}

def Kapy_AtomicCASOp : Kapy_Op<
  "atomic_cas",
  [
    SameOperandsAndResultLayout,
    DeclareOpInterfaceMethods<MemoryEffectsOpInterface>,
    TypesMatchWith<"from value type infer result type",
                   "value", "result", "$_self">,
    TypesMatchWith<"from value type infer flag type",
                   "value", "flag", "$_self">
  ]
> {
  let description = [{
    For each element, loads `$result` from `$source`, compares it with `$flag`,
    if `$result == $flag`, stores `$value` back, for memory access out of range,
    just pad `$result` with zero.
  }];

  let arguments = (ins
    Kapy_IntFpMemRef:$source,
    Kapy_IntFpLike:$flag,
    Kapy_IntFpLike:$value,
    Kapy_MemSemanticAttr:$mem_semantic
  );
  let results = (outs Kapy_IntFpLike:$result);

  let assemblyFormat = [{
    $source `,` $flag `,` $value attr-dict `:` qualified(type($source)) `->`
    type($value)
  }];

  let hasVerifier = 1;
}

def Kapy_SplatOp : Kapy_Op<"splat", [Pure, SameOperandsAndResultElementType]> {
  let arguments = (ins Kapy_IntFp:$operand);
  let results = (outs Kapy_IntFpTensor:$result);

  let assemblyFormat = [{
    $operand attr-dict `:` type($operand) `->` type($result)
  }];

  let hasFolder = 1;
}

def Kapy_UnsqueezeOp : Kapy_Op<
  "unsqueeze",
  [
    Pure,
    SameOperandsAndResultElementType,
    DeclareOpInterfaceMethods<InferTypeOpInterface>
  ]
> {
  let arguments = (ins Kapy_IntFpTensor:$operand, I32Attr:$axis);
  let results = (outs Kapy_IntFpTensor:$result);

  let assemblyFormat = [{
    $operand attr-dict `:` type($operand) `->` type($result)
  }];

  let hasCanonicalizeMethod = 1;
  let hasFolder = 1;
  let hasVerifier = 1;
}

def Kapy_BroadcastOp : Kapy_Op<
  "broadcast",
  [Pure, SameOperandsAndResultElementType, SameOperandsAndResultLayout]
> {
  let description = [{
    Broadcasts the tensor dimensions with size 1 to a new size.
  }];

  let arguments = (ins Kapy_IntFpTensor:$operand);
  let results = (outs Kapy_IntFpTensor:$result);

  let assemblyFormat = [{
    $operand attr-dict `:` type($operand) `->` type($result)
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
  let arguments = (ins Kapy_IntFpTensor:$operand);
  let results = (outs Kapy_IntFpTensor:$result);

  let assemblyFormat = [{
    $operand attr-dict `:` type($operand) `->` type($result)
  }];

  let hasCanonicalizeMethod = 1;
  let hasFolder = 1;
  let hasVerifier = 1;
}

def Kapy_ProgramIdOp : Kapy_Op<"program_id", [Pure]> {
  let arguments = (ins Kapy_ProgramAxisAttr:$axis);
  let results = (outs I32:$result);

  let assemblyFormat = "$axis attr-dict `:` type($result)";
}

def Kapy_MatmulOp : Kapy_Op<
  "matmul",
  [
    Pure,
    TypesMatchWith<"from accum type infer result type",
                   "accum", "result", "$_self">
  ]
> {
  let description = [{
    Computes `$result = matmul($lhs, $rhs) + $accum`.
    Attribute `$matmul_format` describes how to exercise the tensor core when
    `$lhs` and `$rhs` are f32. It can be one of: `ieee, tf32, tf32x3`.
    ieee: Do not use tensor core, implement matmul in software.
    tf32: Use tensor core with tf32.
    tf32x3: Implement matmul with the 3xtf32 trick.
    If `$lhs` and `$rhs` are not f32, this attrbiute is ignored.
  }];

  let arguments = (ins
    Kapy_IntFpTensor:$lhs,
    Kapy_IntFpTensor:$rhs,
    Kapy_IntFpTensor:$accum,
    Kapy_DefaultValuedMatmulFormatAttr:$matmul_format
  );
  let results = (outs Kapy_IntFpTensor:$result);

  let assemblyFormat = [{
    $lhs `,` $rhs `,` $accum attr-dict `:` type($lhs) `,` type($rhs) `->`
    type($accum)
  }];

  let hasVerifier = 1;
}

def Kapy_ReduceOp : Kapy_Op<
  "reduce",
  [
    IsolatedFromAbove,
    Pure,
    SameOperandsAndResultElementType,
    KapySingleBlockTerminator,
    DeclareOpInterfaceMethods<InferTypeOpInterface>
  ]
> {
  let arguments = (ins Kapy_IntFpTensor:$operand, I32Attr:$axis);
  let regions = (region SizedRegion<1>:$lambda);
  let results = (outs Kapy_IntFpLike:$result);

  let hasCustomAssemblyFormat = 1;
  let hasRegionVerifier = 1;
}

// def Kapy_ScanOp : Kapy_Op<
//   "scan",
//   [
//     IsolatedFromAbove,
//     Pure,
//     SameOperandsAndResultType,
//     KapySingleBlockTerminator
//   ]
// > {
//   let arguments = (ins
//     Kapy_IntFpTensor:$operand, I32Attr:$axis, BoolAttr:$reverse
//   );
//   let regions = (region SizedRegion<1>:$lambda);
//   let results = (outs Kapy_IntFpTensor:$result);
//
//   let hasCustomAssemblyFormat = 1;
//   let hasRegionVerifier = 1;
// }

def Kapy_YieldOp : Kapy_Op<
  "yield",
  [
    Pure,
    ReturnLike,
    Terminator,
    // ParentOneOf<["ReduceOp", "ScanOp"]>
    HasParent<"ReduceOp">
  ]
> {
  let arguments = (ins Kapy_IntFp:$operand);
  
  let assemblyFormat = "$operand attr-dict `:` type($operand)";
}

def Kapy_ElementwiseExternOp : Kapy_Op<
  "elementwise_extern",
  [
    Elementwise,
    SameOperandsAndResultLayout,
    DeclareOpInterfaceMethods<MemoryEffectsOpInterface>
  ]
> {
  let description = [{
    Calls an external function `$sym_name` implemented in `$lib_path/$lib_name`
    with `$operands`.
  }];

  let arguments = (ins
    Variadic<Kapy_IntFpLike>:$operands,
    StrAttr:$lib_name,
    StrAttr:$lib_path,
    StrAttr:$sym_name,
    BoolAttr:$is_pure
  );
  let results = (outs Kapy_IntFpLike:$result);

  let assemblyFormat = [{
    $operands attr-dict `:` functional-type($operands, $result)
  }];
}

def Kapy_ElementwiseInlineOp : Kapy_Op<
  "elementwise_inline",
  [
    Elementwise,
    SameOperandsAndResultLayout,
    DeclareOpInterfaceMethods<MemoryEffectsOpInterface>
  ]
> {
  let description = [{
    Runs inline assembly applying an elementwise operation to a group of packed
    values from `$operands`.
    The assembly block is given `num_packed_values` values at a time and repeat
    until all the values in `$operands` are processed.
  }];

  let arguments = (ins
    Variadic<Kapy_IntFpLike>:$operands,
    StrAttr:$asm_string,
    StrAttr:$constraints,
    I32Attr:$num_packed_values,
    BoolAttr:$is_pure
  );
  let results = (outs Variadic<Kapy_IntFpLike>:$result);

  let assemblyFormat = [{
    $operands attr-dict `:` functional-type($operands, $result)
  }];

  let hasVerifier = 1;
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
    OpAsmOpInterface
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
        "FunctionType":$type,
        CArg<"ArrayRef<NamedAttribute>", "{}">:$attrs,
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
  "return", [HasParent<"FuncOp">, Pure, ReturnLike, Terminator]
> {
  let arguments = (ins Variadic<AnyType>:$operands);

  let builders = [
    OpBuilder<(ins), "build($_builder, $_state, std::nullopt);">
  ];

  let assemblyFormat = "attr-dict ($operands^ `:` type($operands))?";

  let hasVerifier = 1;
}

#endif // KAPY_DIALECT_KAPY_IR_OPS
