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
include "kapy/Dialect/Kapy/IR/Types.td"
include "kapy/Dialect/Kapy/IR/OpTraits.td"

class Kapy_Op<string keyword, list<Trait> traits = []>
    : Op<Kapy_Dialect, keyword, traits>;

def Kapy_FPToFPOp : Kapy_Op<
  "fptofp", [Pure, SameOperandsAndResultShape, SameOperandsAndResultLayout]
> {
  let description = [{
    Cast between floating-point types, attribute `$rounding_mode` must be set
    for down cast.
  }];

  let arguments = (ins
    Kapy_FloatLike:$source,
    OptionalAttr<Kapy_RoundingModeAttr>:$rounding_mode
  );
  let results = (outs Kapy_FloatLike:$result);

  let assemblyFormat = [{
    $source attr-dict `:` type($source) `to` type($result)
  }];

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

def Kapy_MakeGlobalOp : Kapy_Op<"make_global", [Pure]> {
  let arguments = (ins
    I64:$address,
    I32:$size0,
    I32:$size1,
    I32:$stride0,
    I32:$stride1
  );
  let results = (outs Kapy_Global:$result);

  let assemblyFormat = [{
    $address `[` $size0 `,` $size1 `]` `[` $stride0 `,` $stride1 `]` attr-dict
    `:` qualified(type($result))
  }];

  let extraClassDeclaration = [{
    ValueRange getShape() { return {getSize0(), getSize1()}; }
    ValueRange getStrides() { return {getStride0(), getStride1()}; }
  }];
}

def Kapy_LoadGlobalOp : Kapy_Op<
  "load_global",
  [
    DeclareOpInterfaceMethods<MemoryEffectsOpInterface>,
    TypesMatchWith<"from other type infer result type",
                   "other", "result", "$_self">
  ]
> {
  let description = [{
    For each element, loads `$result` from `$source`.
    For memory access out of range, pads `$result` with `$other`.

    When `$cp_async` is set, will implemented by "cp.async..." instructions.
  }];

  let arguments = (ins
    UnitAttr:$cp_async,
    Kapy_Global:$source,
    I32:$offset0,
    I32:$offset1,
    Kapy_Tensor:$other,
    Kapy_DefaultValuedCacheModifierAttr:$cache_modifier,
    Kapy_DefaultValuedEvictPriorityAttr:$evict_priority,
    DefaultValuedAttr<BoolAttr, "false">:$is_volatile
  );
  let results = (outs Kapy_Tensor:$result);

  let assemblyFormat = [{
    (`cp_async` $cp_async^)? $source `[` $offset0 `,` $offset1 `]` `,` $other
    attr-dict `:` qualified(type($source)) `->` type($other)
  }];

  let extraClassDeclaration = [{
    ValueRange getOffsets() { return {getOffset0(), getOffset1()}; }
  }];
}

def Kapy_StoreGlobalOp : Kapy_Op<
  "store_global",
  [
    DeclareOpInterfaceMethods<MemoryEffectsOpInterface>
  ]
> {
  let description = [{
    For each element, stores `$value` to `$target`.
    For memory access out of range, do nothing.
  }];

  let arguments = (ins
    Kapy_Global:$target,
    I32:$offset0,
    I32:$offset1,
    Kapy_Tensor:$value,
    Kapy_DefaultValuedCacheModifierAttr:$cache_modifier,
    Kapy_DefaultValuedEvictPriorityAttr:$evict_priority
  );

  let assemblyFormat = [{
    $target `[` $offset0 `,` $offset1 `]` `,` $value attr-dict `:`
    qualified(type($target)) `,` type($value)
  }];

  let extraClassDeclaration = [{
    ValueRange getOffsets() { return {getOffset0(), getOffset1()}; }
  }];
}

def Kapy_AtomicRMWGlobalOp : Kapy_Op<
  "atomic_rmw_global",
  [
    DeclareOpInterfaceMethods<MemoryEffectsOpInterface>,
    TypesMatchWith<"from value type infer other type",
                   "value", "other", "$_self">,
    TypesMatchWith<"from value type infer result type",
                   "value", "result", "$_self">
  ]
> {
  let description = [{
    For each element, loads `$result` from `$source`, computes with `$value` and
    stores the computation result back.
    For memory access out of range, pads `$result` with `$other`.
  }];

  let arguments = (ins
    Kapy_AtomicRMWKindAttr:$kind,
    Kapy_Global:$source,
    I32:$offset0,
    I32:$offset1,
    Kapy_Tensor:$value,
    Kapy_Tensor:$other,
    Kapy_MemSemanticAttr:$mem_semantic,
    Kapy_MemoryScopeAttr:$memory_scope
  );
  let results = (outs Kapy_Tensor:$result);

  let assemblyFormat = [{
    $kind $source `[` $offset0 `,` $offset1 `]` `,` $value `,` $other attr-dict
    `:` qualified(type($source)) `->` type($value)
  }];

  let extraClassDeclaration = [{
    ValueRange getOffsets() { return {getOffset0(), getOffset1()}; }
  }];
}

def Kgpu_CpAsyncWaitOp : Kapy_Op<"cp_async_wait", [NoMemoryEffect]> {
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
  let arguments = (ins Kapy_ProgramAxisAttr:$axis);
  let results = (outs I32:$result);

  let assemblyFormat = "$axis attr-dict `:` type($result)";
}

def Kapy_MatmulOp : Kapy_Op<
  "matmul",
  [
    Pure,
    TypesMatchWith<"from acc type infer result type", "acc", "result", "$_self">
  ]
> {
  let description = [{
    Computes `$result = matmul($lhs, $rhs) + $acc`.
  }];

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

def Kapy_ElementwiseExternOp : Kapy_Op<
  "elementwise_extern",
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
    BoolAttr:$is_pure
  );
  let results = (outs Kapy_NumberLike:$result);

  let assemblyFormat = [{
    $operands attr-dict `:` functional-type($operands, $result)
  }];
}

def Kapy_ElementwiseInlineOp : Kapy_Op<
  "elementwise_inline",
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
    BoolAttr:$is_pure
  );
  let results = (outs Kapy_NumberLike:$result);

  let assemblyFormat = [{
    $operands attr-dict `:` functional-type($operands, $result)
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
    ValidTensorShape
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
