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
//===----------------------------------------------------------------------===//
// 
// This file is modified from the triton project.
// https://github.com/triton-lang/triton
//
//===----------------------------------------------------------------------===//

#ifndef KAPY_DIALECT_KGPU_IR_OPS
#define KAPY_DIALECT_KGPU_IR_OPS

include "mlir/Interfaces/SideEffectInterfaces.td"

include "kapy/Dialect/Kapy/IR/Enums.td"
include "kapy/Dialect/Kapy/IR/OpTraits.td"
include "kapy/Dialect/Kapy/IR/Types.td"
include "kapy/Dialect/Kgpu/IR/Dialect.td"
include "kapy/Dialect/Kgpu/IR/Attrs.td"
include "kapy/Dialect/Kgpu/IR/Types.td"

def SharedMemory : Resource<"mlir::kapy::SharedMemory">;

class Kgpu_Op<string keyword, list<Trait> traits = []>
    : Op<Kgpu_Dialect, keyword, traits # [ValidTensorShape]>;

def Kgpu_ChangeOp : Kgpu_Op<
  "change", [Pure, SameOperandsAndResultShape, SameOperandsAndResultElementType]
> {
  let arguments = (ins Kapy_IntFpTensor:$operand);
  let results = (outs Kapy_IntFpTensor:$result);

  let assemblyFormat = [{
    $operand attr-dict `:` type($operand) `to` type($result)
  }];

  let hasCanonicalizeMethod = 1;
}

def Kgpu_AsyncWaitOp : Kgpu_Op<"async_wait", [Pure]> {
  let arguments = (ins Variadic<Kgpu_Tok>:$input_tokens, I32Attr:$number);
  let results = (outs Kgpu_Tok:$token);

  let builders = [
    OpBuilder<
      (ins "int32_t":$number),
      [{
        auto tokenType = KgpuTokenType::get($_builder.getContext(), 2);
        build($_builder, $_state, tokenType, ValueRange(), number);
      }]
    >,
    OpBuilder<
      (ins "ValueRange":$inputTokens, "int32_t":$number),
      [{
        auto tokenType = KgpuTokenType::get($_builder.getContext(), 2);
        build($_builder, $_state, tokenType, inputTokens, number);
      }]
    >
  ];

  let assemblyFormat = [{
    $input_tokens attr-dict `:` functional-type($input_tokens, $token)
  }];
}

def Kgpu_AsyncCommitOp : Kgpu_Op<"async_commit", [Pure]> {
  let arguments = (ins Variadic<Kgpu_Tok>:$input_tokens);
  let results = (outs Kgpu_Tok:$token);

  let builders = [
    OpBuilder<
      (ins "ValueRange":$inputTokens),
      [{
        auto tokenType = KgpuTokenType::get($_builder.getContext(), 1);
        build($_builder, $_state, tokenType, inputTokens);
      }]
    >
  ];

  let assemblyFormat = [{
    $input_tokens attr-dict `:` functional-type($input_tokens, $token)
  }];
}

def Kgpu_AsyncCopyOp : Kgpu_Op<
  "async_copy", [DeclareOpInterfaceMethods<MemoryEffectsOpInterface>]
> {
  let description = [{
    For each element, copies asynchronously from global memref `$source` to
    shared memref `$target`, for memory access out of range, pad `$target` with
    zero.
  }];

  let arguments = (ins
    Kapy_IntFpMemRef:$source,
    Kapy_IntFpMemRef:$target,
    Kgpu_RegistersLayoutAttr:$layout,
    Kapy_DefaultValuedCacheModifierAttr:$cache_modifier,
    Kapy_DefaultValuedEvictPriorityAttr:$evict_priority,
    DefaultValuedAttr<BoolAttr, "false">:$is_volatile
  );
  let results = (outs Kgpu_Tok:$token);

  let builders = [
    OpBuilder<
      (ins
        "Value":$source,
        "Value":$target,
        "RegistersLayoutAttr":$layout,
        "CacheModifier":$cacheModifier,
        "EvictPriority":$evictPriority,
        "bool":$isVolatile
      ),
      [{
        auto tokenType = KgpuTokenType::get($_builder.getContext(), 0);
        build($_builder, $_state, tokenType, source, target, layout,
              cacheModifier, evictPriority, isVolatile);
      }]
    >
  ];

  let assemblyFormat = [{
    $source `,` $target attr-dict `:` functional-type(operands, $token)
  }];

  let hasVerifier = 1;
}

def Kgpu_LocalAllocOp : Kgpu_Op<
  "local_alloc", [DeclareOpInterfaceMethods<MemoryEffectsOpInterface>]
> {
  let arguments = (ins Optional<Kapy_IntFpTensor>:$operand);
  let results = (outs Kapy_IntFpMemRef:$result);

  let builders = [
    OpBuilder<(ins "Type":$type),
              "build($_builder, $_state, type, std::nullopt);">
  ];

  let assemblyFormat = [{
    ($operand^)? attr-dict `:` functional-type($operand, $result)
  }];

  let hasVerifier = 1;
}

def Kgpu_LocalFreeOp : Kgpu_Op<
  "local_free", [MemoryEffects<[MemFree<SharedMemory>]>]
> {
  let arguments = (ins Kapy_IntFpMemRef:$operand);

  let assemblyFormat = "$operand attr-dict `:` qualified(type($operand))";

  let hasVerifier = 1;
}

def Kgpu_LocalLoadOp : Kgpu_Op<
  "local_load", [DeclareOpInterfaceMethods<MemoryEffectsOpInterface>]
> {
  let arguments = (ins
    Kapy_IntFpMemRef:$source, Optional<Kgpu_Tok>:$input_token
  );
  let results = (outs Kapy_IntFpTensor:$result);

  let builders = [
    OpBuilder<(ins "Type":$type, "Value":$source),
              "build($_builder, $_state, type, source, std::nullopt);">
  ];

  let assemblyFormat = [{
    $source(`,` $input_token^)? attr-dict `:` functional-type(operands, $result)
  }];

  let hasVerifier = 1;
}

#endif // KAPY_DIALECT_KGPU_IR_OPS
