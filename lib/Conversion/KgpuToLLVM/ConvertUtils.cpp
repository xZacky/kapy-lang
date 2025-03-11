//===- ConvertUtils.cpp -----------------------------------------*- C++ -*-===//
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

#include "kapy/Conversion/KgpuToLLVM/ConvertUtils.h"
#include "mlir/Dialect/Arith/IR/Arith.h"

using namespace mlir;
using namespace mlir::kapy;
using namespace mlir::LLVM;

Value arith::createConstantF16(RewriterBase &rewriter, Location loc,
                               float value) {
  auto f16Type = FloatType::getF16(rewriter.getContext());
  return rewriter.create<arith::ConstantOp>(loc, f16Type,
                                            rewriter.getF16FloatAttr(value));
}

Value arith::createConstantF32(RewriterBase &rewriter, Location loc,
                               float value) {
  auto f32Type = FloatType::getF32(rewriter.getContext());
  return rewriter.create<arith::ConstantOp>(loc, f32Type,
                                            rewriter.getF32FloatAttr(value));
}

Value arith::createConstantF64(RewriterBase &rewriter, Location loc,
                               double value) {
  auto f64Type = FloatType::getF64(rewriter.getContext());
  return rewriter.create<arith::ConstantOp>(loc, f64Type,
                                            rewriter.getF64FloatAttr(value));
}

Value arith::createConstantI1(RewriterBase &rewriter, Location loc,
                              bool value) {
  auto i1Type = rewriter.getIntegerType(1);
  return rewriter.create<arith::ConstantOp>(loc, i1Type,
                                            IntegerAttr::get(i1Type, value));
}

Value arith::createConstantI32(RewriterBase &rewriter, Location loc,
                               int32_t value) {
  auto i32Type = rewriter.getIntegerType(32);
  return rewriter.create<arith::ConstantOp>(loc, i32Type,
                                            IntegerAttr::get(i32Type, value));
}

Value arith::createConstantI64(RewriterBase &rewriter, Location loc,
                               int64_t value) {
  auto i64Type = rewriter.getIntegerType(64);
  return rewriter.create<arith::ConstantOp>(loc, i64Type,
                                            IntegerAttr::get(i64Type, value));
}

SmallVector<Value> kapy::unpackLLVMStruct(RewriterBase &rewriter, Location loc,
                                          Value llvmStruct) {
  auto bodyTypes = cast<LLVMStructType>(llvmStruct.getType()).getBody();
  SmallVector<Value> values(bodyTypes.size());
  for (unsigned i = 0; i < bodyTypes.size(); ++i)
    values[i] = llvm_extractvalue(bodyTypes[i], llvmStruct, i);
  return values;
}

Value kapy::packToLLVMStruct(RewriterBase &rewriter, Location loc,
                             LLVMStructType structType, ValueRange values) {
  auto bodyTypes = structType.getBody();
  assert(bodyTypes.size() == values.size());
  Value llvmStruct = llvm_undef(structType);
  for (auto it : llvm::enumerate(values)) {
    auto value = it.value();
    auto index = it.index();
    assert(value && value.getType() == bodyTypes[index]);
    llvmStruct = llvm_insertvalue(structType, llvmStruct, value, index);
  }
  return llvmStruct;
}

SmallVector<Value> kapy::unpackI32Value(RewriterBase &rewriter, Location loc,
                                        Type elementType, Value i32Value) {
  SmallVector<Value> elements;
  auto bitWidth = elementType.getIntOrFloatBitWidth();
  auto vectorType = VectorType::get(32 / bitWidth, elementType);
  Value vector = llvm_bitcast(vectorType, i32Value);
  for (unsigned i = 0; i < 32 / bitWidth; ++i)
    elements.push_back(llvm_extractelement(vector, arith_constant_i32(i)));
  return elements;
}

Value kapy::packToI32Value(RewriterBase &rewriter, Location loc,
                           Type elementType, ValueRange elements) {
  auto bitWidth = elementType.getIntOrFloatBitWidth();
  assert(elements.size() == 32 / bitWidth);
  auto vectorType = VectorType::get(32 / bitWidth, elementType);
  Value vector = llvm_undef(vectorType);
  for (unsigned i = 0; i < 32 / bitWidth; ++i)
    vector = llvm_insertelement(vector, elements[i], arith_constant_i32(i));
  return llvm_bitcast(rewriter.getIntegerType(32), vector);
}
