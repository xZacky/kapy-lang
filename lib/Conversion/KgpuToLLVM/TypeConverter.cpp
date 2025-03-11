//===- TypeConverter.cpp ----------------------------------------*- C++ -*-===//
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

#include "kapy/Conversion/KgpuToLLVM/TypeConverter.h"
#include "kapy/Dialect/Kapy/IR/Kapy.h"
#include "kapy/Dialect/Kgpu/IR/Kgpu.h"
#include "kapy/Support/CommonUtils.h"
#include "mlir/Dialect/LLVMIR/LLVMTypes.h"

using namespace mlir;
using namespace mlir::kapy;
using namespace mlir::LLVM;

KgpuToLLVMTypeConverter::KgpuToLLVMTypeConverter(
    MLIRContext *context, LowerToLLVMOptions &options,
    const DataLayoutAnalysis *analysis)
    : LLVMTypeConverter(context, options, analysis) {
  addConversion([&](RankedTensorType tensorType) -> std::optional<Type> {
    return convertType(tensorType);
  });
  addConversion([&](Float8E4M3Type f8E4M3Type) -> std::optional<Type> {
    return IntegerType::get(f8E4M3Type.getContext(), 8);
  });
  addConversion([&](Float8E5M2Type f8E5M2Type) -> std::optional<Type> {
    return IntegerType::get(f8E5M2Type.getContext(), 8);
  });
}

Type KgpuToLLVMTypeConverter::convertType(RankedTensorType tensorType) {
  if (inGlobalMemory(tensorType)) {
    SmallVector<Type> bodyTypes;
    auto *context = tensorType.getContext();
    auto pointerType = LLVMPointerType::get(context, 1);
    bodyTypes.push_back(pointerType);
    for (unsigned i = 0; i < 6; ++i)
      bodyTypes.push_back(IntegerType::get(context, 32));
    return LLVMStructType::getLiteral(context, bodyTypes);
  }
  if (inSharedMemory(tensorType)) {
    SmallVector<Type> bodyTypes;
    auto *context = tensorType.getContext();
    auto pointerType = LLVMPointerType::get(context, 3);
    bodyTypes.push_back(pointerType);
    for (unsigned i = 0; i < 6; ++i)
      bodyTypes.push_back(IntegerType::get(context, 32));
    return LLVMStructType::getLiteral(context, bodyTypes);
  }
  if (inRegisterFile(tensorType)) {
    auto *context = tensorType.getContext();
    auto layout = getLayout<FragmentsLayoutAttr>(tensorType);
    auto loopSize = product(layout.getLoopSpace(tensorType.getShape()));
    SmallVector<Type> types(loopSize, convertType(tensorType.getElementType()));
    return LLVMStructType::getLiteral(context, types);
  }
  llvm_unreachable("unsupported memory space");
}
