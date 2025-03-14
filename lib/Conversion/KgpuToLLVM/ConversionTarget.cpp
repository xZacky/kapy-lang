//===- ConversionTarget.cpp -------------------------------------*- C++ -*-===//
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

#include "kapy/Conversion/KgpuToLLVM/ConversionTarget.h"
#include "kapy/Conversion/KgpuToLLVM/TypeConverter.h"
#include "kapy/Dialect/Kapy/IR/Kapy.h"
#include "kapy/Dialect/Kgpu/IR/Kgpu.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/LLVMIR/NVVMDialect.h"

using namespace mlir;
using namespace mlir::kapy;

KgpuToLLVMConversionTarget::KgpuToLLVMConversionTarget(
    MLIRContext *context, const KgpuToLLVMTypeConverter &typeConverter)
    : ConversionTarget(*context) {
  addLegalDialect<LLVM::LLVMDialect, NVVM::NVVMDialect>();
  addLegalOp<UnrealizedConversionCastOp>();

  addIllegalDialect<KapyDialect, KgpuDialect>();
  addLegalOp<ProgramIdOp, WarpIdOp, LaneIdOp>();

  addDynamicallyLegalDialect<arith::ArithDialect, //
                             math::MathDialect,   //
                             scf::SCFDialect>([&](Operation *op) {
    bool hasLegalRegions = true;
    for (auto &region : op->getRegions())
      hasLegalRegions &= typeConverter.isLegal(op);
    return hasLegalRegions && typeConverter.isLegal(op);
  });

  addIllegalOp<arith::DivFOp>();
  addDynamicallyLegalOp<arith::AddFOp>([&](arith::AddFOp op) {
    return typeConverter.isLegal(op) && !op.getType().isBF16();
  });
  addDynamicallyLegalOp<arith::SubFOp>([&](arith::SubFOp op) {
    return typeConverter.isLegal(op) && !op.getType().isBF16();
  });
  addDynamicallyLegalOp<arith::MulFOp>([&](arith::MulFOp op) {
    return typeConverter.isLegal(op) && !op.getType().isBF16();
  });
  addDynamicallyLegalOp<arith::TruncFOp>([&](arith::TruncFOp op) {
    return typeConverter.isLegal(op) && !op.getType().isBF16();
  });
  addDynamicallyLegalOp<arith::ExtFOp>([&](arith::ExtFOp op) {
    return typeConverter.isLegal(op) && !op.getType().isBF16();
  });
  addDynamicallyLegalOp<arith::SIToFPOp>([&](arith::SIToFPOp op) {
    return typeConverter.isLegal(op) && !op.getType().isBF16();
  });
  addDynamicallyLegalOp<arith::FPToSIOp>([&](arith::FPToSIOp op) {
    return typeConverter.isLegal(op) && !op.getType().isBF16();
  });
  addDynamicallyLegalOp<math::ExpOp>([&](math::ExpOp op) {
    return typeConverter.isLegal(op) && !op.getType().isF32();
  });
};
