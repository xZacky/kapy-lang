//===- SplatLikeOpToLLVM.cpp ------------------------------------*- C++ -*-===//
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
#include "kapy/Conversion/KgpuToLLVM/Patterns.h"
#include "kapy/Dialect/Kapy/IR/Kapy.h"
#include "mlir/Conversion/LLVMCommon/Pattern.h"

using namespace mlir;
using namespace mlir::kapy;
using namespace mlir::LLVM;

namespace {

class SplatOpConversion : public ConvertOpToLLVMPattern<SplatOp> {
public:
  using ConvertOpToLLVMPattern<SplatOp>::ConvertOpToLLVMPattern;

  virtual LogicalResult
  matchAndRewrite(SplatOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    assert(op->getNumOperands() == 1 && op->getNumResults() == 1);
    auto loc = op.getLoc();
    auto resultType = getResultStructType(op);
    auto value = adaptor.getSource();
    SmallVector<Value> values(resultType.getBody().size(), value);
    auto resultStruct = packToLLVMStruct(rewriter, loc, resultType, values);
    rewriter.replaceOp(op, resultStruct);
    return success();
  }

private:
  LLVMStructType getResultStructType(SplatOp op) const {
    auto resultType = typeConverter->convertType(op.getType());
    return cast<LLVMStructType>(resultType);
  }
};

class SplatConstantOpConversion
    : public ConvertOpToLLVMPattern<arith::ConstantOp> {
public:
  using ConvertOpToLLVMPattern<arith::ConstantOp>::ConvertOpToLLVMPattern;

  virtual LogicalResult
  matchAndRewrite(arith::ConstantOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    assert(op->getNumOperands() == 0 && op->getNumResults() == 1);
    auto loc = op.getLoc();
    auto resultType = getResultStructType(op);
    auto splatAttr = dyn_cast<SplatElementsAttr>(op.getValue());
    if (!splatAttr)
      return failure();
    auto valueAttr = splatAttr.getSplatValue<TypedAttr>();
    auto value = arith_constant(valueAttr);
    SmallVector<Value> values(resultType.getBody().size(), value);
    auto resultStruct = packToLLVMStruct(rewriter, loc, resultType, values);
    rewriter.replaceOp(op, resultStruct);
    return success();
  }

private:
  LLVMStructType getResultStructType(arith::ConstantOp op) const {
    auto resultType = typeConverter->convertType(op.getType());
    return cast<LLVMStructType>(resultType);
  }
};

} // namespace

void kapy::populateSplatLikeOpToLLVMConversionPatterns(
    const LLVMTypeConverter &typeConverter, RewritePatternSet &patterns) {
  patterns.add<SplatOpConversion, SplatConstantOpConversion>(typeConverter);
}
