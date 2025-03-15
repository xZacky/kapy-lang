//===- BroadcastOpToLLVM.cpp ------------------------------------*- C++ -*-===//
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
#include "kapy/Dialect/Kgpu/IR/Kgpu.h"
#include "kapy/Support/CommonUtils.h"
#include "mlir/Conversion/LLVMCommon/Pattern.h"

using namespace mlir;
using namespace mlir::kapy;
using namespace mlir::LLVM;

namespace {

class BroadcastOpConversion : public ConvertOpToLLVMPattern<BroadcastOp> {
public:
  using ConvertOpToLLVMPattern<BroadcastOp>::ConvertOpToLLVMPattern;

  virtual LogicalResult
  matchAndRewrite(BroadcastOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    assert(op->getNumOperands() == 1 && op->getNumResults() == 1);
    auto loc = op.getLoc();

    auto sourceType = op.getSource().getType();
    auto resultType = op.getType();
    auto sourceLayout = getLayout<FragmentsLayoutAttr>(sourceType);
    auto resultLayout = getLayout<FragmentsLayoutAttr>(resultType);
    auto oldMap = sourceLayout.getAffineMap(sourceType.getShape(), 3);
    auto newMap = resultLayout.getAffineMap(resultType.getShape(), 2);

    auto sourceStruct = adaptor.getSource();
    auto axis = getBroadcastAxis(op);
    auto elementType = getSourceElementType(op);
    auto loopSize = product(resultLayout.getLoopSpace(resultType.getShape()));

    SmallVector<Value> resultValues;
    for (int64_t loopIv = 0; loopIv < loopSize; ++loopIv) {
      auto indices = newMap.compose({0, loopIv});
      indices[axis] = 0;
      auto inputId = oldMap.compose(indices)[0];
      resultValues.push_back(
          llvm_extractvalue(elementType, sourceStruct, inputId));
    }

    auto structType = getResultStructType(op);
    auto resultStruct =
        packToLLVMStruct(rewriter, loc, structType, resultValues);
    rewriter.replaceOp(op, resultStruct);
    return success();
  }

private:
  unsigned getBroadcastAxis(BroadcastOp op) const {
    auto oldShape = op.getSource().getType().getShape();
    auto newShape = op.getType().getShape();
    for (unsigned i = 0; i < 2; ++i)
      if (oldShape[i] == 1 && newShape[i] != 1)
        return i;
    llvm_unreachable("can not find the broadcast axis");
  }

  Type getSourceElementType(BroadcastOp op) const {
    auto elementType = getElementTypeOrSelf(op.getSource().getType());
    return typeConverter->convertType(elementType);
  }

  LLVMStructType getResultStructType(BroadcastOp op) const {
    auto resultType = typeConverter->convertType(op.getType());
    return cast<LLVMStructType>(resultType);
  }
};

} // namespace

void kapy::populateBroadcastOpToLLVMConversionPattern(
    const LLVMTypeConverter &typeConverter, RewritePatternSet &patterns) {
  patterns.add<BroadcastOpConversion>(typeConverter);
}
