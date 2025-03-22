//===- ReduceOpToLLVM.cpp ------------------------------------*- C++ -*-===//
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
#include "kapy/Support/LayoutUtils.h"
#include "mlir/Conversion/LLVMCommon/Pattern.h"
#include "mlir/Dialect/LLVMIR/NVVMDialect.h"

using namespace mlir;
using namespace mlir::kapy;
using namespace mlir::LLVM;

namespace {

class ReduceOpConversion : public ConvertOpToLLVMPattern<ReduceOp> {
public:
  using ConvertOpToLLVMPattern<ReduceOp>::ConvertOpToLLVMPattern;

  virtual LogicalResult
  matchAndRewrite(ReduceOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    assert(op->getNumOperands() == 1 && op->getNumResults() == 1);
    auto loc = op.getLoc();
    auto axis = op.getAxis();
    auto sourceType = op.getSource().getType();
    auto sourceLayout = getLayout<FragmentsLayoutAttr>(sourceType);
    auto loopSpace = sourceLayout.getLoopSpace(sourceType.getShape());
    auto sourceValues = unpackLLVMStruct(rewriter, loc, adaptor.getSource());
    SmallVector<Value> accValues;
    if (axis == 0) {
      for (int64_t loopIv1 = 0; loopIv1 < loopSpace[1]; ++loopIv1) {
        if (sourceLayout.isRowMajor())
          accValues.push_back(sourceValues[loopIv1]);
        else
          accValues.push_back(sourceValues[loopIv1 * loopSpace[0]]);
      }
    } else {
      for (int64_t loopIv0 = 0; loopIv0 < loopSpace[0]; ++loopIv0) {
        if (sourceLayout.isRowMajor())
          accValues.push_back(sourceValues[loopIv0 * loopSpace[1]]);
        else
          accValues.push_back(sourceValues[loopIv0]);
      }
    }
    // Reduce within lane.
    if (axis == 0) {
      for (int64_t loopIv0 = 1; loopIv0 < loopSpace[0]; ++loopIv0) {
        for (int64_t loopIv1 = 0; loopIv1 < loopSpace[1]; ++loopIv1) {
          int64_t loopIv;
          if (sourceLayout.isRowMajor())
            loopIv = loopIv0 * loopSpace[1] + loopIv1;
          else
            loopIv = loopIv0 + loopIv1 * loopSpace[0];
          Value curValue = sourceValues[loopIv];
          accumulate(rewriter, op.getRegion(), accValues[loopIv1], curValue);
        }
      }
    } else {
      for (int64_t loopIv0 = 0; loopIv0 < loopSpace[0]; ++loopIv0) {
        for (int64_t loopIv1 = 1; loopIv1 < loopSpace[1]; ++loopIv1) {
          int64_t loopIv;
          if (sourceLayout.isRowMajor())
            loopIv = loopIv0 * loopSpace[1] + loopIv1;
          else
            loopIv = loopIv0 + loopIv1 * loopSpace[0];
          Value curValue = sourceValues[loopIv];
          accumulate(rewriter, op.getRegion(), accValues[loopIv0], curValue);
        }
      }
    }
    // Reduce within warp.
    ReduceOpHelper helper(op);
    auto laneOffset = helper.getLaneOffset();
    auto numShfls = helper.getNumShfls();
    auto shflKind = NVVM::ShflKind::bfly;
    auto elementType = getSourceElementType(op);
    Value memberMask = arith_constant_i32(0xFFFFFFFF);
    Value groupClamp = arith_constant_i32(0x00001F1F);
    for (unsigned i = 0; i < numShfls; ++i) {
      Value offset = arith_constant_i32(laneOffset * exp2(i));
      for (unsigned j = 0; j < accValues.size(); ++j) {
        Value toShfl = accValues[j];
        Value shfled = rewriter.create<NVVM::ShflOp>(
            loc, elementType, memberMask, toShfl, offset, groupClamp, shflKind,
            UnitAttr());
        accumulate(rewriter, op.getRegion(), accValues[j], shfled);
      }
    }
    auto resultType = op.getType();
    auto resultLayout = getLayout<FragmentsLayoutAttr>(resultType);
    loopSpace = resultLayout.getLoopSpace(resultType.getShape());
    SmallVector<Value> resultValues(product(loopSpace));
    for (int64_t loopIv0 = 0; loopIv0 < loopSpace[0]; ++loopIv0) {
      for (int64_t loopIv1 = 0; loopIv1 < loopSpace[1]; ++loopIv1) {
        int64_t loopIv;
        if (sourceLayout.isRowMajor())
          loopIv = loopIv0 * loopSpace[1] + loopIv1;
        else
          loopIv = loopIv0 + loopIv1 * loopSpace[0];
        resultValues[loopIv] = accValues[axis == 1 ? loopIv0 : loopIv1];
      }
    }
    auto structType = getResultStructType(op);
    packAndReplace(rewriter, op, structType, resultValues);
    return success();
  }

private:
  void accumulate(ConversionPatternRewriter &rewriter, Region &region,
                  Value &accValue, Value curValue) const {
    // Create a new copy of the reduce block and inline it.
    auto *parent = rewriter.getBlock()->getParent();
    rewriter.cloneRegionBefore(region, &parent->front());
    auto &block = parent->front();
    auto returnOp = cast<kapy::ReturnOp>(block.getTerminator());
    SmallVector<Value, 2> argValues(2);
    argValues[0] = accValue;
    argValues[1] = curValue;
    auto &ip = *rewriter.getInsertionPoint();
    rewriter.inlineBlockBefore(&block, &ip, argValues);
    accValue = returnOp.getOperand(0);
    // Delete the terminator, which is no longer used.
    rewriter.eraseOp(returnOp);
  }

  Type getSourceElementType(ReduceOp op) const {
    auto elementType = op.getSource().getType().getElementType();
    return typeConverter->convertType(elementType);
  }

  LLVMStructType getResultStructType(ReduceOp op) const {
    auto resultType = typeConverter->convertType(op.getType());
    return cast<LLVMStructType>(resultType);
  }
};

} // namespace

void kapy::populateReduceOpToLLVMConversionPattern(
    const LLVMTypeConverter &typeConverter, RewritePatternSet &patterns) {
  patterns.add<ReduceOpConversion>(typeConverter);
}
