//===- ParallelIdOpToLLVM.cpp -----------------------------------*- C++ -*-===//
//
// This file implements class to make ProgramIdOp, WarpIdOp, LaneIdOp to LLVM
// compatible.
//
//===----------------------------------------------------------------------===//

#include "kapy/Conversion/KgpuToLLVM/PTXBuilder.h"
#include "kapy/Conversion/KgpuToLLVM/Patterns.h"
#include "kapy/Dialect/Kapy/IR/Kapy.h"
#include "mlir/Conversion/LLVMCommon/Pattern.h"
#include "mlir/Dialect/LLVMIR/NVVMDialect.h"

using namespace mlir;
using namespace mlir::kapy;

namespace {

class ProgramIdOpConversion : public ConvertOpToLLVMPattern<ProgramIdOp> {
public:
  using ConvertOpToLLVMPattern<ProgramIdOp>::ConvertOpToLLVMPattern;

  virtual LogicalResult
  matchAndRewrite(ProgramIdOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    assert(op->getNumOperands() == 0 && op->getNumResults() == 1);
    auto i32Type = rewriter.getIntegerType(32);
    if (op.getAxis() == 0) {
      rewriter.replaceOpWithNewOp<NVVM::BlockIdXOp>(op, i32Type);
      return success();
    }
    if (op.getAxis() == 1) {
      rewriter.replaceOpWithNewOp<NVVM::BlockIdYOp>(op, i32Type);
      return success();
    }
    if (op.getAxis() == 2) {
      rewriter.replaceOpWithNewOp<NVVM::BlockIdZOp>(op, i32Type);
      return success();
    }
    return op->emitOpError("axis must be 0, 1, 2");
  }
};

class WarpIdOpConversion : public ConvertOpToLLVMPattern<WarpIdOp> {
public:
  using ConvertOpToLLVMPattern<WarpIdOp>::ConvertOpToLLVMPattern;

  virtual LogicalResult
  matchAndRewrite(WarpIdOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    assert(op->getNumOperands() == 0 && op->getNumResults() == 1);
    auto loc = op.getLoc();
    PTXBuilder builder;
    auto &mov = *builder.create("mov.u32");
    auto *dstOperand = builder.newOperand("=r");
    auto *srcOperand = builder.newConstantOperand("%warpid");
    mov(dstOperand, srcOperand);
    auto i32Type = rewriter.getIntegerType(32);
    auto warpId = builder.launch(rewriter, loc, i32Type, false);
    rewriter.replaceOp(op, warpId);
    return success();
  }
};

class LaneIdOpConversion : public ConvertOpToLLVMPattern<LaneIdOp> {
public:
  using ConvertOpToLLVMPattern<LaneIdOp>::ConvertOpToLLVMPattern;

  virtual LogicalResult
  matchAndRewrite(LaneIdOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    assert(op->getNumOperands() == 0 && op->getNumResults() == 1);
    auto i32Type = rewriter.getIntegerType(32);
    rewriter.replaceOpWithNewOp<NVVM::LaneIdOp>(op, i32Type);
    return success();
  }
};

} // namespace

void kapy::populateParallelIdOpToLLVMConversionPatterns(
    const LLVMTypeConverter &typeConverter, RewritePatternSet &patterns) {
  patterns.add<ProgramIdOpConversion>(typeConverter);
  patterns.add<WarpIdOpConversion, LaneIdOpConversion>(typeConverter);
}
