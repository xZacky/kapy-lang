//===- ChangeOpToLLVM.cpp ------------------------------------*- C++ -*-===//
//
// This file implements class to make ChangeOp to LLVM compatible.
//
//===----------------------------------------------------------------------===//

#include "kapy/Conversion/KgpuToLLVM/ConvertUtils.h"
#include "kapy/Conversion/KgpuToLLVM/Patterns.h"
#include "kapy/Dialect/Kapy/IR/Kapy.h"
#include "kapy/Dialect/Kgpu/IR/Kgpu.h"
#include "kapy/Support/LayoutUtils.h"
#include "mlir/Conversion/LLVMCommon/Pattern.h"
#include "mlir/Dialect/LLVMIR/NVVMDialect.h"

using namespace mlir;
using namespace mlir::kapy;
using namespace mlir::LLVM;

namespace {

class ChangeOpConversion : public ConvertOpToLLVMPattern<ChangeOp> {
public:
  using ConvertOpToLLVMPattern<ChangeOp>::ConvertOpToLLVMPattern;

  virtual LogicalResult
  matchAndRewrite(ChangeOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    assert(op->getNumOperands() == 1 && op->getNumResults() == 1);
    auto loc = op.getLoc();
    auto sourceStruct = adaptor.getSource();
    auto sourceValues = unpackLLVMStruct(rewriter, loc, sourceStruct);
    auto sourceType = cast<LLVMStructType>(sourceStruct.getType());
    auto fromSize = sourceType.getBody().size();
    auto elementType = sourceType.getBody()[0];
    auto vectorType = VectorType::get(fromSize, elementType);
    Value sourceVector = llvm_undef(vectorType);
    for (int64_t loopIv = 0; loopIv < fromSize; ++loopIv) {
      sourceVector = llvm_insertelement(vectorType, sourceVector, //
                                        sourceValues[loopIv],     //
                                        arith_constant_i32(loopIv));
    }
    ChangeOpHelper helper(op);
    auto map = helper.getShflIdxMap();
    auto shflKind = NVVM::ShflKind::idx;
    auto resultType = getResultStructType(op);
    auto thisSize = resultType.getBody().size();
    Value thisId = rewriter.create<LaneIdOp>(loc);
    Value memberMask = arith_constant_i32(0xFFFFFFFF);
    Value groupClamp = arith_constant_i32(0x00200020);
    SmallVector<Value> resultValues;
    for (int64_t loopIv = 0; loopIv < thisSize; ++loopIv) {
      auto mapped = map.compose({0, loopIv});
      if (mapped[0] == 0) {
        Value fromIv = arith_constant_i32(mapped[1]);
        Value toMove = llvm_extractelement(elementType, sourceVector, fromIv);
        resultValues.push_back(toMove);
      } else {
        Value thisIv = arith_constant_i32(loopIv);
        Value fromId =
            expandAffineExpr(rewriter, loc, map.getResult(0), {thisId, thisIv});
        Value fromIv =
            expandAffineExpr(rewriter, loc, map.getResult(1), {thisId, thisIv});
        Value toShfl = llvm_extractelement(elementType, sourceVector, fromIv);
        Value shfled = rewriter.create<NVVM::ShflOp>(
            loc, elementType, memberMask, toShfl, fromId, groupClamp, shflKind,
            UnitAttr());
        resultValues.push_back(shfled);
      }
    }
    packAndReplace(rewriter, op, resultType, resultValues);
    return success();
  }

private:
  LLVMStructType getResultStructType(ChangeOp op) const {
    auto resultType = typeConverter->convertType(op.getType());
    return cast<LLVMStructType>(resultType);
  }
};

} // namespace

void kapy::populateChangeOpToLLVMConversionPattern(
    const LLVMTypeConverter &typeConverter, RewritePatternSet &patterns) {
  patterns.add<ChangeOpConversion>(typeConverter);
}
