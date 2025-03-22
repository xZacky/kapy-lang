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
    auto sourceValues = unpackLLVMStruct(rewriter, loc, adaptor.getSource());
    ChangeOpHelper helper(op);
    auto map = helper.getShflIdxMap();
    auto ivsNeedShfl = helper.getIvsNeedShfl();
    auto elementType = getSourceElementType(op);
    auto resultType = getResultStructType(op);
    auto loopSize = resultType.getBody().size();
    auto shflKind = NVVM::ShflKind::idx;
    auto equal = arith::CmpIPredicate::eq;
    Value laneId = rewriter.create<LaneIdOp>(loc);
    Value memberMask = arith_constant_i32(0xFFFFFFFF);
    Value groupClamp = arith_constant_i32(0x00001F1F);
    SmallVector<Value> resultValues;
    for (int64_t loopIv = 0; loopIv < loopSize; ++loopIv) {
      Value thisIv = arith_constant_i32(loopIv);
      Value fromId =
          expandAffineExpr(rewriter, loc, map.getResult(0), {laneId, thisIv});
      Value fromIv =
          expandAffineExpr(rewriter, loc, map.getResult(1), {laneId, thisIv});
      if (ivsNeedShfl.empty()) {
        Value result = sourceValues[0];
        for (auto it : llvm::enumerate(sourceValues)) {
          if (it.index() == 0)
            continue;
          result = arith_select(                                         //
              arith_cmpi(equal, arith_constant_i32(it.index()), fromIv), //
              it.value(), result);
        }
        resultValues.push_back(result);
      } else {
        SmallVector<std::pair<int64_t, Value>> sendIvToShfled;
        for (int64_t sendIv : ivsNeedShfl) {
          Value toShfl = sourceValues[sendIv];
          Value shfled = rewriter.create<NVVM::ShflOp>(
              loc, elementType, memberMask, toShfl, fromId, groupClamp,
              shflKind, UnitAttr());
          sendIvToShfled.push_back({sendIv, shfled});
        }
        Value result = sendIvToShfled[0].second;
        for (auto [sendIv, shfled] : sendIvToShfled) {
          if (sendIv == sendIvToShfled[0].first)
            continue;
          result = arith_select(                                     //
              arith_cmpi(equal, arith_constant_i32(sendIv), fromIv), //
              shfled, result);
        }
        resultValues.push_back(result);
      }
    }
    packAndReplace(rewriter, op, resultType, resultValues);
    return success();
  }

private:
  Type getSourceElementType(ChangeOp op) const {
    auto elementType = op.getSource().getType().getElementType();
    return typeConverter->convertType(elementType);
  }

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
