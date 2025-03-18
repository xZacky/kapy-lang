//===- CpAsyncOpToLLVM.cpp -------------------------------------*- C++ -*-===//
//
// This file implements class to make CpAsyncGlobalToSharedOp and
// CpAsyncWaitGroupOp to LLVM compatible.
//
//===----------------------------------------------------------------------===//

#include "kapy/Conversion/KgpuToLLVM/ConvertUtils.h"
#include "kapy/Conversion/KgpuToLLVM/PTXBuilder.h"
#include "kapy/Conversion/KgpuToLLVM/Patterns.h"
#include "kapy/Dialect/Kapy/IR/Kapy.h"
#include "kapy/Dialect/Kgpu/IR/Kgpu.h"
#include "mlir/Conversion/LLVMCommon/Pattern.h"
#include "mlir/Dialect/LLVMIR/NVVMDialect.h"

using namespace mlir;
using namespace mlir::kapy;
using namespace mlir::LLVM;

namespace {

class CpAsyncGlobalToSharedOpConversion
    : public ConvertOpToLLVMPattern<CpAsyncGlobalToSharedOp> {
public:
  using ConvertOpToLLVMPattern<CpAsyncGlobalToSharedOp>::ConvertOpToLLVMPattern;

  virtual LogicalResult
  matchAndRewrite(CpAsyncGlobalToSharedOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    assert(op->getNumOperands() == 3 && op->getNumResults() == 0);
    auto loc = op.getLoc();

    auto sourceType = op.getSource().getType();
    auto globalLayout = getLayout<Strided2dLayoutAttr>(sourceType);

    auto targetType = op.getTarget().getType();
    auto sharedLayout = getLayout<SwizzlingLayoutAttr>(targetType);
    auto bankParam = sharedLayout.getBankParam();
    auto lineParam = sharedLayout.getLineParam();

    auto loaderType = op.getLoader().getType();
    auto loaderLayout = getLayout<FragmentsLayoutAttr>(loaderType);
    auto laneLoops = loaderLayout.getLaneLoops();
    auto loopSpace = loaderLayout.getLoopSpace(sourceType.getShape());
    auto map = loaderLayout.getAffineMap(sourceType.getShape(), 2);

    auto bitWidth = getIntOrFloatBitWidth(sourceType);
    auto simdSize = laneLoops[globalLayout.isColMajor() ? 0 : 1];
    simdSize = std::min(simdSize, bankParam * 32 / bitWidth);
    auto copySize = std::to_string(simdSize * bitWidth / 8);

    DenseMap<int64_t, SmallVector<int64_t, 8>> instIdToLoopIvs;
    for (int64_t loopIv0 = 0; loopIv0 < loopSpace[0]; ++loopIv0) {
      for (int64_t loopIv1 = 0; loopIv1 < loopSpace[1]; ++loopIv1) {
        int64_t instId;
        if (globalLayout.isRowMajor())
          instId = (loopIv0 * loopSpace[1] + loopIv1) / simdSize;
        else
          instId = (loopIv0 + loopIv1 * loopSpace[0]) / simdSize;
        int64_t loopIv;
        if (loaderLayout.isRowMajor())
          loopIv = loopIv0 * loopSpace[1] + loopIv1;
        else
          loopIv = loopIv0 + loopIv1 * loopSpace[0];
        instIdToLoopIvs[instId].push_back(loopIv);
      }
    }
    auto numInsts = instIdToLoopIvs.size();

    auto ldPointerType = LLVMPointerType::get(getContext(), 1);
    auto stPointerType = LLVMPointerType::get(getContext(), 3);
    auto elementType = getSourceElementType(op);
    auto i32Type = rewriter.getIntegerType(32);
    auto voidType = LLVMVoidType::get(getContext());
    auto lessThan = arith::CmpIPredicate::ult;

    Value sourceStruct = adaptor.getSource();
    Value targetStruct = adaptor.getTarget();
    Value ldPointer = llvm_extractvalue(ldPointerType, sourceStruct, 0);
    Value ldStart0 = llvm_extractvalue(i32Type, sourceStruct, 1);
    Value ldStart1 = llvm_extractvalue(i32Type, sourceStruct, 2);
    Value ldEnd0 = llvm_extractvalue(i32Type, sourceStruct, 3);
    Value ldEnd1 = llvm_extractvalue(i32Type, sourceStruct, 4);
    Value one = arith_constant_i32(1);
    Value ldStride0 = globalLayout.isRowMajor()
                          ? llvm_extractvalue(i32Type, sourceStruct, 5)
                          : one;
    Value ldStride1 = globalLayout.isColMajor()
                          ? llvm_extractvalue(i32Type, sourceStruct, 6)
                          : one;
    Value stPointer = llvm_extractvalue(stPointerType, targetStruct, 0);
    Value stStart0 = llvm_extractvalue(i32Type, targetStruct, 1);
    Value stStart1 = llvm_extractvalue(i32Type, targetStruct, 2);
    Value stEnd0 = llvm_extractvalue(i32Type, targetStruct, 3);
    Value stEnd1 = llvm_extractvalue(i32Type, targetStruct, 4);
    Value stStride0 = arith_constant_i32(sharedLayout.getStride0());
    Value stStride1 = arith_constant_i32(sharedLayout.getStride1());
    Value bankValue = arith_constant_i32(bankParam);
    Value lineValue = arith_constant_i32(lineParam);
    Value byteWidth = arith_constant_i32(bitWidth / 8);
    Value const128 = arith_constant_i32(128);
    Value four = arith_constant_i32(4);
    Value bankSize = arith_constant_i32(32 / bitWidth);
    Value lineSize = arith_constant_i32(1024 / bitWidth);
    Value laneId = rewriter.create<LaneIdOp>(loc);

    SmallVector<Value> ldPointers, stPointers, predicates;
    for (int64_t instId = 0; instId < numInsts; ++instId) {
      Value loopIv = arith_constant_i32(instIdToLoopIvs[instId][0]);
      Value index0 =
          expandAffineExpr(rewriter, loc, map.getResult(0), {laneId, loopIv});
      Value index1 =
          expandAffineExpr(rewriter, loc, map.getResult(1), {laneId, loopIv});

      Value ldIndex0 = arith_addi(ldStart0, index0);
      Value ldIndex1 = arith_addi(ldStart1, index1);
      ldPointers.push_back(
          llvm_getelementptr(ldPointerType, elementType, ldPointer,
                             arith_addi(arith_muli(ldIndex0, ldStride0),
                                        arith_muli(ldIndex1, ldStride1))));

      Value stIndex0 = arith_addi(stStart0, index0);
      Value stIndex1 = arith_addi(stStart1, index1);
      // before swizzling
      Value elemOffset = arith_addi(arith_muli(stIndex0, stStride0),
                                    arith_muli(stIndex1, stStride1));
      Value byteOffset = arith_muli(elemOffset, byteWidth);
      Value bankId = arith_divui(arith_remui(byteOffset, const128), four);
      Value lineId = arith_divui(byteOffset, const128);
      // apply swizzling
      Value xorResult = arith_xori(arith_divui(bankId, bankValue),
                                   arith_remui(lineId, lineValue));
      Value newBankId = arith_addi(arith_muli(xorResult, bankValue),
                                   arith_remui(bankId, bankValue));
      Value bankOffset = arith_muli(newBankId, bankSize);
      Value lineOffset = arith_muli(lineId, lineSize);
      stPointers.push_back(
          llvm_getelementptr(stPointerType, elementType, stPointer,
                             arith_addi(bankOffset, lineOffset)));

      predicates.push_back(
          arith_andi(arith_andi(arith_cmpi(lessThan, ldIndex0, ldEnd0),
                                arith_cmpi(lessThan, ldIndex1, ldEnd1)),
                     arith_andi(arith_cmpi(lessThan, stIndex0, stEnd0),
                                arith_cmpi(lessThan, stIndex1, stEnd1))));
    }

    for (int64_t instId = 0; instId < numInsts; ++instId) {
      PTXBuilder builder;
      auto &cp = *builder.create("cp.async");
      cp.o(stringifyCacheModifier(op.getCacheModifier()).str());
      cp.shared().global();

      auto *dstOperand = builder.newAddressOperand(stPointers[instId], "r");
      auto *srcOperand = builder.newAddressOperand(ldPointers[instId], "l");
      auto *sizeConst = builder.newConstantOperand(copySize);
      cp(dstOperand, srcOperand, sizeConst).predicate(predicates[instId], "b");
      builder.launch(rewriter, loc, voidType);
    }

    rewriter.create<NVVM::CpAsyncCommitGroupOp>(loc);
    rewriter.eraseOp(op);
    return success();
  }

private:
  Type getSourceElementType(CpAsyncGlobalToSharedOp op) const {
    auto elementType = op.getSource().getType().getElementType();
    return typeConverter->convertType(elementType);
  }
};

class CpAsyncWaitGroupOpConversion
    : public ConvertOpToLLVMPattern<CpAsyncWaitGroupOp> {
public:
  using ConvertOpToLLVMPattern<CpAsyncWaitGroupOp>::ConvertOpToLLVMPattern;

  virtual LogicalResult
  matchAndRewrite(CpAsyncWaitGroupOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    assert(op->getNumOperands() == 0 && op->getNumResults() == 0);
    auto nAttr = op.getNumPendingAttr();
    rewriter.replaceOpWithNewOp<NVVM::CpAsyncWaitGroupOp>(op, nAttr);
    return success();
  }
};

} // namespace

void kapy::populateCpAsyncOpToLLVMConversionPatterns(
    const LLVMTypeConverter &typeConverter, RewritePatternSet &patterns) {
  patterns.add<CpAsyncGlobalToSharedOpConversion>(typeConverter);
  patterns.add<CpAsyncWaitGroupOpConversion>(typeConverter);
}
