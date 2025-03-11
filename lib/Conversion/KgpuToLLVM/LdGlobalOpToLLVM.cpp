//===- LdGlobalOpToLLVM.cpp -------------------------------------*- C++ -*-===//
//
// This file implements class to make LdGlobalOp to LLVM compatible.
//
//===----------------------------------------------------------------------===//

#include "kapy/Conversion/KgpuToLLVM/ConvertUtils.h"
#include "kapy/Conversion/KgpuToLLVM/PTXBuilder.h"
#include "kapy/Conversion/KgpuToLLVM/Patterns.h"
#include "kapy/Dialect/Kapy/IR/Kapy.h"
#include "kapy/Dialect/Kgpu/IR/Kgpu.h"
#include "mlir/Conversion/LLVMCommon/Pattern.h"
#include "mlir/Dialect/Affine/Utils.h"
#include "llvm/ADT/MapVector.h"

using namespace mlir;
using namespace mlir::kapy;
using namespace mlir::LLVM;
using namespace mlir::NVVM;
using namespace mlir::affine;

namespace {

class LdGlobalOpConversion : public ConvertOpToLLVMPattern<LdGlobalOp> {
public:
  using ConvertOpToLLVMPattern::ConvertOpToLLVMPattern;

  virtual LogicalResult
  matchAndRewrite(LdGlobalOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    assert(op->getNumOperands() == 1 && op->getNumResults() == 1);
    auto loc = op.getLoc();

    auto sourceType = op.getSource().getType();
    auto globalLayout = getLayout<Strided2dLayoutAttr>(sourceType);

    auto tensorLayout = getLayout<FragmentsLayoutAttr>(op.getType());
    auto laneLoops = tensorLayout.getLaneLoops();
    auto loopSpace = tensorLayout.getLoopSpace(sourceType.getShape());
    auto map = tensorLayout.getAffineMap(sourceType.getShape(), 1);

    auto bitWidth = getIntOrFloatBitWidth(sourceType);
    auto simdSize = laneLoops[globalLayout.isColMajor() ? 0 : 1];
    auto numPacks = simdSize * bitWidth / 32;

    llvm::MapVector<int64_t, SmallVector<int64_t, 8>> instIdToLoopIvs;
    for (int64_t loopIv0 = 0; loopIv0 < loopSpace[0]; ++loopIv0) {
      for (int64_t loopIv1 = 0; loopIv1 < loopSpace[1]; ++loopIv1) {
        int64_t instId;
        if (globalLayout.isRowMajor())
          instId = (loopIv0 * loopSpace[1] + loopIv1) / simdSize;
        else
          instId = (loopIv0 + loopIv1 * loopSpace[0]) / simdSize;
        int64_t loopIv;
        if (tensorLayout.isRowMajor())
          loopIv = loopIv0 * loopSpace[1] + loopIv1;
        else
          loopIv = loopIv0 + loopIv1 * loopSpace[0];
        instIdToLoopIvs[instId].push_back(loopIv);
      }
    }

    auto pointerType = LLVMPointerType::get(getContext(), 1);
    auto elementType = getSourceElementType(op);
    auto i32Type = rewriter.getIntegerType(32);
    auto lessThan = arith::CmpIPredicate::ult;

    Value sourceStruct = adaptor.getSource();
    Value pointer = llvm_extractvalue(pointerType, sourceStruct, 0);
    Value one = arith_constant_i32(1);
    Value stride0 = globalLayout.isRowMajor()
                        ? llvm_extractvalue(i32Type, sourceStruct, 1)
                        : one;
    Value stride1 = globalLayout.isColMajor()
                        ? llvm_extractvalue(i32Type, sourceStruct, 2)
                        : one;
    Value start0 = llvm_extractvalue(i32Type, sourceStruct, 3);
    Value start1 = llvm_extractvalue(i32Type, sourceStruct, 4);
    Value end0 = llvm_extractvalue(i32Type, sourceStruct, 5);
    Value end1 = llvm_extractvalue(i32Type, sourceStruct, 6);
    Value laneId = createLaneIdOp(rewriter, loc);

    SmallVector<Value> pointers;
    SmallVector<Value> masks;
    for (auto instId : llvm::make_first_range(instIdToLoopIvs)) {
      Value loopIv = arith_constant_i32(instIdToLoopIvs[instId][0]);
      Value index0 = expandAffineExpr(rewriter, loc, map.getResult(0),
                                      {laneId, loopIv}, {});
      Value index1 = expandAffineExpr(rewriter, loc, map.getResult(1),
                                      {laneId, loopIv}, {});
      index0 = arith_addi(start0, index0);
      index1 = arith_addi(start1, index1);

      pointers.push_back(
          llvm_getelementptr(pointerType, elementType, pointer,
                             arith_addi(arith_muli(index0, stride0),
                                        arith_muli(index1, stride1))));

      masks.push_back(arith_andi(arith_cmpi(lessThan, index0, end0),
                                 arith_cmpi(lessThan, index1, end1)));
    }

    SmallVector<Value> resultValues(loopSpace[0] * loopSpace[1]);
    for (auto instId : llvm::make_first_range(instIdToLoopIvs)) {
      PTXBuilder builder;
      auto &ld = builder.create("ld")->o("volatile", op.isVolatile());
      ld.global();
      if (op.getCacheModifier() != CacheModifier::NONE)
        ld.o(stringifyCacheModifier(op.getCacheModifier()).str());
      ld.o("L1::" + stringifyEvictPriority(op.getEvictPriority()).str());
      ld.v(numPacks, numPacks > 1).b(32);

      auto *dstOperand = builder.newListOperand();
      for (unsigned i = 0; i < numPacks; ++i)
        dstOperand->listPushBack(builder.newOperand("=r", true));
      auto *srcOperand = builder.newAddressOperand(pointers[instId], "l");
      ld(dstOperand, srcOperand).predicate(masks[instId]);

      SmallVector<Type> packTypes(numPacks, i32Type);
      Type packsType;
      if (numPacks > 1)
        packsType = LLVMStructType::getLiteral(getContext(), packTypes);
      else
        packsType = packTypes[0];
      auto packs = builder.launch(rewriter, loc, packsType);
      for (unsigned i = 0; i < numPacks; ++i) {
        Value pack;
        if (isa<LLVMStructType>(packsType))
          pack = llvm_extractvalue(i32Type, packs, i);
        else
          pack = packs;
        if (bitWidth < 32) {
          auto packType = VectorType::get(32 / bitWidth, elementType);
          pack = llvm_bitcast(packType, pack);
        }
        for (unsigned j = 0; j < 32 / bitWidth; ++j) {
          auto loopIv = instIdToLoopIvs[instId][i * (32 / bitWidth) + j];
          resultValues[loopIv] =
              llvm_extractelement(elementType, pack, arith_constant_i32(j));
        }
      }
    }

    auto resultType = getResultStructType(op);
    auto resultStruct =
        packToLLVMStruct(rewriter, loc, resultType, resultValues);
    rewriter.replaceOp(op, resultStruct);
    return success();
  }

private:
  Type getSourceElementType(LdGlobalOp op) const {
    auto elementType = op.getSource().getType().getElementType();
    return typeConverter->convertType(elementType);
  }

  LLVMStructType getResultStructType(LdGlobalOp op) const {
    auto resultType = typeConverter->convertType(op.getType());
    return cast<LLVMStructType>(resultType);
  }
};

} // namespace

void kapy::populateLdGlobalOpToLLVMConversionPattern(
    const LLVMTypeConverter &typeConverter, RewritePatternSet &patterns) {
  patterns.add<LdGlobalOpConversion>(patterns);
}
