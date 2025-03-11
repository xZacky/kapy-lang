//===- StGlobalOpToLLVM.cpp -------------------------------------*- C++ -*-===//
//
// This file implements class to make StGlobalOp to LLVM compatible.
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

class StGlobalOpConversion : public ConvertOpToLLVMPattern<StGlobalOp> {
public:
  using ConvertOpToLLVMPattern::ConvertOpToLLVMPattern;

  virtual LogicalResult
  matchAndRewrite(StGlobalOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    assert(op->getNumOperands() == 2 && op->getNumResults() == 0);
    auto loc = op.getLoc();

    auto targetType = op.getTarget().getType();
    auto globalLayout = getLayout<Strided2dLayoutAttr>(targetType);

    auto sourceType = op.getSource().getType();
    auto tensorLayout = getLayout<FragmentsLayoutAttr>(sourceType);
    auto laneLoops = tensorLayout.getLaneLoops();
    auto loopSpace = tensorLayout.getLoopSpace(sourceType.getShape());
    auto map = tensorLayout.getAffineMap(sourceType.getShape(), 2);

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
    auto i8Type = rewriter.getIntegerType(8);
    auto i32Type = rewriter.getIntegerType(32);
    auto lessThan = arith::CmpIPredicate::ult;

    Value sourceStruct = adaptor.getSource();
    Value targetStruct = adaptor.getTarget();
    Value pointer = llvm_extractvalue(pointerType, targetStruct, 0);
    Value one = arith_constant_i32(1);
    Value stride0 = globalLayout.isRowMajor()
                        ? llvm_extractvalue(i32Type, targetStruct, 1)
                        : one;
    Value stride1 = globalLayout.isColMajor()
                        ? llvm_extractvalue(i32Type, targetStruct, 2)
                        : one;
    Value start0 = llvm_extractvalue(i32Type, targetStruct, 3);
    Value start1 = llvm_extractvalue(i32Type, targetStruct, 4);
    Value end0 = llvm_extractvalue(i32Type, targetStruct, 5);
    Value end1 = llvm_extractvalue(i32Type, targetStruct, 6);
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

    for (auto instId : llvm::make_first_range(instIdToLoopIvs)) {
      PTXBuilder builder;
      auto &st = builder.create("st")->global();
      if (op.getCacheModifier() != CacheModifier::NONE)
        st.o(stringifyCacheModifier(op.getCacheModifier()).str());
      st.o("L1::" + stringifyEvictPriority(op.getEvictPriority()).str());
      st.v(numPacks, numPacks > 1).b(32);

      auto *dstOperand = builder.newAddressOperand(pointers[instId], "l");
      SmallVector<Value> packs;
      for (unsigned i = 0; i < numPacks; ++i) {
        if (bitWidth < 32) {
          auto packType = VectorType::get(32 / bitWidth, elementType);
          Value pack = llvm_undef(packType);
          for (unsigned j = 0; j < 32 / bitWidth; ++j) {
            auto loopIv = instIdToLoopIvs[instId][i * (32 / bitWidth) + j];
            Value value = llvm_extractvalue(elementType, sourceStruct, loopIv);
            if (value.getType().isInteger(1))
              value = arith_extsi(i8Type, value);
            pack = llvm_insertelement(packType, pack, value,
                                      arith_constant_i32(j));
          }
          packs.push_back(pack);
        } else {
          auto loopIv = instIdToLoopIvs[instId][i];
          Value value = llvm_extractvalue(elementType, sourceStruct, loopIv);
          if (value.getType().isF32())
            value = llvm_bitcast(i32Type, value);
          packs.push_back(value);
        }
      }
      auto *srcOperand = builder.newListOperand();
      for (auto pack : packs)
        srcOperand->listPushBack(builder.newOperand(pack, "r"));
      st(dstOperand, srcOperand).predicate(masks[instId]);

      auto voidType = LLVMVoidType::get(getContext());
      builder.launch(rewriter, loc, voidType);
    }

    rewriter.eraseOp(op);
    return success();
  }

private:
  Type getSourceElementType(StGlobalOp op) const {
    auto elementType = getElementTypeOrSelf(op.getSource().getType());
    return typeConverter->convertType(elementType);
  }
};

} // namespace

void kapy::populateStGlobalOpToLLVMConversionPattern(
    const LLVMTypeConverter &typeConverter, RewritePatternSet &patterns) {
  patterns.add<StGlobalOpConversion>(typeConverter);
}
