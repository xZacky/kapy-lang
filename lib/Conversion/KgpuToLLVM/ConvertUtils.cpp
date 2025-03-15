//===- ConvertUtils.cpp -----------------------------------------*- C++ -*-===//
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
#include "kapy/Dialect/Kapy/IR/Kapy.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/AffineExprVisitor.h"

using namespace mlir;
using namespace mlir::kapy;
using namespace mlir::LLVM;

Value arith::createConstant(OpBuilder &rewriter, Location loc,
                            TypedAttr value) {
  return rewriter.create<arith::ConstantOp>(loc, value.getType(), value);
}

Value arith::createConstantF16(OpBuilder &rewriter, Location loc, float value) {
  auto f16Type = FloatType::getF16(rewriter.getContext());
  return rewriter.create<arith::ConstantOp>(loc, f16Type,
                                            rewriter.getF16FloatAttr(value));
}

Value arith::createConstantF32(OpBuilder &rewriter, Location loc, float value) {
  auto f32Type = FloatType::getF32(rewriter.getContext());
  return rewriter.create<arith::ConstantOp>(loc, f32Type,
                                            rewriter.getF32FloatAttr(value));
}

Value arith::createConstantF64(OpBuilder &rewriter, Location loc,
                               double value) {
  auto f64Type = FloatType::getF64(rewriter.getContext());
  return rewriter.create<arith::ConstantOp>(loc, f64Type,
                                            rewriter.getF64FloatAttr(value));
}

Value arith::createConstantI1(OpBuilder &rewriter, Location loc, bool value) {
  auto i1Type = rewriter.getIntegerType(1);
  return rewriter.create<arith::ConstantOp>(loc, i1Type,
                                            IntegerAttr::get(i1Type, value));
}

Value arith::createConstantI32(OpBuilder &rewriter, Location loc,
                               int32_t value) {
  auto i32Type = rewriter.getIntegerType(32);
  return rewriter.create<arith::ConstantOp>(loc, i32Type,
                                            IntegerAttr::get(i32Type, value));
}

Value arith::createConstantI64(OpBuilder &rewriter, Location loc,
                               int64_t value) {
  auto i64Type = rewriter.getIntegerType(64);
  return rewriter.create<arith::ConstantOp>(loc, i64Type,
                                            IntegerAttr::get(i64Type, value));
}

GEPOp LLVM::createGEPOp(OpBuilder &rewriter, Location loc, Type pointerType,
                        Type elementType, Value pointer, Value offset) {
  return rewriter.create<GEPOp>(loc, pointerType, elementType, pointer, offset);
}

SmallVector<Value> kapy::unpackLLVMStruct(OpBuilder &rewriter, Location loc,
                                          Value llvmStruct) {
  auto bodyTypes = cast<LLVMStructType>(llvmStruct.getType()).getBody();
  SmallVector<Value> values(bodyTypes.size());
  for (unsigned i = 0; i < bodyTypes.size(); ++i)
    values[i] = llvm_extractvalue(bodyTypes[i], llvmStruct, i);
  return values;
}

Value kapy::packToLLVMStruct(OpBuilder &rewriter, Location loc,
                             LLVMStructType structType, ValueRange values) {
  auto bodyTypes = structType.getBody();
  assert(bodyTypes.size() == values.size());
  Value llvmStruct = rewriter.create<UndefOp>(loc, structType);
  for (auto it : llvm::enumerate(values)) {
    assert(it.value() && it.value().getType() == bodyTypes[it.index()]);
    llvmStruct =
        llvm_insertvalue(structType, llvmStruct, it.value(), it.index());
  }
  return llvmStruct;
}

namespace {
/// Modified from mlir/lib/Dialect/Affine/Utils/Utils.cpp.
/// Create ConstantOp with i32 type but not ConstantIndexOp so should ensure all
/// the values can be represented by i32. AffineSymbolExpr is not supported.
class AffineApplyExpander
    : public AffineExprVisitor<AffineApplyExpander, Value> {
public:
  AffineApplyExpander(OpBuilder &rewriter, Location loc, ValueRange inputs)
      : rewriter(rewriter), loc(loc), inputs(inputs) {}

  template <typename OpT> Value buildBinaryExpr(AffineBinaryOpExpr expr) {
    auto lhs = visit(expr.getLHS());
    auto rhs = visit(expr.getRHS());
    if (!lhs || !rhs)
      return nullptr;
    auto op = rewriter.create<OpT>(loc, lhs, rhs);
    return op.getResult();
  }

  Value visitAddExpr(AffineBinaryOpExpr expr) {
    return buildBinaryExpr<arith::AddIOp>(expr);
  }

  Value visitMulExpr(AffineBinaryOpExpr expr) {
    return buildBinaryExpr<arith::MulIOp>(expr);
  }

  /// Euclidean modulo operation: negative RHS is not allowed.
  /// Remainder of the euclidean integer division is always non-negative.
  ///
  /// Implemented as
  ///
  ///     a mod b =
  ///         let remainder = srem a, b;
  ///             negative = a < 0 in
  ///         select negative, remainder + b, remainder.
  Value visitModExpr(AffineBinaryOpExpr expr) {
    if (auto rhsConst = dyn_cast<AffineConstantExpr>(expr.getRHS())) {
      if (rhsConst.getValue() <= 0) {
        emitError(loc, "modulo by non-positive value is not supported");
        return Value();
      }
    }
    auto lhs = visit(expr.getLHS());
    auto rhs = visit(expr.getRHS());
    assert(lhs && rhs && "unexpected affine expr lowering failure");

    Value remainder = arith_remsi(lhs, rhs);
    Value zero = arith_constant_i32(0);
    Value isRemainderNegative =
        arith_cmpi(arith::CmpIPredicate::slt, remainder, zero);
    Value correctedRemainder = arith_addi(remainder, rhs);
    return arith_select(isRemainderNegative, correctedRemainder, remainder);
  }

  /// Floor division operation (rounds towards negative infinity).
  ///
  /// For positive divisors, it can be implemented without branching and with a
  /// single division operation as
  ///
  ///        a floordiv b =
  ///            let negative = a < 0 in
  ///            let absolute = negative ? -a - 1 : a in
  ///            let quotient = absolute / b in
  ///                negative ? -quotient - 1 : quotient
  ///
  /// Note: this lowering does not use arith.floordivsi because the lowering of
  /// that to arith.divsi (see populateCeilFloorDivExpandOpsPatterns) generates
  /// not one but two arith.divsi. That could be changed to one divsi, but one
  /// way or another, going through arith.floordivsi will result in more complex
  /// IR because arith.floordivsi is more general than affine floordiv in that
  /// it supports negative RHS.
  Value visitFloorDivExpr(AffineBinaryOpExpr expr) {
    if (auto rhsConst = dyn_cast<AffineConstantExpr>(expr.getRHS())) {
      if (rhsConst.getValue() <= 0) {
        emitError(loc, "division by non-positive value is not supported");
        return Value();
      }
    }
    auto lhs = visit(expr.getLHS());
    auto rhs = visit(expr.getRHS());
    assert(lhs && rhs && "unexpected affine expr lowering failure");

    Value zero = arith_constant_i32(0);
    Value none = arith_constant_i32(-1);
    Value negative = arith_cmpi(arith::CmpIPredicate::slt, lhs, zero);
    Value negatedDecremented = arith_subi(none, lhs);
    Value dividend = arith_select(negative, negatedDecremented, lhs);
    Value quotient = arith_divsi(dividend, rhs);
    Value correctedQuotient = arith_subi(none, quotient);
    return arith_select(negative, correctedQuotient, quotient);
  }

  /// Ceiling division operation (rounds towards positive infinity).
  ///
  /// For positive divisors, it can be implemented without branching and with a
  /// single division operation as
  ///
  ///     a ceildiv b =
  ///         let negative = a <= 0 in
  ///         let absolute = negative ? -a : a - 1 in
  ///         let quotient = absolute / b in
  ///             negative ? -quotient : quotient + 1
  ///
  /// Note: not using arith.ceildivsi for the same reason as explained in the
  /// visitFloorDivExpr comment.
  Value visitCeilDivExpr(AffineBinaryOpExpr expr) {
    if (auto rhsConst = dyn_cast<AffineConstantExpr>(expr.getRHS())) {
      if (rhsConst.getValue() <= 0) {
        emitError(loc, "division by non-positive value is not supported");
        return Value();
      }
    }
    auto lhs = visit(expr.getLHS());
    auto rhs = visit(expr.getRHS());
    assert(lhs && rhs && "unexpected affine expr lowering failure");

    Value zero = arith_constant_i32(0);
    Value one = arith_constant_i32(1);
    Value nonPositive = arith_cmpi(arith::CmpIPredicate::sle, lhs, zero);
    Value negated = arith_subi(zero, lhs);
    Value decremented = arith_subi(lhs, one);
    Value dividend = arith_select(nonPositive, negated, decremented);
    Value quotient = arith_divsi(dividend, rhs);
    Value negatedQuotient = arith_subi(zero, quotient);
    Value incrementedQuotient = arith_addi(quotient, one);
    return arith_select(nonPositive, negatedQuotient, incrementedQuotient);
  }

  Value visitConstantExpr(AffineConstantExpr expr) {
    return arith_constant_i32(expr.getValue());
  }

  Value visitDimExpr(AffineDimExpr expr) { return inputs[expr.getPosition()]; }

private:
  OpBuilder &rewriter;
  Location loc;
  ValueRange inputs;
};

} // namespace

Value kapy::expandAffineExpr(OpBuilder &rewriter, Location loc, AffineExpr expr,
                             ValueRange inputs) {
  return AffineApplyExpander(rewriter, loc, inputs).visit(expr);
}

int64_t kapy::generateInitConstant(Type elementType,
                                   PaddingOption paddingOption) {
  if (elementType.isInteger()) {
    assert(paddingOption == PaddingOption::ZERO);
    return 0;
  }
  if (elementType.isF32()) {
    switch (paddingOption) {
    case PaddingOption::ZERO:
      return 0;
    case PaddingOption::QNAN:
      return 0x7FC00000;
    case PaddingOption::PINF:
      return 0x7F800000;
    case PaddingOption::NINF:
      return 0xFF800000;
    }
  }
  if (elementType.isF16()) {
    switch (paddingOption) {
    case PaddingOption::ZERO:
      return 0;
    case PaddingOption::QNAN:
      return 0x7E007E00;
    case PaddingOption::PINF:
      return 0x7C007C00;
    case PaddingOption::NINF:
      return 0xFC00FC00;
    }
  }
  if (elementType.isBF16()) {
    switch (paddingOption) {
    case PaddingOption::ZERO:
      return 0;
    case PaddingOption::QNAN:
      return 0x7FC07FC0;
    case PaddingOption::PINF:
      return 0x7F807F80;
    case PaddingOption::NINF:
      return 0xFF80FF80;
    }
  }
  if (elementType.isFloat8E4M3()) {
    switch (paddingOption) {
    case PaddingOption::ZERO:
      return 0;
    case PaddingOption::QNAN:
      return 0x7C7C7C7C;
    case PaddingOption::PINF:
      return 0x78787878;
    case PaddingOption::NINF:
      return 0xF8F8F8F8;
    }
  }
  if (elementType.isFloat8E5M2()) {
    switch (paddingOption) {
    case PaddingOption::ZERO:
      return 0;
    case PaddingOption::QNAN:
      return 0x7E7E7E7E;
    case PaddingOption::PINF:
      return 0x7C7C7C7C;
    case PaddingOption::NINF:
      return 0xFCFCFCFC;
    }
  }
  llvm_unreachable("unsupported element type");
}
