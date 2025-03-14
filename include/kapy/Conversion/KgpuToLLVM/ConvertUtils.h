//===- ConvertUtils.h -------------------------------------------*- C++ -*-===//
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

#ifndef KAPY_CONVERSION_KGPUTOLLVM_CONVERTUTILS_H
#define KAPY_CONVERSION_KGPUTOLLVM_CONVERTUTILS_H

#include "mlir/Conversion/LLVMCommon/TypeConverter.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/IR/AffineExpr.h"

// Shortcuts for some commonly used LLVM operations.
#define llvm_inttoptr(...) rewriter.create<LLVM::IntToPtrOp>(loc, __VA_ARGS__)
#define llvm_ptrtoint(...) rewriter.create<LLVM::PtrToIntOp>(loc, __VA_ARGS__)
#define llvm_bitcast(...) rewriter.create<LLVM::BitcastOp>(loc, __VA_ARGS__)
#define llvm_getelementptr(...) LLVM::createGEPOp(rewriter, loc, __VA_ARGS__)
#define llvm_insertvalue(...)                                                  \
  rewriter.create<LLVM::InsertValueOp>(loc, __VA_ARGS__)
#define llvm_extractvalue(...)                                                 \
  rewriter.create<LLVM::ExtractValueOp>(loc, __VA_ARGS__)
#define llvm_insertelement(...)                                                \
  rewriter.create<LLVM::InsertElementOp>(loc, __VA_ARGS__)
#define llvm_extractelement(...)                                               \
  rewriter.create<LLVM::ExtractElementOp>(loc, __VA_ARGS__)
#define llvm_addressof(...) rewriter.create<LLVM::AddressOfOp>(loc, __VA_ARGS__)
#define llvm_undef(...) rewriter.create<LLVM::UndefOp>(loc, __VA_ARGS__)
#define llvm_null(...) rewriter.create<LLVM::NullOp>(loc, __VA_ARGS__)

// Shortcuts for some commonly used arith operations.
#define arith_constant(...) arith::createConstant(rewriter, loc, __VA_ARGS__)
#define arith_constant_f16(...)                                                \
  arith::createConstantF16(rewriter, loc, __VA_ARGS__)
#define arith_constant_f32(...)                                                \
  arith::createConstantF32(rewriter, loc, __VA_ARGS__)
#define arith_constant_f64(...)                                                \
  arith::createConstantF64(rewriter, loc, __VA_ARGS__)
#define arith_constant_i1(...)                                                 \
  LLVM::createConstantI1(loc, rewriter, __VA_ARGS__)
#define arith_constant_i32(...)                                                \
  arith::createConstantI32(rewriter, loc, __VA_ARGS__)
#define arith_constant_i64(...)                                                \
  arith::createConstantI64(rewriter, loc, __VA_ARGS__)
#define arith_addi(...) rewriter.create<arith::AddIOp>(loc, __VA_ARGS__)
#define arith_addf(...) rewriter.create<arith::AddFOp>(loc, __VA_ARGS__)
#define arith_subi(...) rewriter.create<arith::SubIOp>(loc, __VA_ARGS__)
#define arith_subf(...) rewriter.create<arith::SubFOp>(loc, __VA_ARGS__)
#define arith_muli(...) rewriter.create<arith::MulIOp>(loc, __VA_ARGS__)
#define arith_mulf(...) rewriter.create<arith::MulFOp>(loc, __VA_ARGS__)
#define arith_divui(...) rewriter.create<arith::DivUIOp>(loc, __VA_ARGS__)
#define arith_divsi(...) rewriter.create<arith::DivSIOp>(loc, __VA_ARGS__)
#define arith_divf(...) rewriter.create<arith::DivFOp>(loc, __VA_ARGS__)
#define arith_floordivsi(...)                                                  \
  rewriter.create<arith::FloorDivSIOp>(loc, __VA_ARGS__)
#define arith_ceildivui(...)                                                   \
  rewriter.create<arith::CeilDivUIOp>(loc, __VA_ARGS__)
#define arith_ceildivsi(...)                                                   \
  rewriter.create<arith::CeilDivSIOp>(loc, __VA_ARGS__)
#define arith_remui(...) rewriter.create<arith::RemUIOp>(loc, __VA_ARGS__)
#define arith_remsi(...) rewriter.create<arith::RemSIOp>(loc, __VA_ARGS__)
#define arith_remf(...) rewriter.create<arith::RemFOp>(loc, __VA_ARGS__)
#define arith_andi(...) rewriter.create<arith::AndIOp>(loc, __VA_ARGS__)
#define arith_ori(...) rewriter.create<arith::OrIOp>(loc, __VA_ARGS__)
#define arith_xori(...) rewriter.create<arith::XOrIOp>(loc, __VA_ARGS__)
#define arith_shli(...) rewriter.create<arith::ShLIOp>(loc, __VA_ARGS__)
#define arith_shrui(...) rewriter.create<arith::ShRUIOp>(loc, __VA_ARGS__)
#define arith_shrsi(...) rewriter.create<arith::ShRSIOp>(loc, __VA_ARGS__)
#define arith_maxui(...) rewriter.create<arith::MaxUIOp>(loc, __VA_ARGS__)
#define arith_minsi(...) rewriter.create<arith::MinSIOp>(loc, __VA_ARGS__)
#define arith_maximumf(...) rewriter.create<arith::MaximumFOp>(loc, __VA_ARGS__)
#define arith_maxnumf(...) rewriter.create<arith::MaxNumFOp>(loc, __VA_ARGS__)
#define arith_minimumf(...) rewriter.create<arith::MinimumFOp>(loc, __VA_ARGS__)
#define arith_minnumf(...) rewriter.create<arith::MinNumFOp>(loc, __VA_ARGS__)
#define arith_cmpf(...) rewriter.create<arith::CmpFOp>(loc, __VA_ARGS__)
#define arith_cmpi(...) rewriter.create<arith::CmpIOp>(loc, __VA_ARGS__)
#define arith_trunci(...) rewriter.create<arith::TruncIOp>(loc, __VA_ARGS__)
#define arith_truncf(...) rewriter.create<arith::TruncFOp>(loc, __VA_ARGS__)
#define arith_extui(...) rewriter.create<arith::ExtUIOp>(loc, __VA_ARGS__)
#define arith_extsi(...) rewriter.create<arith::ExtSIOp>(loc, __VA_ARGS__)
#define arith_extf(...) rewriter.create<arith::ExtFOp>(loc, __VA_ARGS__)
#define arith_uitofp(...) rewriter.create<arith::UIToFPOp>(loc, __VA_ARGS__)
#define arith_fptoui(...) rewriter.create<arith::FPToUIOp>(loc, __VA_ARGS__)
#define arith_sitofp(...) rewriter.create<arith::SIToFPOp>(loc, __VA_ARGS__)
#define arith_fptosi(...) rewriter.create<arith::FPToSIOp>(loc, __VA_ARGS__)
#define arith_select(...) rewriter.create<arith::SelectOp>(loc, __VA_ARGS__)

namespace mlir {
namespace arith {

Value createConstant(OpBuilder &rewriter, Location loc, TypedAttr value);

Value createConstantF16(OpBuilder &rewriter, Location loc, float value);
Value createConstantF32(OpBuilder &rewriter, Location loc, float value);
Value createConstantF64(OpBuilder &rewriter, Location loc, double value);

Value createConstantI1(OpBuilder &rewriter, Location loc, bool value);
Value createConstantI32(OpBuilder &rewriter, Location loc, int32_t value);
Value createConstantI64(OpBuilder &rewriter, Location loc, int64_t value);

} // namespace arith

namespace LLVM {

GEPOp createGEPOp(OpBuilder &rewriter, Location loc, Type pointerType,
                  Type elementType, Value pointer, Value offset);

} // namespace LLVM

namespace kapy {

class MultipleValuesRange
    : public llvm::iterator_range<SmallVector<SmallVector<Value>>::iterator> {
  using ContainerT = SmallVector<SmallVector<Value>>;

public:
  using llvm::iterator_range<ContainerT::iterator>::iterator_range;

  ContainerT::reference operator[](ContainerT::size_type index) {
    return begin()[index];
  }
  ContainerT::const_reference operator[](ContainerT::size_type index) const {
    return begin()[index];
  }
  ContainerT::size_type size() const { return end() - begin(); }
};

SmallVector<Value> unpackLLVMStruct(OpBuilder &rewriter, Location loc,
                                    Value llvmStruct);

Value packToLLVMStruct(OpBuilder &rewriter, Location loc,
                       LLVM::LLVMStructType structType, ValueRange values);

Value expandAffineExpr(OpBuilder &rewriter, Location loc, AffineExpr expr,
                       ValueRange inputs);

enum class PaddingOption : uint32_t;
int64_t generateInitConstant(Type elementType, PaddingOption paddingOption);

} // namespace kapy
} // namespace mlir

#endif // KAPY_CONVERSION_KGPUTOLLVM_CONVERTUTILS_H
