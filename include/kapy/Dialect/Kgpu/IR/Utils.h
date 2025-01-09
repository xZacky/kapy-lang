//===- Utils.h --------------------------------------------------*- C++ -*-===//
//
// This file defines several utilities that commonly used.
//
//===----------------------------------------------------------------------===//

#ifndef KAPY_DIALECT_KGPU_IR_UTILS_H
#define KAPY_DIALECT_KGPU_IR_UTILS_H

#include "mlir/IR/AffineExpr.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallVector.h"

namespace mlir {

using llvm::ArrayRef;
using llvm::SmallVector;

namespace kapy {

template <typename I> SmallVector<I, 4> computeStrides(ArrayRef<I> shape) {
  static_assert(std::is_integral_v<I>);
  auto rank = shape.size();
  SmallVector<I, 4> strides(rank, 1);
  for (unsigned i = 0; i < rank; ++i)
    for (unsigned j = i + 1; j < rank; ++j)
      strides[i] *= shape[j];
  return strides;
}
template <typename ShapeT> auto computeStrides(const ShapeT &shape) {
  return computeStrides(ArrayRef(shape));
}

template <typename I> I linearize(ArrayRef<I> indices, ArrayRef<I> shape) {
  static_assert(std::is_integral_v<I>);
  auto strides = computeStrides(shape);
  auto rank = shape.size();
  I index = 0;
  for (unsigned i = 0; i < rank; ++i)
    index += indices[i] * strides[i];
  return index;
}
template <typename IndicesT, typename ShapeT>
auto linearize(const IndicesT &indices, const ShapeT &shape) {
  return linearize(ArrayRef(indices), ArrayRef(shape));
}

template <typename I>
SmallVector<I, 4> delinearize(I index, ArrayRef<I> shape) {
  static_assert(std::is_integral_v<I>);
  auto strides = computeStrides(shape);
  auto rank = shape.size();
  SmallVector<I, 4> indices(rank, index);
  for (unsigned i = 0; i < rank; ++i) {
    if (i != 0)
      indices[i] %= strides[i - 1];
    if (i != rank - 1)
      indices[i] /= strides[i];
  }
  return indices;
}
template <typename I, typename ShapeT>
auto delinearize(I index, const ShapeT &shape) {
  return delinearize(index, ArrayRef(shape));
}

template <typename I>
AffineExpr linearize(ArrayRef<AffineExpr> exprs, ArrayRef<I> shape) {
  static_assert(std::is_integral_v<I>);
  auto strides = computeStrides(shape);
  auto rank = shape.size();
  auto expr = getAffineConstantExpr(0, exprs.begin()->getContext());
  for (unsigned i = 0; i < rank; ++i)
    expr = expr + exprs[i] * strides[i];
  return expr;
}
template <typename ShapeT>
auto linearize(ArrayRef<AffineExpr> exprs, const ShapeT &shape) {
  return linearize(exprs, ArrayRef(shape));
}

template <typename I>
SmallVector<AffineExpr, 4> delinearize(AffineExpr expr, ArrayRef<I> shape) {
  static_assert(std::is_integral_v<I>);
  auto strides = computeStrides(shape);
  auto rank = shape.size();
  SmallVector<AffineExpr, 4> exprs(rank, expr);
  for (unsigned i = 0; i < rank; ++i) {
    if (i != 0)
      exprs[i] = exprs[i] % strides[i - 1];
    if (i != rank - 1)
      exprs[i] = exprs[i].floorDiv(strides[i]);
  }
  return exprs;
}
template <typename ShapeT>
auto delinearize(AffineExpr expr, const ShapeT &shape) {
  return delinearize(expr, ArrayRef(shape));
}

} // namespace kapy
} // namespace mlir

#endif // KAPY_DIALECT_KGPU_IR_UTILS_H
