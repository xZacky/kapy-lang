//===- CommonUtils.h --------------------------------------------*- C++ -*-===//
//
// This file defines several utilities that commonly used.
//
//===----------------------------------------------------------------------===//

#ifndef KAPY_SUPPORT_COMMON_UTILS_H
#define KAPY_SUPPORT_COMMON_UTILS_H

#include "mlir/IR/AffineExpr.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallVector.h"
#include <numeric>

namespace mlir {
namespace kapy {

template <typename T> T ceilDiv(T a, T b) {
  static_assert(std::is_integral_v<T>);
  return (a + b - 1) / b;
}

template <typename T> T product(ArrayRef<T> array) {
  static_assert(std::is_integral_v<T>);
  if (array.empty())
    return 1;
  return std::accumulate(array.begin(), array.end(), 1, std::multiplies());
}
template <typename ArrayT> auto product(const ArrayT &array) {
  return product(ArrayRef(array));
}

template <typename T> SmallVector<T, 2> transpose(ArrayRef<T> array) {
  assert(array.size() == 2);
  SmallVector<T, 2> result;
  result.push_back(array[1]);
  result.push_back(array[0]);
  return result;
}
template <typename ArrayT> auto transpose(const ArrayT &array) {
  return transpose(ArrayRef(array));
}

template <unsigned N = 2, typename T>
SmallVector<T, N> computeStrides(ArrayRef<T> shape) {
  static_assert(std::is_integral_v<T>);
  auto rank = shape.size();
  SmallVector<T, N> strides(rank, 1);
  for (unsigned i = 0; i < rank; ++i)
    for (unsigned j = i + 1; j < rank; ++j)
      strides[i] *= shape[j];
  return strides;
}
template <unsigned N = 2, typename ArrayT>
auto computeStrides(const ArrayT &shape) {
  return computeStrides<N>(ArrayRef(shape));
}

template <unsigned N = 2, typename T, typename U>
T linearize(ArrayRef<T> coords, ArrayRef<U> shape) {
  static_assert(std::is_integral_v<T> && std::is_integral_v<U>);
  auto strides = computeStrides<N>(shape);
  auto rank = shape.size();
  T linear = 0;
  for (unsigned i = 0; i < rank; ++i)
    linear += coords[i] * strides[i];
  return linear;
}
template <unsigned N = 2, typename ArrayT, typename ArrayU>
auto linearize(const ArrayT &coords, const ArrayU &shape) {
  return linearize<N>(ArrayRef(coords), ArrayRef(shape));
}

template <unsigned N = 2, typename T>
SmallVector<T, N> delinearize(T linear, ArrayRef<T> shape) {
  static_assert(std::is_integral_v<T>);
  auto strides = computeStrides<N>(shape);
  auto rank = shape.size();
  SmallVector<T, N> coords(rank, linear);
  for (unsigned i = 0; i < rank; ++i) {
    if (i != 0)
      coords[i] %= strides[i - 1];
    if (i != rank - 1)
      coords[i] /= strides[i];
  }
  return coords;
}
template <unsigned N = 2, typename T, typename ArrayT>
auto delinearize(T linear, const ArrayT &shape) {
  return delinearize<N>(linear, ArrayRef(shape));
}

template <unsigned N = 2, typename T>
AffineExpr linearize(ArrayRef<AffineExpr> coords, ArrayRef<T> shape) {
  static_assert(std::is_integral_v<T>);
  auto strides = computeStrides<N>(shape);
  auto rank = shape.size();
  auto linear = getAffineConstantExpr(0, coords.begin()->getContext());
  for (unsigned i = 0; i < rank; ++i)
    linear = linear + coords[i] * strides[i];
  return linear;
}
template <unsigned N = 2, typename ArrayT>
auto linearize(ArrayRef<AffineExpr> coords, const ArrayT &shape) {
  return linearize<N>(coords, ArrayRef(shape));
}

template <unsigned N = 2, typename T>
SmallVector<AffineExpr, N> delinearize(AffineExpr linear, ArrayRef<T> shape) {
  static_assert(std::is_integral_v<T>);
  auto strides = computeStrides<N>(shape);
  auto rank = shape.size();
  SmallVector<AffineExpr, N> coords(rank, linear);
  for (unsigned i = 0; i < rank; ++i) {
    if (i != 0)
      coords[i] = coords[i] % strides[i - 1];
    if (i != rank - 1)
      coords[i] = coords[i].floorDiv(strides[i]);
  }
  return coords;
}
template <unsigned N = 2, typename ArrayT>
auto delinearize(AffineExpr linear, const ArrayT &shape) {
  return delinearize<N>(linear, ArrayRef(shape));
}

} // namespace kapy
} // namespace mlir

#endif // KAPY_SUPPORT_COMMON_UTILS_H
