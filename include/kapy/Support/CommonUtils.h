//===- CommonUtils.h --------------------------------------------*- C++ -*-===//
//
// This file defines several utilities that commonly used.
//
//===----------------------------------------------------------------------===//

#ifndef KAPY_SUPPORT_COMMON_UTILS_H
#define KAPY_SUPPORT_COMMON_UTILS_H

#include "mlir/IR/AffineExpr.h"
#include "mlir/Support/LLVM.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallVector.h"
#include <numeric>

namespace mlir {
namespace kapy {

template <typename T> T ceilDiv(T a, T b) {
  static_assert(std::is_integral_v<T>);
  return (a + b - 1) / b;
}

template <typename T> T summation(ArrayRef<T> array) {
  static_assert(std::is_integral_v<T>);
  return std::accumulate(array.begin(), array.end(), 0, std::plus());
}
template <typename ArrayT> auto summation(const ArrayT &array) {
  return summation(ArrayRef(array));
}

template <typename T> T product(ArrayRef<T> array) {
  static_assert(std::is_integral_v<T>);
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

template <typename T> SmallVector<T> getStrides(ArrayRef<T> shape) {
  static_assert(std::is_integral_v<T>);
  SmallVector<T> strides(shape.size(), 1);
  for (unsigned i = 0; i < shape.size(); ++i)
    for (unsigned j = i + 1; j < shape.size(); ++j)
      strides[i] *= shape[j];
  return strides;
}
template <typename ArrayT> auto getStrides(const ArrayT &shape) {
  return getStrides(ArrayRef(shape));
}

template <typename T, typename U>
T linearize(ArrayRef<T> indices, ArrayRef<U> shape) {
  static_assert(std::is_integral_v<T> && std::is_integral_v<U>);
  auto strides = getStrides(shape);
  T index = 0;
  for (unsigned i = 0; i < shape.size(); ++i)
    index += indices[i] * strides[i];
  return index;
}
template <typename ArrayT, typename ArrayU>
auto linearize(const ArrayT &indices, const ArrayU &shape) {
  return linearize(ArrayRef(indices), ArrayRef(shape));
}

template <typename T> SmallVector<T> delinearize(T index, ArrayRef<T> shape) {
  static_assert(std::is_integral_v<T>);
  auto strides = getStrides(shape);
  SmallVector<T> indices(shape.size(), index);
  for (unsigned i = 0; i < shape.size(); ++i) {
    if (i != 0)
      indices[i] %= strides[i - 1];
    if (i != shape.size() - 1)
      indices[i] /= strides[i];
  }
  return indices;
}
template <typename T, typename ArrayT>
auto delinearize(T index, const ArrayT &shape) {
  return delinearize(index, ArrayRef(shape));
}

template <typename T>
AffineExpr linearize(ArrayRef<AffineExpr> indices, ArrayRef<T> shape) {
  static_assert(std::is_integral_v<T>);
  auto strides = getStrides(shape);
  auto index = getAffineConstantExpr(0, indices.begin()->getContext());
  for (unsigned i = 0; i < shape.size(); ++i)
    index = index + indices[i] * strides[i];
  return index;
}
template <typename ArrayT>
auto linearize(ArrayRef<AffineExpr> indices, const ArrayT &shape) {
  return linearize(indices, ArrayRef(shape));
}

template <typename T>
SmallVector<AffineExpr> delinearize(AffineExpr index, ArrayRef<T> shape) {
  static_assert(std::is_integral_v<T>);
  auto strides = getStrides(shape);
  SmallVector<AffineExpr> indices(shape.size(), index);
  for (unsigned i = 0; i < shape.size(); ++i) {
    if (i != 0)
      indices[i] = indices[i] % strides[i - 1];
    if (i != shape.size() - 1)
      indices[i] = indices[i].floorDiv(strides[i]);
  }
  return indices;
}
template <typename ArrayT>
auto delinearize(AffineExpr index, const ArrayT &shape) {
  return delinearize(index, ArrayRef(shape));
}

template <typename T> T getSizeInBytes(T numElements, T bitWidth) {
  static_assert(std::is_integral_v<T>);
  return numElements * ceilDiv<T>(bitWidth, 8);
}

} // namespace kapy
} // namespace mlir

#endif // KAPY_SUPPORT_COMMON_UTILS_H
