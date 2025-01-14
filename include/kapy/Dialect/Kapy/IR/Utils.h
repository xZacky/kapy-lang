//===- Utils.h --------------------------------------------------*- C++ -*-===//
//
// This file defines several utilities that commonly used.
//
//===----------------------------------------------------------------------===//

#ifndef KAPY_DIALECT_KAPY_IR_UTILS_H
#define KAPY_DIALECT_KAPY_IR_UTILS_H

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallVector.h"
#include <numeric>

namespace mlir {

using llvm::ArrayRef;
using llvm::SmallVector;

namespace kapy {

template <typename I> I ceilDiv(I a, I b) {
  static_assert(std::is_integral_v<I>);
  return (a + b - 1) / b;
}

/// Return the product of an integer array, return 0 when the array is empty.
template <typename I> I product(ArrayRef<I> array) {
  static_assert(std::is_integral_v<I>);
  if (array.empty())
    return 0;
  return std::accumulate(array.begin(), array.end(), 1, std::multiplies());
}
template <typename ArrayT> auto product(const ArrayT &array) {
  return product(ArrayRef(array));
}

template <typename I> bool isIota(ArrayRef<I> order) {
  static_assert(std::is_integral_v<I>);
  for (unsigned i = 0; i < order.size(); ++i)
    if (order[i] != i)
      return false;
  return true;
}
template <typename OrderT> bool isIota(const OrderT &order) {
  return isIota(ArrayRef(order));
}

template <typename I> SmallVector<I, 4> makeIota(unsigned rank) {
  static_assert(std::is_integral_v<I>);
  SmallVector<I, 4> result(rank);
  for (unsigned i = 0; i < rank; ++i)
    result[i] = i;
  return result;
}

template <typename T, typename I>
SmallVector<T, 4> permute(ArrayRef<T> array, ArrayRef<I> order) {
  static_assert(std::is_integral_v<I>);
  assert(array.size() == order.size());
  SmallVector<T, 4> result;
  for (unsigned i : order)
    result.push_back(array[i]);
  return result;
}
template <typename ArrayT, typename OrderT>
auto permute(const ArrayT &array, const OrderT &order) {
  return permute(ArrayRef(array), ArrayRef(order));
}

template <typename I> SmallVector<I, 4> inverse(ArrayRef<I> order) {
  static_assert(std::is_integral_v<I>);
  auto rank = order.size();
  SmallVector<I, 4> result(rank);
  for (unsigned i = 0; i < rank; ++i)
    result[order[i]] = i;
  return result;
}
template <typename ArrayT> auto inverse(const ArrayT &order) {
  return inverse(ArrayRef(order));
}

} // namespace kapy
} // namespace mlir

#endif // KAPY_DIALECT_KAPY_IR_UTILS_H
