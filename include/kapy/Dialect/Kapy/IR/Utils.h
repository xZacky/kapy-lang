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

} // namespace kapy
} // namespace mlir

#endif // KAPY_DIALECT_KAPY_IR_UTILS_H
