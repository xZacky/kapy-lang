//===- TransformUtils.h -----------------------------------------*- C++ -*-===//
//
// This file defines functions for transformation.
//
//===----------------------------------------------------------------------===//

#ifndef KAPY_DIALECT_KAPY_TRANSFORMS_TRANSFORMUTILS_H
#define KAPY_DIALECT_KAPY_TRANSFORMS_TRANSFORMUTILS_H

#include "mlir/IR/Value.h"

namespace mlir {
namespace kapy {

/// Propagate a new memory layout from `value`.
void propagateMemoryLayout(Value value, Attribute layout,
                           DenseSet<Value> &seen);

} // namespace kapy
} // namespace mlir

#endif // KAPY_DIALECT_KAPY_TRANSFORMS_TRANSFORMUTILS_H
