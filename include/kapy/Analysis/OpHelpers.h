//===- OpHelpers.h ----------------------------------------------*- C++ -*-===//
//
// This file defines helpers for operations using shuffle.
//
//===----------------------------------------------------------------------===//

#ifndef KAPY_ANALYSIS_OPHELPERS_H
#define KAPY_ANALYSIS_OPHELPERS_H

#include "mlir/IR/BuiltinTypes.h"

namespace mlir {
namespace kapy {

class ReduceOp;
class ChangeOp;

class ReduceOpHelper {
public:
  explicit ReduceOpHelper(ReduceOp reduceOp);
  explicit ReduceOpHelper(RankedTensorType sourceType, unsigned axis);

  /// Get number of shuffles for each warp.
  int64_t getNumShfls() const;

  /// Get lane offset on the reduction axis.
  int64_t getLaneOffset() const;

private:
  RankedTensorType sourceType;
  unsigned axis;
};

class ChangeOpHelper {
public:
  explicit ChangeOpHelper(ChangeOp changeOp);
  explicit ChangeOpHelper(RankedTensorType sourceType,
                          RankedTensorType resultType);

  /// Get number of shuffles for each warp.
  int64_t getNumShfls() const;

  /// Get an AffineMap to compute the source lane id for "shfl.sync.idx...".
  AffineMap getShflIdxMap() const;

private:
  RankedTensorType sourceType;
  RankedTensorType resultType;
};

} // namespace kapy
} // namespace mlir

#endif // KAPY_ANALYSIS_OPHELPERS_H
