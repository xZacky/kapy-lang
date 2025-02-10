//===- OpHelpers.h ----------------------------------------------*- C++ -*-===//
//
// This file defines helpers for operations that need shared memory as scratch.
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

  /// Get the minimum offset for "shfl.sync.bfly...".
  int64_t getMinShflBflyOffset() const;

  /// We represent lane sync cost by the number of shuffles for each warp.
  int64_t getLaneSyncCost() const { return laneSyncCost; }
  /// We represent warp sync cost by the number of shared memory transactions
  /// for each warp.
  int64_t getWarpSyncCost() const { return warpSyncCost; }

  SmallVector<int64_t, 2> getScratchShape() const;

  uint64_t getScratchSizeInBytes() const;

private:
  RankedTensorType sourceType;
  unsigned axis;
  int64_t laneSyncCost = 0;
  int64_t warpSyncCost = 0;

  void runRelationAnalysis();

  int64_t getScratchNumCols() const;
};

class ChangeOpHelper {
public:
  explicit ChangeOpHelper(ChangeOp changeOp);
  explicit ChangeOpHelper(RankedTensorType sourceType,
                          RankedTensorType resultType);

  /// Return true if we use a transposed scratch.
  bool useTransposedScratch() const;

  /// We represent lane sync cost by the number of shuffles for each warp.
  int64_t getLaneSyncCost() const { return laneSyncCost; }
  /// We represent warp sync cost by the number of shared memory transactions
  /// for each warp.
  int64_t getWarpSyncCost() const { return warpSyncCost; }

  SmallVector<int64_t, 2> getScratchShape() const;

  uint64_t getScratchSizeInBytes() const;

private:
  RankedTensorType sourceType;
  RankedTensorType resultType;
  int64_t laneSyncCost = 0;
  int64_t warpSyncCost = 0;

  void runRelationAnalysis();

  int64_t getScratchNumCols() const;
};

} // namespace kapy
} // namespace mlir

#endif // KAPY_ANALYSIS_OPHELPERS_H
