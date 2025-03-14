//===- LayoutUtils.h --------------------------------------------*- C++ -*-===//
//
// This file defines classes and functions about layout.
//
//===----------------------------------------------------------------------===//

#ifndef KAPY_SUPPORT_LAYOUTUTILS_H
#define KAPY_SUPPORT_LAYOUTUTILS_H

#include "mlir/IR/BuiltinTypes.h"

namespace mlir {

class Operation;
class RankedTensorType;

namespace kapy {

class LdMatrixOp;
class MatmulOp;
class ReduceOp;
class ChangeOp;
class FragmentsLayoutAttr;
class SwizzlingLayoutAttr;

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
  unsigned axis = 0;
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

/// Get a fragments layout with the given parameters, this function will try to
/// make as many contiguous threads on the major axis as possible.
FragmentsLayoutAttr getFragmentsLayout(ArrayRef<int64_t> laneLoops,
                                       RankedTensorType tensorType,
                                       bool rowMajor = true);

/// Call the function above but set lane loops as [1, 1].
FragmentsLayoutAttr getFragmentsLayout(RankedTensorType tensorType,
                                       bool rowMajor = true);

std::array<FragmentsLayoutAttr, 2> getDefaultLayouts(LdMatrixOp op);

std::array<FragmentsLayoutAttr, 3> getDefaultLayouts(MatmulOp op);

/// Get all the candidate layouts for the given operation (must be global access
/// operation). We will select from these layouts in KgpuOptimizeLayoutPass.
SetVector<FragmentsLayoutAttr> getCandidateLayouts(Operation *op);

/// Get swizzling layouts that can avoid bank conflict for the given access.
SetVector<SwizzlingLayoutAttr> getSwizzlingLayouts(RankedTensorType sharedType,
                                                   RankedTensorType tensorType,
                                                   int64_t alignment);

} // namespace kapy
} // namespace mlir

#endif // KAPY_SUPPORT_LAYOUTUTILS_H
