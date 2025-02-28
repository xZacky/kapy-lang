//===- LayoutUtils.h --------------------------------------------*- C++ -*-===//
//
// This file defines functions for layout optimization.
//
//===----------------------------------------------------------------------===//

#ifndef KAPY_ANALYSIS_LAYOUTUTILS_H
#define KAPY_ANALYSIS_LAYOUTUTILS_H

#include "mlir/Support/LLVM.h"

namespace mlir {

class Operation;
class RankedTensorType;

namespace kapy {

class MatmulOp;
class FragmentsLayoutAttr;

/// Get a fragments layout with the given parameters, this function will try to
/// make as many contiguous threads on the major axis as possible.
FragmentsLayoutAttr getFragmentsLayout(ArrayRef<int64_t> laneLoops,
                                       RankedTensorType tensorType,
                                       bool rowMajor = true);

/// Call the function above but set lane loops as [1, 1].
FragmentsLayoutAttr getFragmentsLayout(RankedTensorType tensorType,
                                       bool rowMajor = true);

std::array<FragmentsLayoutAttr, 3> getOperandLayouts(MatmulOp matmulOp);

/// Get all the candidate layouts for the given operation (must be memory access
/// operation), we will choose from these layouts in KgpuOptimizeLayoutPass.
SetVector<FragmentsLayoutAttr> getCandidateLayouts(Operation *op);

/// Return true if global memory access is coalesced.
bool isCoalescedGlobalAccess(RankedTensorType globalType,
                             RankedTensorType tensorType, int64_t alignment);

/// Return true if shared memory access is 0 bank conflict.
bool is0ConflictSharedAccess(RankedTensorType sharedType,
                             RankedTensorType tensorType);

} // namespace kapy
} // namespace mlir

#endif // KAPY_ANALYSIS_LAYOUTUTILS_H
