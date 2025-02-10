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
class Attribute;
class RankedTensorType;

namespace kapy {

class MatmulOp;
class GlobalMemRefType;
class FragmentsLayoutAttr;
class SwizzleMemLayoutAttr;

/// Get a fragments layout with the given parameters, this function will try to
/// make as many contiguous threads on the major axis as possible.
FragmentsLayoutAttr getFragmentsLayout(ArrayRef<int64_t> laneLoops,
                                       RankedTensorType tensorType,
                                       int64_t numWarps, bool rowMajor = true);

/// Call the function above but set lane loops as [1, 1].
FragmentsLayoutAttr getFragmentsLayout(RankedTensorType tensorType,
                                       int64_t numWarps, bool rowMajor = true);

/// Get a fragments layout for the given MatmulOp.
FragmentsLayoutAttr getFragmentsLayout(MatmulOp matmulOp, int64_t numWarps);

/// Get all the candidate layouts for the given operation (must be memory access
/// or MatmulOp), we will choose from these layouts in KgpuOptimizeLayoutPass.
SetVector<Attribute> getCandidateLayouts(Operation *op, int64_t numWarps);

/// Check if memory access is coalesced.
// bool isCoalesced(GlobalMemRefType globalType, RankedTensorType tensorType);

} // namespace kapy
} // namespace mlir

#endif // KAPY_ANALYSIS_LAYOUTUTILS_H
