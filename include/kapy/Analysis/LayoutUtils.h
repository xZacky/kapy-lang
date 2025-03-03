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
class LdMatrixOp;
class FragmentsLayoutAttr;
class SwizzlingLayoutAttr;

/// Get a fragments layout with the given parameters, this function will try to
/// make as many contiguous threads on the major axis as possible.
FragmentsLayoutAttr getFragmentsLayout(ArrayRef<int64_t> laneLoops,
                                       RankedTensorType tensorType,
                                       bool rowMajor = true);

/// Call the function above but set lane loops as [1, 1].
FragmentsLayoutAttr getFragmentsLayout(RankedTensorType tensorType,
                                       bool rowMajor = true);

std::array<FragmentsLayoutAttr, 3> getDefaultLayouts(MatmulOp op);

std::array<FragmentsLayoutAttr, 2> getDefaultLayouts(LdMatrixOp op);

/// Get all the candidate layouts for the given operation (must be global access
/// operation). We will select from these layouts in KgpuOptimizeLayoutPass.
SetVector<FragmentsLayoutAttr> getCandidateLayouts(Operation *op);

/// Get swizzling layouts that can avoid bank conflicts for the given access.
SetVector<SwizzlingLayoutAttr> getSwizzlingLayouts(RankedTensorType tensorType);

} // namespace kapy
} // namespace mlir

#endif // KAPY_ANALYSIS_LAYOUTUTILS_H
