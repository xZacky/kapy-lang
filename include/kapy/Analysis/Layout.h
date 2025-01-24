//===- Layout.h -------------------------------------------------*- C++ -*-===//
//
// This file defines functions for optimizations of layouts.
//
//===----------------------------------------------------------------------===//

#ifndef KAPY_ANALYSIS_LAYOUT_H
#define KAPY_ANALYSIS_LAYOUT_H

#include "mlir/Support/LLVM.h"

namespace mlir {

class MLIRContext;
class Operation;
class Attribute;
class RankedTensorType;

namespace kapy {

class MatmulOp;
class SharedMemLayoutAttr;
class FragmentsLayoutAttr;
class NvidiaMmaLayoutAttr;

FragmentsLayoutAttr getFragmentsLayout(MLIRContext *context,
                                       ArrayRef<int64_t> laneLoops,
                                       ArrayRef<int64_t> shape,
                                       int64_t numWarps,
                                       bool needTranspose = false);
FragmentsLayoutAttr getFragmentsLayout(MLIRContext *context,
                                       ArrayRef<int64_t> shape,
                                       int64_t numWarps);

NvidiaMmaLayoutAttr getNvidiaMmaLayout(MatmulOp matmulOp, int64_t numWarps);

SharedMemLayoutAttr getSharedMemLayout(RankedTensorType tensorType,
                                       bool needTranspose = false);

SetVector<Attribute> getCandidateLayouts(Operation *op, int64_t numWarps);

} // namespace kapy
} // namespace mlir

#endif // KAPY_ANALYSIS_LAYOUT_H
