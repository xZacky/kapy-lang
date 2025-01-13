//===- Layout.h -------------------------------------------------*- C++ -*-===//
//
// This file defines functions for optimizations of layouts.
//
//===----------------------------------------------------------------------===//

#ifndef KAPY_ANALYSIS_LAYOUT_H
#define KAPY_ANALYSIS_LAYOUT_H

#include "mlir/IR/Attributes.h"
#include "mlir/IR/Operation.h"
#include "llvm/ADT/ArrayRef.h"

namespace mlir {

using llvm::ArrayRef;

namespace kapy {

class MatmulOp;
class SharedMemLayoutAttr;
class RegistersLayoutAttr;
class NvidiaMmaLayoutAttr;
class MmOperandLayoutAttr;

RegistersLayoutAttr getRegistersLayout(MLIRContext *context,
                                       ArrayRef<int64_t> laneLoops,
                                       ArrayRef<int64_t> shape,
                                       ArrayRef<unsigned> order,
                                       int64_t numWarps);
RegistersLayoutAttr getRegistersLayout(MLIRContext *context,
                                       ArrayRef<int64_t> laneLoops,
                                       ArrayRef<int64_t> shape,
                                       int64_t numWarps);
RegistersLayoutAttr getRegistersLayout(MLIRContext *context,
                                       ArrayRef<int64_t> shape,
                                       int64_t numWarps);

bool isNvidiaMmaToMmOperandShortcut(NvidiaMmaLayoutAttr nvmmaLayout,
                                    MmOperandLayoutAttr mmopdLayout);

bool isNvidiaMmaToRegistersShortcut(NvidiaMmaLayoutAttr nvmmaLayout,
                                    RegistersLayoutAttr regisLayout);

NvidiaMmaLayoutAttr getNvidiaMmaLayout(MatmulOp matmulOp, int64_t numWarps);

SharedMemLayoutAttr getSharedMemLayout(MLIRContext *context,
                                       MmOperandLayoutAttr mmopdLayout,
                                       ArrayRef<int64_t> shape,
                                       ArrayRef<unsigned> order);

SetVector<Attribute> getCandidateLayouts(Operation *op, int64_t numWarps);

} // namespace kapy
} // namespace mlir

#endif // KAPY_ANALYSIS_LAYOUT_H
