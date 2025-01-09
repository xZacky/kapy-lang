//===- Layout.h -------------------------------------------------*- C++ -*-===//
//
// This file defines functions for optimizations of layouts.
//
//===----------------------------------------------------------------------===//

#ifndef KAPY_ANALYSIS_LAYOUT_H
#define KAPY_ANALYSIS_LAYOUT_H

#include "mlir/IR/MLIRContext.h"
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
                                       ArrayRef<int64_t> loopsPerWarp,
                                       ArrayRef<int64_t> loopsPerLane,
                                       ArrayRef<int64_t> shape,
                                       int64_t numWarps);
RegistersLayoutAttr getRegistersLayout(MLIRContext *context,
                                       ArrayRef<int64_t> shape,
                                       int64_t numWarps);

bool isNvidiaMmaToMmOperandShortcut(NvidiaMmaLayoutAttr nvmmaLayout,
                                    MmOperandLayoutAttr mmopdLayout);

NvidiaMmaLayoutAttr getNvidiaMmaLayout(MatmulOp matmulOp, int64_t numWarps);

SharedMemLayoutAttr getSharedMemLayout(MLIRContext *context,
                                       MmOperandLayoutAttr mmopdLayout,
                                       ArrayRef<int64_t> shape,
                                       ArrayRef<unsigned> order);

} // namespace kapy
} // namespace mlir

#endif // KAPY_ANALYSIS_LAYOUT_H
