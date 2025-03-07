#ifndef KAPY_PASSES_H
#define KAPY_PASSES_H

#include "kapy/Conversion/KapyToKgpu/Passes.h"
#include "kapy/Dialect/Kapy/Transforms/Passes.h"
#include "kapy/Dialect/Kgpu/Transforms/Passes.h"
#include "mlir/InitAllPasses.h"

using namespace mlir;

inline void registerAllKapyPasses() {
  registerAllPasses();
  kapy::registerKapyPasses();
  kapy::registerKgpuPasses();
  kapy::registerConvertKapyToKgpuPass();
}

#endif // KAPY_PASSES_H
