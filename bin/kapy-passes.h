#ifndef KAPY_PASSES_H
#define KAPY_PASSES_H

#include "kapy/Conversion/KapyToKgpu/Passes.h"
#include "kapy/Dialect/Kapy/Transform/Passes.h"

#include "mlir/InitAllPasses.h"

using namespace mlir;

inline void registerAllKapyPasses() {
  registerAllPasses();
  registerKapyTransformPasses();
  registerKapyToKgpuPasses();
}

#endif // KAPY_PASSES_H
