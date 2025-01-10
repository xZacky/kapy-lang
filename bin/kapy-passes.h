#ifndef KAPY_PASSES_H
#define KAPY_PASSES_H

#include "kapy/Conversion/KapyToKgpu/Passes.h"
#include "kapy/Dialect/Kapy/Transform/Passes.h"
#include "kapy/Dialect/Kgpu/Transform/Passes.h"

#include "mlir/InitAllPasses.h"

namespace mlir {
namespace test {

void registerKapyTestIntegerPass();

} // namespace test
} // namespace mlir

using namespace mlir;

inline void registerAllKapyPasses() {
  registerAllPasses();
  registerKapyTransformPasses();
  registerKapyToKgpuPasses();
  registerKgpuTransformPasses();

  test::registerKapyTestIntegerPass();
}

#endif // KAPY_PASSES_H
