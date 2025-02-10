#ifndef KAPY_PASSES_H
#define KAPY_PASSES_H

// #include "kapy/Conversion/KapyToKgpu/Passes.h"
// #include "kapy/Dialect/Kapy/Transforms/Passes.h"
// #include "kapy/Dialect/Kgpu/Transforms/Passes.h"
#include "mlir/InitAllPasses.h"

namespace mlir {
namespace test {

void registerTestIntegerAnalysisPass();
void registerTestAllocAnalysisPass();

} // namespace test
} // namespace mlir

using namespace mlir;

inline void registerAllKapyPasses() {
  registerAllPasses();
  // kapy::registerKapyPasses();
  // kapy::registerKgpuPasses();
  // kapy::registerConvertKapyToKgpuPass();
  test::registerTestIntegerAnalysisPass();
  test::registerTestAllocAnalysisPass();
}

#endif // KAPY_PASSES_H
