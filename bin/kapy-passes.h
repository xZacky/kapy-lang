#ifndef KAPY_PASSES_H
#define KAPY_PASSES_H

#include "kapy/Conversion/KapyToKgpu/Passes.h"
// #include "kapy/Dialect/Kapy/Transforms/Passes.h"
// #include "kapy/Dialect/Kgpu/Transforms/Passes.h"
#include "mlir/InitAllPasses.h"

namespace mlir {
namespace test {

void registerTestAlignAnalysisPass();
void registerTestAllocAnalysisPass();
void registerTestOpHelpersPass();

} // namespace test
} // namespace mlir

using namespace mlir;

inline void registerAllKapyPasses() {
  registerAllPasses();
  // kapy::registerKapyPasses();
  // kapy::registerKgpuPasses();
  kapy::registerConvertKapyToKgpuPass();
  test::registerTestAlignAnalysisPass();
  test::registerTestAllocAnalysisPass();
  test::registerTestOpHelpersPass();
}

#endif // KAPY_PASSES_H
