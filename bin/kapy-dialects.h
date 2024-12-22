#ifndef KAPY_DIALECTS_H
#define KAPY_DIALECTS_H

#include "kapy/Dialect/Kapy/IR/Kapy.h"
#include "kapy/Dialect/Kgpu/IR/Kgpu.h"

using namespace mlir;

inline void registerAllKapyDialects(DialectRegistry &registry) {
  registry.insert<kapy::KapyDialect, kapy::KgpuDialect>();
}

#endif // KAPY_DIALECTS_H
