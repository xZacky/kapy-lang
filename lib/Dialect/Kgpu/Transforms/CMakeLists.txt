add_mlir_dialect_library(KgpuTransforms
  AllocSharedMemory.cpp
  Coalesce.cpp
  InsertSyncBarrier.cpp
  OptimizeLayout.cpp

  DEPENDS
  KapyIRIncGen
  KgpuIRIncGen
  KgpuTransformsIncGen

  LINK_LIBS PUBLIC
  KapyIR
  KgpuIR
  KapyAnalysis
  KapySupport
  MLIRPass
  MLIRTransformUtils
)
