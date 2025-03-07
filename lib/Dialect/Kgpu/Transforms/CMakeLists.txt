add_mlir_dialect_library(KgpuTransforms
  Coalesce.cpp
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
