add_mlir_dialect_library(KgpuTransforms
  CoalesceAccess.cpp
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
