add_mlir_dialect_library(KapyTransforms
  Analyze.cpp
  Combine.cpp
  TransformUtils.cpp

  DEPENDS
  KapyIRIncGen
  KapyTransformsIncGen

  LINK_LIBS PUBLIC
  KapyIR
  KapyAnalysis
  MLIRPass
  MLIRTransformUtils
)
