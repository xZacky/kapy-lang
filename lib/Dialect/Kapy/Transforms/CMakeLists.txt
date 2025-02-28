add_mlir_dialect_library(KapyTransforms
  AnalyzeAlignment.cpp
  Combine.cpp

  DEPENDS
  KapyIRIncGen
  KapyTransformsIncGen

  LINK_LIBS PUBLIC
  KapyIR
  KapyAnalysis
  MLIRPass
  MLIRTransformUtils
)
