add_mlir_dialect_library(KapyTransform
  AnalyzeAlignment.cpp
  Combine.cpp
  ReorderBroadcast.cpp
  UpdateMemRefType.cpp

  DEPENDS
  KapyIRIncGen
  KapyTransformIncGen

  LINK_LIBS PUBLIC
  KapyIR
  KapyAnalysis
  MLIRPass
  MLIRTransformUtils
)
