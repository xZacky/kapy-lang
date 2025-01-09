add_mlir_dialect_library(KgpuTransform
  OptimizeMatmul.cpp
  ReorderInstruction.cpp

  DEPENDS
  KapyIRIncGen
  KgpuIRIncGen
  KgpuTransformIncGen

  LINK_LIBS PUBLIC
  KapyIR
  KgpuIR
  KapyAnalysis
  MLIRPass
  MLIRTransformUtils
)
