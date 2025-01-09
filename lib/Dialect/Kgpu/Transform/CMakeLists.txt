add_mlir_dialect_library(KgpuTransform
  CacheMatmulOperand.cpp
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
