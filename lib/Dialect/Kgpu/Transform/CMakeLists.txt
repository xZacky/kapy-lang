add_mlir_dialect_library(KgpuTransform
  CacheMatmulOperand.cpp
  Coalesce.cpp
  OptimizeLayout.cpp
  OptimizeMatmul.cpp
  ReorderInstruction.cpp
  Utils.cpp

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
