add_mlir_dialect_library(KgpuTransform
  AccelerateMatmul.cpp
  OptimizeDotOperand.cpp

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
