add_mlir_library(KapyAnalysis
  AllocAnalysis.cpp
  AnalysisUtils.cpp
  IntegerAnalysis.cpp
  LayoutUtils.cpp
  OpHelpers.cpp

  DEPENDS
  KapyIRIncGen
  KgpuIRIncGen

  LINK_LIBS PUBLIC
  KapyIR
  KgpuIR
  MLIRAnalysis
)
