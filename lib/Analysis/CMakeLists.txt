add_mlir_library(KapyAnalysis
  AliasAnalysis.cpp
  AlignAnalysis.cpp
  AllocAnalysis.cpp
  AnalysisUtils.cpp
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
