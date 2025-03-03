add_mlir_library(TestKapyAnalysis
  TestAlignAnalysis.cpp
  TestAllocAnalysis.cpp
  TestCandidateLayouts.cpp
  TestOpHelpers.cpp

  LINK_LIBS PUBLIC
  MLIRPass
  KapyAnalysis
)
