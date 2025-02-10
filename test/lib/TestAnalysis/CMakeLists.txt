add_mlir_library(TestKapyAnalysis
  TestAllocAnalysis.cpp
  TestIntegerAnalysis.cpp

  LINK_LIBS PUBLIC
  MLIRPass
  KapyAnalysis
)
