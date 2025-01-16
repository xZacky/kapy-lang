add_mlir_library(KapyTestAnalysis
  TestAllocation.cpp
  TestInteger.cpp
  TestMemBar.cpp

  LINK_LIBS PUBLIC
  MLIRPass
  KapyAnalysis
)
