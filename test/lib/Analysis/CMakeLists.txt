add_mlir_library(KapyTestAnalysis
  TestInteger.cpp

  LINK_LIBS PUBLIC
  MLIRPass
  KapyAnalysis
)
