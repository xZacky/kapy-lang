add_mlir_library(KapyAnalysis
  Affine.cpp
  Allocation.cpp
  Integer.cpp
  Layout.cpp
  MemBar.cpp
  OpHelpers.cpp
  Utils.cpp

  DEPENDS
  KapyIRIncGen
  KgpuIRIncGen

  LINK_LIBS PUBLIC
  KapyIR
  KgpuIR
  MLIRAnalysis
)
