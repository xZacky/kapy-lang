add_mlir_library(KapyAnalysis
  # Allocation.cpp
  # MemBar.cpp
  # OpHelpers.cpp
  Utils.cpp

  DEPENDS
  KapyIRIncGen
  KgpuIRIncGen

  LINK_LIBS PUBLIC
  KapyIR
  KgpuIR
  MLIRAnalysis
)
