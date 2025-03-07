add_mlir_library(KapySupport
  LayoutUtils.cpp

  DEPENDS
  KapyIRIncGen
  KgpuIRIncGen

  LINK_LIBS PUBLIC
  KapyIR
  KgpuIR
)
