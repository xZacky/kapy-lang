add_mlir_dialect_library(KgpuIR
  Kgpu.cpp

  DEPENDS
  KgpuIRIncGen

  LINK_LIBS PUBLIC
  KapyIR
)
