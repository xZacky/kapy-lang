add_mlir_dialect_library(KapyIR
  Kapy.cpp
  OpTraits.cpp

  DEPENDS
  KapyIRIncGen

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRArithDialect
  MLIRMathDialect
  MLIRSCFDialect
)
