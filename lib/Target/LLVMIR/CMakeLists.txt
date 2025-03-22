add_mlir_library(LLVMIRPasses
  BreakPhiStructNodes.cpp
  DIScope.cpp

  DEPENDS
  LLVMIRPassesIncGen

  LINK_LIBS
  ${CMAKE_DL_LIBS}

  PUBLIC
  MLIRBuiltinToLLVMIRTranslation
  MLIRIR
  MLIRLLVMDialect
  MLIRLLVMToLLVMIRTranslation
  MLIRNVVMToLLVMIRTranslation
  MLIRSupport
  MLIRTargetLLVMIRExport
)
