add_mlir_dialect_library(KgpuToLLVM
  CallReturnOpToLLVM.cpp
  ConversionPass.cpp
  ConversionTarget.cpp
  ConvertUtils.cpp
  ElementwiseOpToLLVM.cpp
  FuncOpToLLVM.cpp
  LdGlobalOpToLLVM.cpp
  MkGlobalOpToLLVM.cpp
  PTXBuilder.cpp
  SelectOpToLLVM.cpp
  StGlobalOpToLLVM.cpp
  SvGlobalOpToLLVM.cpp
  TypeConverter.cpp

  DEPENDS
  KgpuToLLVMIncGen

  LINK_LIBS PUBLIC
  KapyIR
  KgpuIR
  MLIRPass
  MLIRTransforms
)
