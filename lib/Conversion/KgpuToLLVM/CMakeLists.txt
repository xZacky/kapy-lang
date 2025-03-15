add_mlir_dialect_library(KgpuToLLVM
  BroadcastOpToLLVM.cpp
  CallReturnOpToLLVM.cpp
  ConversionPass.cpp
  ConversionTarget.cpp
  ConvertUtils.cpp
  CpAsyncOpToLLVM.cpp
  ElementwiseOpToLLVM.cpp
  FuncOpToLLVM.cpp
  LdGlobalOpToLLVM.cpp
  LdMatrixOpToLLVM.cpp
  LdSharedOpToLLVM.cpp
  MkGlobalOpToLLVM.cpp
  MkSharedOpToLLVM.cpp
  PTXBuilder.cpp
  SelectOpToLLVM.cpp
  SplatLikeOpToLLVM.cpp
  StGlobalOpToLLVM.cpp
  StSharedOpToLLVM.cpp
  SvGlobalOpToLLVM.cpp
  SvSharedOpToLLVM.cpp
  TypeConverter.cpp

  DEPENDS
  KgpuToLLVMIncGen

  LINK_LIBS PUBLIC
  KapyIR
  KgpuIR
  MLIRPass
  MLIRTransforms
)
