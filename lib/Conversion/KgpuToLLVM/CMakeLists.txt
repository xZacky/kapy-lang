add_mlir_dialect_library(KgpuToLLVM
  ArangeOpToLLVM.cpp
  BroadcastOpToLLVM.cpp
  CallReturnOpToLLVM.cpp
  ChangeOpToLLVM.cpp
  ConversionPass.cpp
  ConversionTarget.cpp
  ConvertUtils.cpp
  CpAsyncOpToLLVM.cpp
  ElementwiseOpToLLVM.cpp
  FuncOpToLLVM.cpp
  LdGlobalOpToLLVM.cpp
  LdMatrixOpToLLVM.cpp
  LdSharedOpToLLVM.cpp
  MatmulOpToLLVM.cpp
  MkGlobalOpToLLVM.cpp
  MkSharedOpToLLVM.cpp
  ParallelIdOpToLLVM.cpp
  PTXBuilder.cpp
  ReduceOpToLLVM.cpp
  SelectOpToLLVM.cpp
  SplatLikeOpToLLVM.cpp
  StGlobalOpToLLVM.cpp
  StSharedOpToLLVM.cpp
  SvGlobalOpToLLVM.cpp
  SvSharedOpToLLVM.cpp
  TransposeOpToLLVM.cpp
  TypeConverter.cpp

  DEPENDS
  KgpuToLLVMIncGen

  LINK_LIBS PUBLIC
  KapyIR
  KgpuIR
  MLIRPass
  MLIRTransforms
)
