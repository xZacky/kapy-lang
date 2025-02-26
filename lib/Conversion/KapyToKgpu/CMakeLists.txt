add_mlir_conversion_library(KapyToKgpu
  ConversionPass.cpp
  TypeConverter.cpp

  DEPENDS
  KapyToKgpuIncGen

  LINK_LIBS PUBLIC
  KapyIR
  KgpuIR
  KapyAnalysis
  MLIRPass
  MLIRTransforms
)
