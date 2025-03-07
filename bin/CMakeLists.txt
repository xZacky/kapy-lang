get_property(dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)
get_property(conversion_libs GLOBAL PROPERTY MLIR_CONVERSION_LIBS)

add_llvm_executable(kapy-opt kapy-opt.cpp PARTIAL_SOURCES_INTENDED)

llvm_update_compile_flags(kapy-opt)
target_link_libraries(kapy-opt
  PRIVATE
  ${dialect_libs}
  ${conversion_libs}
  KapyAnalysis
  KapySupport
  MLIROptLib
  MLIRPass
  MLIRTransforms
)
mlir_check_all_link_libraries(kapy-opt)

add_llvm_executable(kapy-layout kapy-layout.cpp PARTIAL_SOURCES_INTENDED)
target_link_libraries(kapy-layout PRIVATE ${dialect_libs})
