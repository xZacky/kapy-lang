set(LLVM_TARGET_DEFINITIONS Passes.td)
mlir_tablegen(Passes.h.inc -gen-pass-decls -name Kapy)

set(LLVM_TARGET_DEFINITIONS Combine.td)
mlir_tablegen(Combine.cpp.inc -gen-rewriters)

add_public_tablegen_target(KapyTransformsIncGen)
