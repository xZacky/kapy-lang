set(LLVM_TARGET_DEFINITIONS Dialect.td)
mlir_tablegen(Dialect.h.inc -gen-dialect-decls)
mlir_tablegen(Dialect.cpp.inc -gen-dialect-defs)

set(LLVM_TARGET_DEFINITIONS Enums.td)
mlir_tablegen(Enums.h.inc -gen-enum-decls)
mlir_tablegen(Enums.cpp.inc -gen-enum-defs)

set(LLVM_TARGET_DEFINITIONS Attrs.td)
mlir_tablegen(Attrs.h.inc -gen-attrdef-decls)
mlir_tablegen(Attrs.cpp.inc -gen-attrdef-defs)

set(LLVM_TARGET_DEFINITIONS Ops.td)
mlir_tablegen(Ops.h.inc -gen-op-decls)
mlir_tablegen(Ops.cpp.inc -gen-op-defs)

add_public_tablegen_target(KapyIRIncGen)
