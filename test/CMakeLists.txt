llvm_canonicalize_cmake_booleans(
  MLIR_ENABLE_BINDINGS_PYTHON
)

configure_lit_site_cfg(
  ${CMAKE_CURRENT_SOURCE_DIR}/lit.site.cfg.py.in
  ${CMAKE_CURRENT_BINARY_DIR}/lit.site.cfg.py
  MAIN_CONFIG
  ${CMAKE_CURRENT_SOURCe_DIR}/lit.cfg.py
)

set(KAPY_TEST_DEPENDS kapy-opt)

set(FILECHECK_PATH "${LLVM_LIBRARY_DIR}/../bin/FileCheck")
set(LIT_ARGS "-Dfilecheck=${FILECHECK_PATH}")
add_lit_testsuite(check-kapy-lit-tests "Running the kapy regression tests"
  ${CMAKE_CURRENT_BINARY_DIR}
  ARGS ${LIT_ARGS}
  DEPENDS ${KAPY_TEST_DEPENDS}
)

set_target_properties(check-kapy-lit-tests PROPERTIES FOLDER "Tests")

add_lit_testsuites(KAPY-LIT-TESTS ${CMAKE_CURRENT_SOURCE_DIR} DEPENDS ${KAPY_TEST_DEPENDS})
