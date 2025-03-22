#include "pybind11/pybind11.h"

namespace py = pybind11;

void init_kapy_ir(py::module &&m);
void init_kapy_passes(py::module &&m);

PYBIND11_MODULE(libtriton, m) {
  m.doc() = "Python bindings to the Kapy C++ APIs";
  init_kapy_ir(m.def_submodule("ir"));
  init_kapy_passes(m.def_submodule("passes"));
}
