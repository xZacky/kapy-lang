#include "kapy-dialects.h"
#include "kapy-passes.h"

#include "mlir/Tools/mlir-opt/MlirOptMain.h"

int main(int argc, char **argv) {
  DialectRegistry registry;
  registerAllKapyDialects(registry);
  registerAllKapyPasses();

  return asMainReturnCode(
      MlirOptMain(argc, argv, "Kapy compiler driver\n", registry));
}
