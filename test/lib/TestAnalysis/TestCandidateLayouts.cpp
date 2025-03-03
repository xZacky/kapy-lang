#include "kapy/Analysis/LayoutUtils.h"
#include "kapy/Dialect/Kapy/IR/Kapy.h"
#include "kapy/Dialect/Kgpu/IR/Kgpu.h"
#include "mlir/Pass/Pass.h"

using namespace mlir;
using namespace mlir::kapy;

namespace {

class TestCandidateLayoutsPass
    : public PassWrapper<TestCandidateLayoutsPass, OperationPass<ModuleOp>> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(TestCandidateLayoutsPass);

  virtual StringRef getArgument() const override {
    return "test-candidate-layouts";
  }

  virtual void runOnOperation() override {
    auto &os = llvm::outs();
    auto module = getOperation();
    module.walk([&](FuncOp funcOp) {
      auto funcName = SymbolTable::getSymbolName(funcOp).getValue();
      os << "@" << funcName << "\n";
      funcOp.walk([&](Operation *op) {
        if (isa<CpAsyncGlobalToSharedOp>(op))
          return;
        if (isGlobalRead(op) || isGlobalWrite(op)) {
          auto layouts = getCandidateLayouts(op);
          os << op->getName().getIdentifier().getValue() << ": { ";
          llvm::interleaveComma(
              layouts, os, [&](FragmentsLayoutAttr layout) { os << layout; });
          os << " }\n";
          return;
        }
      });
    });
  }
};

} // namespace

namespace mlir {
namespace test {

void registerTestCandidateLayoutsPass() {
  PassRegistration<TestCandidateLayoutsPass>();
}

} // namespace test
} // namespace mlir
