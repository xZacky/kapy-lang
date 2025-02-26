#include "kapy/Analysis/OpHelpers.h"
#include "kapy/Dialect/Kapy/IR/Kapy.h"
#include "kapy/Dialect/Kgpu/IR/Kgpu.h"
#include "mlir/Pass/Pass.h"

using namespace mlir;
using namespace mlir::kapy;

namespace {

class TestOpHelpersPass
    : public PassWrapper<TestOpHelpersPass, OperationPass<ModuleOp>> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(TestOpHelpersPass);

  virtual StringRef getArgument() const override { return "test-op-helpers"; }

  virtual void runOnOperation() override {
    auto &os = llvm::outs();
    auto module = getOperation();
    module.walk([&](FuncOp funcOp) {
      auto funcName = SymbolTable::getSymbolName(funcOp).getValue();
      os << "@" << funcName << "\n";
      funcOp.walk([&](Operation *op) {
        if (auto reduceOp = dyn_cast<ReduceOp>(op)) {
          ReduceOpHelper helper(reduceOp);
          os << "reduce_op : { ";
          os << "num_shfls = " << helper.getNumShfls() << ", ";
          os << "lane_offset = " << helper.getLaneOffset();
          os << " }\n";
          return;
        }
        if (auto changeOp = dyn_cast<ChangeOp>(op)) {
          ChangeOpHelper helper(changeOp);
          os << "change_op : { ";
          os << "num_shfls = " << helper.getNumShfls() << ", ";
          os << helper.getShflIdxMap();
          os << " }\n";
        }
      });
    });
  }
};

} // namespace

namespace mlir {
namespace test {

void registerTestOpHelpersPass() { PassRegistration<TestOpHelpersPass>(); }

} // namespace kapy
} // namespace mlir
