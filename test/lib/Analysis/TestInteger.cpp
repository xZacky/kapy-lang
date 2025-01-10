//===- TestInteger.cpp ------------------------------------------*- C++ -*-===//
//
// Copyright 2018-2020 Philippe Tillet
// Copyright 2020-2022 OpenAI
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.
//
//===----------------------------------------------------------------------===//
//
// This file is modified from the triton project.
// https://github.com/triton-lang/triton
//
//===----------------------------------------------------------------------===//

#include "kapy/Analysis/Integer.h"
#include "kapy/Dialect/Kapy/IR/Kapy.h"

#include "mlir/Pass/Pass.h"

using namespace mlir;
using namespace mlir::kapy;

namespace {

class KapyTestIntegerPass
    : public PassWrapper<KapyTestIntegerPass, OperationPass<ModuleOp>> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(KapyTestIntegerPass);

  StringRef getArgument() const override { return "kapy-test-integer"; }

  virtual void runOnOperation() override {
    auto module = getOperation();
    ModuleIntegerInfoAnalysis analysis(module);
    auto isIntOrMemRef = [](Type type) {
      return type.isIntOrIndex() || isa<KapyMemRefType>(type);
    };
    module.walk([&](FuncOp funcOp) {
      auto &os = llvm::outs();
      auto funcName = SymbolTable::getSymbolName(funcOp).getValue();
      os << "@" << funcName << "\n";
      funcOp.walk([&](Operation *op) {
        if (op->getNumResults() < 1)
          return;
        for (auto result : op->getResults()) {
          if (!isIntOrMemRef(result.getType()))
            return;
          auto *info = analysis.getIntegerInfo(result);
          if (!info || info->isEntryState())
            return;
          result.print(os);
          os << " // ";
          info->print(os);
          os << "\n";
        }
      });
    });
  }
};

} // namespace

namespace mlir {
namespace test {

void registerKapyTestIntegerPass() { PassRegistration<KapyTestIntegerPass>(); }

} // namespace test
} // namespace mlir
