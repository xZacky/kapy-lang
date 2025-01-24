//===- Integer.h ------------------------------------------------*- C++ -*-===//
//
// This file defines classes used by data flow analysis for integers.
//
//===----------------------------------------------------------------------===//

#ifndef KAPY_ANALYSIS_INTEGER_H
#define KAPY_ANALYSIS_INTEGER_H

#include "kapy/Analysis/CallGraph.h"

namespace mlir {
namespace kapy {

constexpr char divisibilityAttrName[] = "kapy.divisibility";

/// This lattice value represents known information of integer.
class IntegerInfo {
public:
  IntegerInfo() = default;

  IntegerInfo(int64_t divisibility) : IntegerInfo(divisibility, std::nullopt) {}

  IntegerInfo(int64_t divisibility, std::optional<int64_t> constant)
      : divisibility(divisibility), constant(constant) {}

  int64_t getDivisibility() const { return divisibility; }
  void setDivisibility(int64_t value) { divisibility = value; }

  bool hasConstant() const { return constant.has_value(); }
  bool hasConstantEqualTo(int64_t value) const {
    if (!hasConstant())
      return false;
    return constant.value() == value;
  }
  int64_t getConstant() const { return constant.value(); }

  bool isEntryState() const { return divisibility == 0 && !hasConstant(); }

  /// Initialize pessimistic lattice state from frunction.
  static IntegerInfo getPessimisticState(FunctionOpInterface funcOp,
                                         unsigned argIndex);

  static IntegerInfo getPessimisticState(Value value);

  static IntegerInfo join(const IntegerInfo &lhs, const IntegerInfo &rhs);

  void print(llvm::raw_ostream &os) const {
    os << "{ divisibility = " << divisibility;
    if (constant.has_value())
      os << ", constant = " << constant.value();
    os << " }";
  }

  bool operator==(const IntegerInfo &other) const {
    return divisibility == other.divisibility && constant == other.constant;
  }

private:
  // Divisibility is the greatest power of two divisor of the integer.
  int64_t divisibility = 0;
  // Constant of the integer if we can infer it.
  std::optional<int64_t> constant;
};

/// Module level IntegerInfo analysis based on the call graph, assuming that we
/// do not have recursive functions.
///
/// Since each function will be called multiple times, we need to calculate the
/// IntegerInfo based on all the callers.
///
/// In the future, we can perform optimization using function cloning so that
/// each call site will have unique IntegerInfo.
class ModuleIntegerInfoAnalysis
    : public CallGraph<DenseMap<Value, IntegerInfo>> {
public:
  explicit ModuleIntegerInfoAnalysis(ModuleOp module)
      : CallGraph<DenseMap<Value, IntegerInfo>>(module) {
    SmallVector<FunctionOpInterface> funcOps;
    walk<WalkOrder::PreOrder, WalkOrder::PostOrder>(
        [](CallOpInterface caller, FunctionOpInterface callee) {},
        [&](FunctionOpInterface funcOp) {
          funcOps.push_back(funcOp);
          funcToData.try_emplace(funcOp, DenseMap<Value, IntegerInfo>());
        });
    SymbolTableCollection symbolTable;
    for (auto funcOp : llvm::reverse(funcOps)) {
      initialize(funcOp);
      funcOp.walk([&](CallOpInterface caller) {
        auto *callable = caller.resolveCallable(&symbolTable);
        auto callee = dyn_cast_if_present<FunctionOpInterface>(callable);
        update(caller, callee);
      });
    }
  }

  IntegerInfo *getIntegerInfo(Value value) {
    auto funcOp =
        value.getParentRegion()->getParentOfType<FunctionOpInterface>();
    auto *valueToInfo = getData(funcOp);
    if (!valueToInfo)
      return nullptr;
    auto it = valueToInfo->find(value);
    if (it == valueToInfo->end())
      return nullptr;
    return &(it->second);
  }

private:
  void initialize(FunctionOpInterface funcOp);
  void update(CallOpInterface caller, FunctionOpInterface callee);
};

} // namespace kapy
} // namespace mlir

#endif // KAPY_ANALYSIS_INTEGER_H
