//===- AlignAnalysis.h ----------------------------------------*- C++ -*-===//
//
// This file defines classes used by data flow analysis for alignment.
//
//===----------------------------------------------------------------------===//

#ifndef KAPY_ANALYSIS_ALIGNANALYSIS_H
#define KAPY_ANALYSIS_ALIGNANALYSIS_H

#include "kapy/Analysis/CallGraph.h"

namespace mlir {
namespace kapy {

class AlignInfo {
public:
  AlignInfo() = default;

  AlignInfo(int64_t alignment) : AlignInfo(alignment, std::nullopt) {}

  AlignInfo(int64_t alignment, std::optional<int64_t> constant)
      : alignment(alignment), constant(constant) {}

  int64_t getAlignment() const { return alignment; }
  void setAlignment(int64_t value) { alignment = value; }

  bool hasConstant() const { return constant.has_value(); }
  bool hasConstantEqualTo(int64_t value) const {
    if (!hasConstant())
      return false;
    return constant.value() == value;
  }
  int64_t getConstant() const { return constant.value(); }

  bool isEntryState() const { return alignment == 0 && !hasConstant(); }

  static AlignInfo getPessimisticState(FunctionOpInterface funcOp,
                                       unsigned argIndex);

  static AlignInfo getPessimisticState(Value value);

  static AlignInfo join(const AlignInfo &lhs, const AlignInfo &rhs);

  void print(llvm::raw_ostream &os) const {
    os << "{ alignment = " << alignment;
    if (constant.has_value())
      os << ", constant = " << constant.value();
    os << " }";
  }

  bool operator==(const AlignInfo &other) const {
    return alignment == other.alignment && constant == other.constant;
  }

private:
  // Alignment is the greatest power of 2 divisor. Entry state is 0.
  int64_t alignment = 0;
  // Constant if we can infer it.
  std::optional<int64_t> constant;
};

class ModuleAlignAnalysis : public CallGraph<DenseMap<Value, AlignInfo>> {
public:
  explicit ModuleAlignAnalysis(ModuleOp module)
      : CallGraph<DenseMap<Value, AlignInfo>>(module) {
    SmallVector<FunctionOpInterface> funcOps;
    walk<WalkOrder::PreOrder, WalkOrder::PostOrder>(
        [](CallOpInterface caller, FunctionOpInterface callee) {},
        [&](FunctionOpInterface funcOp) {
          funcOps.push_back(funcOp);
          funcToData.try_emplace(funcOp, DenseMap<Value, AlignInfo>());
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

  AlignInfo *getAlignInfo(Value value) {
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

#endif // KAPY_ANALYSIS_ALIGNANALYSIS_H
