//===- AliasAnalysis --------------------------------------------*- C++ -*-===//
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

#ifndef KAPY_ANALYSIS_ALIASANALYSIS_H
#define KAPY_ANALYSIS_ALIASANALYSIS_H

#include "mlir/Analysis/AliasAnalysis.h"
#include "mlir/Analysis/DataFlow/SparseAnalysis.h"

namespace mlir {
namespace kapy {

class AliasInfo {
public:
  AliasInfo() = default;
  AliasInfo(Value value) { insert(value); }

  void insert(Value value) { aliasSet.insert(value); }

  const DenseSet<Value> &getAliasSet() const { return aliasSet; }

  bool operator==(const AliasInfo &other) const {
    return aliasSet == other.aliasSet;
  }

  static AliasInfo getPessimisticState(MLIRContext *context = nullptr) {
    return AliasInfo();
  }

  static AliasInfo getPessimisticState(Value value) { return AliasInfo(); }

  static AliasInfo join(const AliasInfo &lhs, const AliasInfo &rhs);

  void print(llvm::raw_ostream &os) const {
    llvm::interleaveComma(aliasSet, os, [&](Value value) { value.print(os); });
  }

private:
  DenseSet<Value> aliasSet;
};

class AliasAnalysis : public dataflow::SparseForwardDataFlowAnalysis<
                          dataflow::Lattice<AliasInfo>> {
public:
  using dataflow::SparseForwardDataFlowAnalysis<
      dataflow::Lattice<AliasInfo>>::SparseForwardDataFlowAnalysis;

  using dataflow::SparseForwardDataFlowAnalysis<
      dataflow::Lattice<AliasInfo>>::getLatticeElement;

  virtual void setToEntryState(dataflow::Lattice<AliasInfo> *lattice) override {
    auto info = AliasInfo::getPessimisticState(lattice->getPoint());
    propagateIfChanged(lattice, lattice->join(info));
  }

  virtual void
  visitOperation(Operation *op,
                 ArrayRef<const dataflow::Lattice<AliasInfo> *> operands,
                 ArrayRef<dataflow::Lattice<AliasInfo> *> results) override;
};

} // namespace kapy
} // namespace mlir

#endif // KAPY_ANALYSIS_ALIASANALYSIS_H
