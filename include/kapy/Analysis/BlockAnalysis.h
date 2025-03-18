//===- BlockAnalysis.h ------------------------------------------*- C++ -*-===//
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

#ifndef KAPY_ANALYSIS_BLOCKANALYSIS_H
#define KAPY_ANALYSIS_BLOCKANALYSIS_H

#include "kapy/Analysis/AllocAnalysis.h"
#include <set>

namespace mlir {
namespace kapy {

class BlockInfo {
public:
  BlockInfo() = default;

  BlockInfo &join(const BlockInfo &other) {
    readIntervals.insert(other.readIntervals.begin(),
                         other.readIntervals.end());
    writeIntervals.insert(other.writeIntervals.begin(),
                          other.writeIntervals.end());
    return *this;
  }

  bool isIntersected(const BlockInfo &other) const {
    return /*RAW*/ isIntersected(writeIntervals, other.readIntervals) ||
           /*WAR*/ isIntersected(readIntervals, other.writeIntervals) ||
           /*WAW*/ isIntersected(writeIntervals, other.writeIntervals);
  }

  void sync() {
    readIntervals.clear();
    writeIntervals.clear();
  }

  bool operator==(const BlockInfo &other) const {
    return readIntervals == other.readIntervals &&
           writeIntervals == other.writeIntervals;
  }
  bool operator!=(const BlockInfo &other) const { return !(*this == other); }

private:
  std::set<Interval<uint64_t>> readIntervals;
  std::set<Interval<uint64_t>> writeIntervals;

  bool isIntersected(const std::set<Interval<uint64_t>> &lhsSet,
                     const std::set<Interval<uint64_t>> &rhsSet) const {
    for (const auto &lhs : lhsSet)
      for (const auto &rhs : rhsSet)
        if (lhs.intersects(rhs))
          return true;
    return false;
  }

  friend class BlockAnalysis;
};

class BlockAnalysis {
public:
  explicit BlockAnalysis(AllocInfo *allocInfo) : allocInfo(allocInfo) {
    builder = std::make_unique<OpBuilder>(allocInfo->getFunction());
  }

  /// Run this analysis on the function, insert a barrier if necessary.
  void run(DenseMap<FunctionOpInterface, BlockInfo> &funcToInfo) const;

private:
  AllocInfo *allocInfo;
  std::unique_ptr<OpBuilder> builder;

  /// Visit an operation and update the given BlockInfo.
  void visit(Operation *op, BlockInfo &infoToUpdate,
             DenseMap<FunctionOpInterface, BlockInfo> &funcToInfo) const;

  /// Collect the successors of the terminator.
  void visit(Operation *op, SmallVectorImpl<Block *> &successors) const;

  void insertBarrier(Operation *op, bool after = false) const;
};

class ModuleBlockAnalysis : public CallGraph<BlockInfo> {
public:
  ModuleBlockAnalysis(ModuleAllocAnalysis *allocAnalysis)
      : CallGraph<BlockInfo>(allocAnalysis->getModule()),
        allocAnalysis(allocAnalysis) {}

  void run() {
    walk<WalkOrder::PreOrder, WalkOrder::PostOrder>(
        [](CallOpInterface caller, FunctionOpInterface callee) {},
        [&](FunctionOpInterface funcOp) {
          auto *allocInfo = allocAnalysis->getData(funcOp);
          auto [it, inserted] = funcToData.try_emplace(funcOp, BlockInfo());
          if (inserted) {
            BlockAnalysis blockAnalysis(allocInfo);
            blockAnalysis.run(funcToData);
          }
        });
  }

private:
  ModuleAllocAnalysis *allocAnalysis;
};

} // namespace kapy
} // namespace mlir

#endif // KAPY_ANALYSIS_BLOCKANALYSIS_H
