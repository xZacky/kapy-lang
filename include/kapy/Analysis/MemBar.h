//===- MemBar.h -------------------------------------------------*- C++ -*-===//
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

#ifndef KAPY_ANALYSIS_MEMBAR_H
#define KAPY_ANALYSIS_MEMBAR_H

#include "kapy/Analysis/Allocation.h"

#include <set>

namespace mlir {
namespace kapy {

class MemBarAnalysis;

class BlockInfo {
public:
  BlockInfo() = default;

  BlockInfo &join(const BlockInfo &other) {
    readIntervals.insert(other.readIntervals.begin(),
                         other.readIntervals.end());
    writeIntervals.insert(other.readIntervals.begin(),
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
  std::set<Interval<int64_t>> readIntervals;
  std::set<Interval<int64_t>> writeIntervals;

  bool isIntersected(const std::set<Interval<int64_t>> &intervals0,
                     const std::set<Interval<int64_t>> &intervals1) const {
    for (const auto &interval0 : intervals0)
      for (const auto &interval1 : intervals1)
        if (interval0.intersects(interval1))
          return true;
    return false;
  }

  friend class MemBarAnalysis;
};

class MemBarAnalysis {
public:
  /// Create a new MemBarAnalysis that generates the shared memory barrier in
  /// the following circumstances:
  /// - RAW: If a shared memory write is followed by a shared memory read, and
  ///   their address are intersected.
  /// - WAR: If a shared memory read is followed by a shared memory write, and
  ///   their address are intersected.
  /// - WAW: Not possible because overlapped memory allocation is not allowed.
  /// - RAR: No write is performed.
  /// Temporary storage of operations such as reduce and change are considered
  /// as both a shared memory write and read.
  /// If the temporary storage is written but not read, it is considered as the
  /// problem of the operation itself.
  MemBarAnalysis() = default;
  explicit MemBarAnalysis(Allocation *allocation) : allocation(allocation) {}

  /// Run this analysis on the given function, insert a barrier if necessary.
  void run(DenseMap<FunctionOpInterface, BlockInfo> &funcInfos) const;

private:
  /// Apply this analysis based on the basic blocks.
  /// TODO: Explain why we don't use ForwardAnalysis.
  void resolve(FunctionOpInterface funcOp,
               DenseMap<FunctionOpInterface, BlockInfo> &funcInfos,
               OpBuilder &builder) const;

  void update(Operation *op, BlockInfo &infoToUpdate,
              DenseMap<FunctionOpInterface, BlockInfo> &funcInfos,
              OpBuilder &builder) const;

  /// Collect the successors of the terminator.
  void visitTerminator(Operation *op,
                       SmallVectorImpl<Block *> &successors) const;

  void insertBarrier(Operation *op, OpBuilder &builder) const;

  Allocation *allocation = nullptr;
};

/// Post-order traversal on the call graph to insert memory barrier instructions
/// for each function. Each function maintains a map that includes all potential
/// buffers after returning. In this way users do not have to explicitly insert
/// memory barriers before and after function calls, but might be a little bit
/// conservative.
class ModuleMemBarAnalysis : public CallGraph<BlockInfo> {
public:
  ModuleMemBarAnalysis(ModuleAllocationAnalysis *allocationAnalysis)
      : CallGraph<BlockInfo>(allocationAnalysis->getModule()),
        allocationAnalysis(allocationAnalysis) {}

  void run() {
    walk<WalkOrder::PreOrder, WalkOrder::PostOrder>(
        [](CallOpInterface caller, FunctionOpInterface callee) {},
        [&](FunctionOpInterface funcOp) {
          auto *allocation = allocationAnalysis->getData(funcOp);
          auto [it, inserted] = funcToData.try_emplace(funcOp, BlockInfo());
          if (inserted) {
            MemBarAnalysis membarAnalysis(allocation);
            membarAnalysis.run(funcToData);
          }
        });
  }

private:
  ModuleAllocationAnalysis *allocationAnalysis = nullptr;
};

} // namespace kapy
} // namespace mlir

#endif // KAPY_ANALYSIS_MEMBAR_H
