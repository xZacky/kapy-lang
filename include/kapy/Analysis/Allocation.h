//===- Allocation.h ---------------------------------------------*- C++ -*-===//
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

#ifndef KAPY_ANALYSIS_ALLOCATION_H
#define KAPY_ANALYSIS_ALLOCATION_H

#include "kapy/Analysis/CallGraph.h"
#include <atomic>

namespace mlir {
namespace kapy {

class AllocationAnalysis;

/// Modified from llvm-15.0: llvm/ADT/AddressRanges.h
/// A class that represents an interval, sepcified using a start and an end
/// values: [start, end).
template <typename T> class Interval {
public:
  Interval() = default;
  Interval(T start, T end) : START(start), END(end) {
    static_assert(std::is_integral_v<T>);
    assert(start <= end);
  }

  T start() const { return START; }
  T end() const { return END; }
  T size() const { return END - START; }

  bool intersects(const Interval &other) const {
    return start() < other.end() && other.start() < end();
  }

  bool operator==(const Interval &other) const {
    return start() == other.start() && end() == other.end();
  }
  bool operator!=(const Interval &other) const { return !(*this == other); }
  bool operator<(const Interval &other) const {
    return std::make_pair(start(), end()) <
           std::make_pair(other.start(), other.end());
  }

private:
  T START = std::numeric_limits<T>::min();
  T END = std::numeric_limits<T>::max();
};
template <typename T> Interval(T, T) -> Interval<T>;

class Allocation {
public:
  using BufferId = int64_t;
  static constexpr BufferId invalidId = -1;
  Allocation() = default;
  explicit Allocation(Operation *op) : operation(op) {}

  void run(DenseMap<FunctionOpInterface, Allocation> &funcToAllocation);

  Operation *getOperation() { return operation; }

  int64_t getOffset(BufferId id) const { return buffers.at(id).offset; }
  int64_t getSize(BufferId id) const { return buffers.at(id).size; }
  Interval<int64_t> getInterval(BufferId id) const {
    const auto &buffer = buffers.at(id);
    return Interval(buffer.offset, buffer.offset + buffer.size);
  }

  /// This method only return the explicit buffer id.
  BufferId getBufferId(Value value) const {
    if (explicits.contains(value))
      return explicits.lookup(value)->id;
    return invalidId;
  }
  /// This method only return the scratch or virtual buffer id.
  BufferId getBufferId(Operation *op) const {
    if (scratchs.contains(op))
      return scratchs.lookup(op)->id;
    if (virtuals.contains(op))
      return virtuals.lookup(op)->id;
    return invalidId;
  }

  bool isExplicit(BufferId id) const {
    return buffers.at(id).kind == Buffer::BufferKind::EXPLICIT;
  }
  bool isScratch(BufferId id) const {
    return buffers.at(id).kind == Buffer::BufferKind::SCRATCH;
  }
  bool isVirtual(BufferId id) const {
    return buffers.at(id).kind == Buffer::BufferKind::VIRTUAL;
  }

  int64_t getAllocatedSize() const { return allocatedSize; }

private:
  struct Buffer {
    // EXPLICIT: LocalAllocOp
    // SCRATCH: ReduceOp, ChangeOp
    // VIRTUAL: CallOp
    enum class BufferKind { EXPLICIT, SCRATCH, VIRTUAL };

    // MT: thread safe
    inline static std::atomic<BufferId> nextId = 0;

    BufferKind kind;
    BufferId id;
    int64_t size;
    int64_t alignment;
    int64_t offset;

    Buffer() : Buffer(BufferKind::EXPLICIT, 0) {}
    Buffer(BufferKind kind, int64_t size)
        : kind(kind), id(nextId++), size(size), alignment(128), offset(0) {}

    bool operator==(const Buffer &other) const { return id == other.id; }

    void setOffsetAligned(int64_t newOffset) {
      offset = llvm::alignTo(newOffset, alignment);
    }
  };

  Operation *operation = nullptr;
  llvm::MapVector<Value, Buffer *> explicits;
  DenseMap<Operation *, Buffer *> scratchs;
  DenseMap<Operation *, Buffer *> virtuals;
  std::map<BufferId, Buffer> buffers;
  int64_t allocatedSize = 0;

  template <Buffer::BufferKind Kind, typename T, typename... Ts>
  void addBuffer(T &key, Ts &&...args) {
    auto buffer = Buffer(Kind, std::forward<Ts>(args)...);
    buffers[buffer.id] = std::move(buffer);
    if constexpr (Kind == Buffer::BufferKind::EXPLICIT)
      explicits[key] = &buffers[buffer.id];
    else if constexpr (Kind == Buffer::BufferKind::SCRATCH)
      scratchs[key] = &buffers[buffer.id];
    else if constexpr (Kind == Buffer::BufferKind::VIRTUAL)
      virtuals[key] = &buffers[buffer.id];
  }

  friend class AllocationAnalysis;
};

class ModuleAllocationAnalysis : public CallGraph<Allocation> {
public:
  explicit ModuleAllocationAnalysis(ModuleOp module)
      : CallGraph<Allocation>(module) {
    walk<WalkOrder::PreOrder, WalkOrder::PostOrder>(
        [](CallOpInterface caller, FunctionOpInterface callee) {},
        [&](FunctionOpInterface funcOp) {
          auto [it, inserted] = funcToData.try_emplace(funcOp, funcOp);
          if (inserted)
            it->second.run(funcToData);
        });
  }

  int64_t getAllocatedSize(FunctionOpInterface funcOp) {
    return getData(funcOp)->getAllocatedSize();
  }

  int64_t getAllocatedSize() {
    int64_t size = 0;
    for (auto funcOp : getRoots())
      size = std::max(size, getAllocatedSize(funcOp));
    return size;
  }
};

} // namespace kapy
} // namespace mlir

#endif // KAPY_ANALYSIS_ALLOCATION_H
