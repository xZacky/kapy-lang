//===- AllocAnalysis.h ------------------------------------------*- C++ -*-===//
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

#ifndef KAPY_ANALYSIS_ALLOCANALYSIS_H
#define KAPY_ANALYSIS_ALLOCANALYSIS_H

#include "kapy/Analysis/CallGraph.h"
#include "llvm/ADT/MapVector.h"
#include <atomic>

namespace mlir {
namespace kapy {

/// Modified from llvm-15.0: llvm/ADT/AddressRanges.h
/// A class that represents an interval, sepcified using a start and an end
/// values: [start, end).
template <typename T> class Interval {
public:
  Interval() = default;
  Interval(T start, T end) : startInt(start), endInt(end) {
    static_assert(std::is_integral_v<T>);
    assert(start <= end);
  }

  T start() const { return startInt; }
  T end() const { return endInt; }
  T size() const { return endInt - startInt; }

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
  T startInt = std::numeric_limits<T>::min();
  T endInt = std::numeric_limits<T>::max();
};
template <typename T> Interval(T, T) -> Interval<T>;

class AllocInfo {
public:
  using BufferId = int64_t;
  static constexpr BufferId INVALID_ID = -1;

  AllocInfo() = default;
  explicit AllocInfo(FunctionOpInterface funcOp) : funcOp(funcOp) {}

  void run(DenseMap<FunctionOpInterface, AllocInfo> &funcToInfo);

  FunctionOpInterface getFunction() const { return funcOp; }

  uint64_t getOffset(BufferId id) const { return buffers.at(id).offset; }
  uint64_t getSize(BufferId id) const { return buffers.at(id).size; }

  Interval<uint64_t> getInterval(BufferId id) const {
    const auto &buffer = buffers.at(id);
    return Interval(buffer.offset, buffer.offset + buffer.size);
  }

  /// This method only return the explicit buffer id.
  BufferId getBufferId(Value value) const {
    if (explicits.contains(value))
      return explicits.lookup(value)->id;
    return INVALID_ID;
  }
  /// This method only return the virtual buffer id.
  BufferId getBufferId(Operation *op) const {
    if (virtuals.contains(op))
      return virtuals.lookup(op)->id;
    return INVALID_ID;
  }

  bool isExplicit(BufferId id) const {
    return buffers.at(id).kind == Buffer::BufferKind::EXPLICIT;
  }
  bool isVirtual(BufferId id) const {
    return buffers.at(id).kind == Buffer::BufferKind::VIRTUAL;
  }

  uint64_t getAllocatedSize() const { return allocatedSize; }

private:
  struct Buffer {
    enum class BufferKind { EXPLICIT, VIRTUAL };

    // MT: thread safe
    inline static std::atomic<BufferId> nextId = 0;

    BufferKind kind;
    BufferId id;
    uint64_t size;
    uint64_t alignment;
    uint64_t offset;

    Buffer() : Buffer(BufferKind::EXPLICIT, 0) {}
    Buffer(BufferKind kind, uint64_t size)
        : kind(kind), id(nextId++), size(size), alignment(128), offset(0) {}

    bool operator==(const Buffer &other) const { return id == other.id; }

    void setOffsetAligned(uint64_t newOffset) {
      offset = llvm::alignTo(newOffset, alignment);
    }
  };

  FunctionOpInterface funcOp;
  llvm::MapVector<Value, Buffer *> explicits;
  llvm::MapVector<Operation *, Buffer *> virtuals;
  llvm::MapVector<Value, SetVector<Buffer *>> aliasSets;
  std::map<BufferId, Buffer> buffers;
  uint64_t allocatedSize = 0;

  template <Buffer::BufferKind Kind, typename KeyT>
  void addBuffer(KeyT key, uint64_t size) {
    auto buffer = Buffer(Kind, size);
    buffers[buffer.id] = std::move(buffer);
    if constexpr (Kind == Buffer::BufferKind::EXPLICIT)
      explicits[key] = &buffers[buffer.id];
    if constexpr (Kind == Buffer::BufferKind::VIRTUAL)
      virtuals[key] = &buffers[buffer.id];
  }

  void addAlias(Value value, Value aliased) {
    aliasSets[value].insert(explicits[aliased]);
  }

  friend class AllocAnalysis;
};

class ModuleAllocAnalysis : public CallGraph<AllocInfo> {
public:
  explicit ModuleAllocAnalysis(ModuleOp module) : CallGraph<AllocInfo>(module) {
    walk<WalkOrder::PreOrder, WalkOrder::PostOrder>(
        [](CallOpInterface caller, FunctionOpInterface callee) {},
        [&](FunctionOpInterface funcOp) {
          auto [it, inserted] = funcToData.try_emplace(funcOp, funcOp);
          if (inserted)
            it->second.run(funcToData);
        });
  }

  uint64_t getAllocatedSize(FunctionOpInterface funcOp) {
    return getData(funcOp)->getAllocatedSize();
  }

  uint64_t getAllocatedSize() {
    uint64_t size = 0;
    for (auto funcOp : getRoots())
      size = std::max(size, getAllocatedSize(funcOp));
    return size;
  }
};

} // namespace kapy
} // namespace mlir

#endif // KAPY_ANALYSIS_ALLOCANALYSIS_H
