//===- CallGraph.h ----------------------------------------------*- C++ -*-===//
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

#ifndef KAPY_ANALYSIS_CALLGRAPH_H
#define KAPY_ANALYSIS_CALLGRAPH_H

#include "mlir/IR/BuiltinOps.h"
#include "mlir/Interfaces/FunctionInterfaces.h"

namespace mlir {
namespace kapy {
/// This class represents a call graph for a given ModuleOp and holds data of
/// type `T` assoicated with each function.
template <typename T> class CallGraph {
public:
  explicit CallGraph(ModuleOp module) : module(module) { build(); }

  /// Walk the call graph and apply the provided update callbacks to the edges
  /// and nodes.
  template <WalkOrder EdgeUpdateOrder = WalkOrder::PreOrder,
            WalkOrder NodeUpdateOrder = WalkOrder::PreOrder,
            typename EdgeUpdaterT, typename NodeUpdaterT>
  void walk(EdgeUpdaterT edgeUpdater, NodeUpdaterT nodeUpdater) {
    DenseSet<FunctionOpInterface> seen;
    for (auto funcOp : roots)
      doWalk<EdgeUpdateOrder, NodeUpdateOrder>(funcOp, seen, edgeUpdater,
                                               nodeUpdater);
  }

  /// Retrieve the data associated with a function.
  T *getData(FunctionOpInterface funcOp) {
    if (funcToData.contains(funcOp))
      return &funcToData[funcOp];
    return nullptr;
  }

  ModuleOp getModule() const { return module; }
  SmallVector<FunctionOpInterface> getRoots() const { return roots; }
  int getNumFunctions() const { return funcToData.size(); }

  bool isRoot(FunctionOpInterface funcOp) const {
    return is_contained(roots, funcOp);
  }

protected:
  ModuleOp module;
  DenseMap<FunctionOpInterface,
           SmallVector<std::pair<CallOpInterface, FunctionOpInterface>>>
      graph;
  DenseMap<FunctionOpInterface, T> funcToData;
  SmallVector<FunctionOpInterface> roots;

private:
  void build() {
    SymbolTableCollection symbolTable;
    DenseSet<FunctionOpInterface> seen;
    module.walk([&](Operation *op) {
      auto funcOp = op->getParentOfType<FunctionOpInterface>();
      if (auto caller = dyn_cast<CallOpInterface>(op)) {
        auto *callable = caller.resolveCallable(&symbolTable);
        auto callee = dyn_cast_or_null<FunctionOpInterface>(callable);
        if (callee) {
          graph[funcOp].emplace_back(caller, callee);
          seen.insert(callee);
        }
      }
    });
    module.walk([&](FunctionOpInterface funcOp) {
      if (!seen.contains(funcOp))
        roots.push_back(funcOp);
    });
  }

  template <WalkOrder EdgeOrder = WalkOrder::PreOrder,
            WalkOrder NodeOrder = WalkOrder::PreOrder, //
            typename EdgeUpdaterT, typename NodeUpdaterT>
  void doWalk(FunctionOpInterface funcOp, DenseSet<FunctionOpInterface> &seen,
              EdgeUpdaterT edgeUpdater, NodeUpdaterT nodeUpdater) {
    if (seen.contains(funcOp))
      llvm_unreachable("cycle detected in the call graph");
    if constexpr (NodeOrder == WalkOrder::PreOrder)
      nodeUpdater(funcOp);
    for (auto [caller, callee] : graph[funcOp]) {
      if constexpr (EdgeOrder == WalkOrder::PreOrder)
        edgeUpdater(caller, callee);
      doWalk<EdgeOrder, NodeOrder>(callee, seen, edgeUpdater, nodeUpdater);
      if constexpr (EdgeOrder == WalkOrder::PostOrder)
        edgeUpdater(caller, callee);
    }
    if constexpr (NodeOrder == WalkOrder::PostOrder)
      nodeUpdater(funcOp);
    seen.erase(funcOp);
  }
};
} // namespace kapy
} // namespace mlir

#endif // KAPY_ANALYSIS_CALLGRAPH_H
