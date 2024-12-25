//===- Utils.h --------------------------------------------------*- C++ -*-===//
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

#ifndef KAPY_ANALYSIS_UTILS_H
#define KAPY_ANALYSIS_UTILS_H

#include "mlir/Analysis/DataFlowFramework.h"
#include "mlir/Analysis/SliceAnalysis.h"
#include "mlir/IR/Dialect.h"

namespace mlir {
namespace kapy {
template <typename DialectT> bool fromDialect(Operation *op) {
  return op->getDialect()->getTypeID() == TypeID::get<DialectT>();
}

/// Return true if the given value is not tensor type or is tensor with only 1
/// element.
bool hasSingleValue(Value value);

/// Multi-root DAG topological sort.
/// Perform a topological sort of operations in the `ops` SetVector and return a
/// topologically sorted SetVector.
/// It is faster than mlir::topologicalSort because it prunes nodes that have
/// been visited before.
SetVector<Operation *> multiRootTopoSort(const SetVector<Operation *> &ops);

/// Get a SetVector for slice of the given operation and topological sort it use
/// the function above.
SetVector<Operation *> multiRootGetSlice(Operation *op,
                                         TransitiveFilter bwFilter = nullptr,
                                         TransitiveFilter fwFilter = nullptr);

/// Return true if there is a path from `srcOp` to `dstOp` in `slice`, and all
/// the operations meet with `filter`.
///
/// Note that all the operations in `slice` can't have region(s).
bool hasRestrictedPath(Operation *srcOp, Operation *dstOp,
                       const SetVector<Operation *> &slice,
                       function_ref<bool(Operation *)> filter);

/// Create a basic DataFlowSolver that contains DeadCodeAnalysis and
/// ConstantAnalysis.
std::unique_ptr<DataFlowSolver> createDataFlowSolver();
} // namespace kapy
} // namespace mlir

#endif // KAPY_ANALYSIS_UTILS_H
