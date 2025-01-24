//===- OpHelpers.cpp --------------------------------------------*- C++ -*-===//
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

#include "kapy/Analysis/OpHelpers.h"
#include "kapy/Analysis/Affine.h"
#include "kapy/Dialect/Kapy/IR/Kapy.h"
#include "kapy/Dialect/Kapy/IR/Utils.h"
#include "kapy/Dialect/Kgpu/IR/Kgpu.h"

using namespace mlir;
using namespace mlir::kapy;
using namespace mlir::presburger;

static FlatLinearConstraints getLaneSet() {
  FlatLinearConstraints set(1);
  set.addBound(BoundType::EQ, 0, 0);
  return set;
}

static FlatLinearConstraints getWarpSet() {
  FlatLinearConstraints set(1);
  set.addBound(BoundType::LB, 0, 0);
  set.addBound(BoundType::UB, 0, 32);
  return set;
}

ReduceOpHelper::ReduceOpHelper(Operation *op) {
  auto reduceOp = cast<ReduceOp>(op);
  auto axis = reduceOp.getAxis();
  auto type = reduceOp.getOperand().getType();
  auto rank = type.getRank();
  auto shape = type.getShape();
  auto tensorMap = getTensorMap(shape, type.getEncoding());
  // We build a FlatAffineRelation between tensor indices of a reduction slice
  // and the corresponding threads.
  FlatAffineRelation reduceRel;
  if (failed(getRelationFromMap(tensorMap, reduceRel)))
    llvm_unreachable("failed to get relation from map");
  // Add bounds for a reduction slice.
  for (unsigned i = 0; i < rank; ++i) {
    if (i != axis) {
      reduceRel.addBound(BoundType::EQ, i, 0);
    } else {
      reduceRel.addBound(BoundType::LB, i, 0);
      reduceRel.addBound(BoundType::UB, i, shape[axis]);
    }
  }
  auto threadSet = reduceRel.getRangeSet();
  laneSynchronous = threadSet.isSubsetOf(getLaneSet());
  warpSynchronous = threadSet.isSubsetOf(getWarpSet());
  if (laneSynchronous || warpSynchronous) {
    scratchSize = 0;
  } else {
    int64_t numWarps = getNumWarps(reduceOp->getParentOfType<ModuleOp>());
    int64_t numWarpsToSync = 0;
    for (int64_t warpId = 0; warpId < numWarps; ++warpId)
      if (threadSet.containsPoint(warpId * 32))
        ++numWarpsToSync;
      else
        break;
    auto bitWidth = getIntOrFloatBitWidth(type);
    scratchSize = ceilDiv<unsigned>(bitWidth, 8) * numWarpsToSync;
    if (rank == 2)
      scratchSize *= shape[axis == 1 ? 0 : 1];
  }
}

ChangeOpHelper::ChangeOpHelper(Operation *op) {
  auto changeOp = cast<ChangeOp>(op);
  auto operandType = changeOp.getOperand().getType();
  auto resultType = changeOp.getType();
  auto shape = resultType.getShape();
  auto operandMap = getTensorMap(shape, operandType.getEncoding());
  auto resultMap = getTensorMap(shape, resultType.getEncoding());
  // FlatAffineRelation between tensor indices and threads for operand.
  FlatAffineRelation operandRel;
  // FlatAffineRelation between tensor indices and threads for result.
  FlatAffineRelation resultRel;
  if (failed(getRelationFromMap(operandMap, operandRel)) ||
      failed(getRelationFromMap(resultMap, resultRel)))
    llvm_unreachable("failed to get relation from map");
  // Swap domain and range, now first variable is thread.
  operandRel.inverse();
  resultRel.inverse();
  // Add bound for a lane.
  operandRel.addBound(BoundType::EQ, 0, 0);
  resultRel.addBound(BoundType::EQ, 0, 0);
  auto operandSet = operandRel.getRangeSet();
  auto resultSet = resultRel.getRangeSet();
  laneSynchronous = operandSet.isEqual(resultSet);
  // Remove bound for a lane.
  operandRel.removeEquality(operandRel.getNumEqualities() - 1);
  resultRel.removeEquality(resultRel.getNumEqualities() - 1);
  // Add bounds for a warp.
  operandRel.addBound(BoundType::LB, 0, 0);
  operandRel.addBound(BoundType::UB, 0, 32);
  resultRel.addBound(BoundType::LB, 0, 0);
  resultRel.addBound(BoundType::UB, 0, 32);
  operandSet = operandRel.getRangeSet();
  resultSet = resultRel.getRangeSet();
  warpSynchronous = operandSet.isEqual(resultSet);
  if (laneSynchronous || warpSynchronous) {
    scratchSize = 0;
  } else {
    // TODO
  }
}
