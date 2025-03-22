//===- BreakPhiStructNodes.cpp -----------------------------*- tablegen -*-===//
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

#include "kapy/Target/LLVMIR/Passes.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/Instructions.h"

using namespace llvm;

static bool processPhiStruct(PHINode *phiNode) {
  auto *structType = dyn_cast<StructType>(phiNode->getType());
  if (!structType)
    return false;
  IRBuilder builder(phiNode);
  auto numOperands = phiNode->getNumIncomingValues();
  auto numElements = structType->getNumElements();
  Value *newStruct = UndefValue::get(structType);
  builder.SetInsertPoint(phiNode->getParent()->getFirstNonPHI());
  auto ip = builder.saveIP();
  for (unsigned i = 0; i < numElements; ++i) {
    builder.SetInsertPoint(phiNode);
    auto *newPhiNode =
        builder.CreatePHI(structType->getElementType(i), numOperands);
    for (unsigned j = 0; j < numOperands; ++j) {
      auto *operand = phiNode->getIncomingValue(j);
      builder.SetInsertPoint(phiNode->getIncomingBlock(j)->getTerminator());
      newPhiNode->addIncoming(builder.CreateExtractValue(operand, i),
                              phiNode->getIncomingBlock(j));
    }
    builder.restoreIP(ip);
    newStruct = builder.CreateInsertValue(newStruct, newPhiNode, i);
    ip = builder.saveIP();
  }
  phiNode->replaceAllUsesWith(newStruct);
  return true;
}

static bool runOnFunction(Function &func) {
  bool changed = false;
  SmallVector<PHINode *> phiNodes;
  for (auto &block : func) {
    for (auto &inst : block) {
      if (auto *phiNode = dyn_cast<PHINode>(&inst)) {
        changed |= processPhiStruct(phiNode);
        continue;
      }
      break;
    }
  }
  return changed;
}

PreservedAnalyses BreakStructPhiNodesPass::run(Function &func,
                                               FunctionAnalysisManager &fam) {
  bool changed = runOnFunction(func);
  return changed ? PreservedAnalyses::none() : PreservedAnalyses::all();
}
