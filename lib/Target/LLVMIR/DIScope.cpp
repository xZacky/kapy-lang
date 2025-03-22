//===- DIScope.h -------------------------------------------*- tablegen -*-===//
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
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "llvm/BinaryFormat/Dwarf.h"
#include "llvm/Support/Path.h"

using namespace mlir;

static FileLineColLoc extractFileLoc(Location loc) {
  if (auto fileLoc = dyn_cast<FileLineColLoc>(loc))
    return fileLoc;
  if (auto nameLoc = dyn_cast<NameLoc>(loc))
    return extractFileLoc(nameLoc.getChildLoc());
  if (auto opaqueLoc = dyn_cast<OpaqueLoc>(loc))
    return extractFileLoc(opaqueLoc.getFallbackLocation());
  if (auto fusedLoc = dyn_cast<FusedLoc>(loc))
    return extractFileLoc(fusedLoc.getLocations().front());
  if (auto callerLoc = dyn_cast<CallSiteLoc>(loc))
    return extractFileLoc(callerLoc.getCaller());
  auto unknownFile = StringAttr::get(loc.getContext(), "<unknown>");
  return FileLineColLoc::get(unknownFile, 0, 0);
}

namespace {

#define GEN_PASS_DEF_LLVMDISCOPE
#include "kapy/Target/LLVMIR/Passes.h.inc"

class LLVMDIScopePass : public impl::LLVMDIScopeBase<LLVMDIScopePass> {
public:
  void setSubprogramAttr(LLVM::LLVMFuncOp funcOp) {
    auto loc = funcOp.getLoc();
    if (loc->findInstanceOf<FusedLocWith<LLVM::DISubprogramAttr>>())
      return;
    auto *context = &getContext();
    LLVM::DICompileUnitAttr compileUnitAttr;
    if (auto module = funcOp->getParentOfType<ModuleOp>()) {
      auto loc = module.getLoc();
      auto fusedCompileUnitAttr =
          loc->findInstanceOf<FusedLocWith<LLVM::DICompileUnitAttr>>();
      if (fusedCompileUnitAttr)
        compileUnitAttr = fusedCompileUnitAttr.getMetadata();
    }

    LLVM::DIFileAttr fileAttr;
    unsigned line = 1;
    unsigned column = 1;
    auto fileLoc = extractFileLoc(loc);
    if (!fileLoc && compileUnitAttr) {
      fileAttr = compileUnitAttr.getFile();
    } else if (!fileLoc) {
      fileAttr = LLVM::DIFileAttr::get(context, "<unknown>", "");
    } else {
      line = fileLoc.getLine();
      column = fileLoc.getColumn();
      auto inputFilePath = fileLoc.getFilename().getValue();
      fileAttr = LLVM::DIFileAttr::get(
          context, llvm::sys::path::filename(inputFilePath),
          llvm::sys::path::parent_path(inputFilePath));
    }
    auto subroutineTypeAttr =
        LLVM::DISubroutineTypeAttr::get(context, llvm::dwarf::DW_CC_normal, {});
    DistinctAttr distinctId;
    auto subprogramFlags = LLVM::DISubprogramFlags::Optimized;
    if (!funcOp.isExternal()) {
      distinctId = DistinctAttr::create(UnitAttr::get(context));
      if (!compileUnitAttr)
        compileUnitAttr = LLVM::DICompileUnitAttr::get(
            distinctId, llvm::dwarf::DW_LANG_C, fileAttr,
            StringAttr::get(context, "kapy"), true,
            LLVM::DIEmissionKind::LineTablesOnly);
      subprogramFlags = subprogramFlags | LLVM::DISubprogramFlags::Definition;
    } else {
      compileUnitAttr = {};
    }

    auto funcNameAttr = funcOp.getNameAttr();
    auto subprogramAttr = LLVM::DISubprogramAttr::get(
        context, distinctId, compileUnitAttr, fileAttr, funcNameAttr,
        funcNameAttr, fileAttr, line, line, subprogramFlags,
        subroutineTypeAttr);
    funcOp->setLoc(FusedLoc::get(context, {loc}, subprogramAttr));
  }

  Location getNestedLoc(Operation *op, LLVM::DIScopeAttr scopeAttr,
                        Location calleeLoc) {
    auto calleeFileName = extractFileLoc(calleeLoc).getFilename();
    auto *context = op->getContext();
    auto calleeFileAttr = LLVM::DIFileAttr::get(
        context, llvm::sys::path::filename(calleeFileName),
        llvm::sys::path::parent_path(calleeFileName));
    auto lexicalBlockFileAttr = LLVM::DILexicalBlockFileAttr::get(
        context, scopeAttr, calleeFileAttr, 0);
    auto loc = calleeLoc;
    if (isa<CallSiteLoc>(calleeLoc)) {
      auto nestedLoc = cast<CallSiteLoc>(calleeLoc).getCallee();
      loc = getNestedLoc(op, lexicalBlockFileAttr, nestedLoc);
    }
    return FusedLoc::get(context, {loc}, lexicalBlockFileAttr);
  }

  void setLexicalBlockFileAttr(Operation *op) {
    auto oldLoc = op->getLoc();
    if (auto callSiteLoc = dyn_cast<CallSiteLoc>(oldLoc)) {
      auto callerLoc = callSiteLoc.getCaller();
      auto calleeLoc = callSiteLoc.getCallee();
      LLVM::DIScopeAttr scopeAttr;
      auto funcOp = op->getParentOfType<LLVM::LLVMFuncOp>();
      auto funcLoc = cast<FusedLoc>(funcOp.getLoc());
      scopeAttr = cast<LLVM::DISubprogramAttr>(funcLoc.getMetadata());
      auto newLoc =
          CallSiteLoc::get(getNestedLoc(op, scopeAttr, calleeLoc), callerLoc);
      op->setLoc(newLoc);
    }
  }

  virtual void runOnOperation() override {
    auto module = getOperation();
    module.walk<WalkOrder::PreOrder>([&](Operation *op) {
      if (isa<LLVM::LLVMFuncOp>(op))
        setSubprogramAttr(cast<LLVM::LLVMFuncOp>(op));
      else
        setLexicalBlockFileAttr(op);
    });
  }
};

} // namespace

std::unique_ptr<Pass> mlir::createLLVMDIScopePass() {
  return std::make_unique<LLVMDIScopePass>();
}
