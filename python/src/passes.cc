//===- passes.cc ------------------------------------------------*- C++ -*-===//
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

#include "mlir/Transforms/Passes.h"
#include "kapy/Conversion/KapyToKgpu/Passes.h"
#include "kapy/Conversion/KgpuToLLVM/Passes.h"
#include "kapy/Dialect/Kapy/Transforms/Passes.h"
#include "kapy/Dialect/Kgpu/Transforms/Passes.h"
#include "kapy/Target/LLVMIR/Passes.h"
#include "mlir/Conversion/ArithToLLVM/ArithToLLVM.h"
#include "mlir/Conversion/MathToLLVM/MathToLLVM.h"
#include "mlir/Conversion/SCFToControlFlow/SCFToControlFlow.h"
#include "mlir/Dialect/Arith/Transforms/Passes.h"
#include "mlir/Dialect/Math/Transforms/Passes.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Target/LLVMIR/ModuleTranslation.h"
#include "pybind11/pybind11.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/LegacyPassManager.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/PassManager.h"
#include "llvm/IR/Verifier.h"
#include "llvm/IRReader/IRReader.h"
#include "llvm/Linker/Linker.h"
#include "llvm/MC/TargetRegistry.h"
#include "llvm/Pass.h"
#include "llvm/Passes/OptimizationLevel.h"
#include "llvm/Passes/PassBuilder.h"
#include "llvm/Passes/StandardInstrumentations.h"
#include "llvm/Support/CodeGen.h"
#include "llvm/Support/TargetSelect.h"
#include "llvm/Target/TargetMachine.h"
#include "llvm/Transforms/IPO/AlwaysInliner.h"
#include "llvm/Transforms/InstCombine/InstCombine.h"

namespace py = pybind11;
using namespace llvm;
using namespace mlir;
using namespace mlir::kapy;

void init_kapy_passes(py::module &&m) {
  py::class_<mlir::PassManager>(m, "PassManager", py::module_local())
      .def(py::init<MLIRContext *>())
      .def("run", [](mlir::PassManager &self, ModuleOp &module) {
        if (failed(self.run(module.getOperation())))
          throw std::runtime_error("pass manager run failed");
      });

  m.def("add_inliner",
        [](mlir::PassManager &pm) { pm.addPass(createInlinerPass()); });
  m.def("add_sccp",
        [](mlir::PassManager &pm) { pm.addPass(createSCCPPass()); });
  m.def("add_canonicalizer",
        [](mlir::PassManager &pm) { pm.addPass(createCanonicalizerPass()); });
  m.def("add_cse", [](mlir::PassManager &pm) { pm.addPass(createCSEPass()); });
  m.def("add_symbol_dce",
        [](mlir::PassManager &pm) { pm.addPass(createSymbolDCEPass()); });
  m.def("add_licm", [](mlir::PassManager &pm) {
    pm.addPass(createLoopInvariantCodeMotionPass());
  });
  m.def("add_unsigned_when_equivalent", [](mlir::PassManager &pm) {
    pm.addPass(arith::createArithUnsignedWhenEquivalentPass());
  });
  m.def("add_int_range_optimizations", [](mlir::PassManager &pm) {
    pm.addPass(arith::createIntRangeOptimizationsPass());
  });
  m.def("add_uplift_to_fma", [](mlir::PassManager &pm) {
    pm.addPass(math::createMathUpliftToFMA());
  });

  m.def("add_combine",
        [](mlir::PassManager &pm) { pm.addPass(createKapyCombinePass()); });
  m.def("add_analyze",
        [](mlir::PassManager &pm) { pm.addPass(createKapyAnalyzePass()); });

  m.def("add_convert_kapy_to_kgpu", [](mlir::PassManager &pm) {
    return pm.addPass(createConvertKapyToKgpuPass());
  });

  m.def("add_coalesce_access", [](mlir::PassManager &pm) {
    pm.addPass(createKgpuCoalesceAccessPass());
  });
  m.def("add_optimize_layout", [](mlir::PassManager &pm) {
    pm.addPass(createKgpuOptimizeLayoutPass());
  });

  m.def("add_convert_scf_to_cf",
        [](mlir::PassManager &pm) { pm.addPass(createConvertSCFToCFPass()); });
  m.def("add_convert_kgpu_to_llvm", [](mlir::PassManager &pm) {
    pm.addPass(createConvertKgpuToLLVMPass());
  });
  m.def("add_convert_arith_to_llvm", [](mlir::PassManager &pm) {
    pm.addPass(createArithToLLVMConversionPass());
  });
  m.def("add_convert_math_to_llvm", [](mlir::PassManager &pm) {
    pm.addPass(createConvertMathToLLVMPass());
  });

  m.def("add_llvm_di_scope",
        [](mlir::PassManager &pm) { pm.addPass(createLLVMDIScopePass()); });

  m.def("init_targets", []() {
    static std::once_flag flag;
    std::call_once(flag, []() {
      InitializeAllTargetInfos();
      InitializeAllTargets();
      InitializeAllTargetMCs();
      InitializeAllAsmParsers();
      InitializeAllAsmPrinters();
    });
  });

  m.def("translate_mlir_to_llvm", [](ModuleOp module, LLVMContext &context) {
    return translateModuleToLLVMIR(module, context);
  });

  m.def("set_nvvm_reflect_ftz", [](llvm::Module *module) {
    auto &context = module->getContext();
    auto *i32Type = llvm::Type::getInt32Ty(context);
    auto *mdFour = ConstantAsMetadata::get(ConstantInt::getSigned(i32Type, 4));
    auto *mdName = MDString::get(context, "nvvm-reflect-ftz");
    auto *mdOne = ConstantAsMetadata::get(ConstantInt::getSigned(i32Type, 1));
    auto *reflect = MDNode::get(context, {mdFour, mdName, mdOne});
    module->addModuleFlag(reflect);
  });

  m.def("translate_llvm_to_ptx", [](llvm::Module *module,
                                    std::string &targetTriple,
                                    std::string &processor,
                                    std::string &features,
                                    std::vector<std::string> &flags,
                                    std::vector<std::string> &paths) {
    std::string error;
    auto *target = llvm::TargetRegistry::lookupTarget(targetTriple, error);

    llvm::TargetOptions targetOptions;
    targetOptions.AllowFPOpFusion = llvm::FPOpFusion::Fast;
    targetOptions.UnsafeFPMath = false;
    targetOptions.NoInfsFPMath = false;
    targetOptions.NoNaNsFPMath = false;
    targetOptions.TrapUnreachable = true;

    std::unique_ptr<llvm::TargetMachine> targetMachine{
        target->createTargetMachine(
            targetTriple, processor, features, targetOptions, llvm::Reloc::PIC_,
            std::nullopt, llvm::CodeGenOptLevel::Aggressive)};
    module->setTargetTriple(targetTriple);
    module->setDataLayout(targetMachine->createDataLayout());

    if (!paths.empty()) {
      llvm::Linker linker(*module);
      for (const auto &path : paths) {
        llvm::SMDiagnostic error;
        auto libModule = llvm::parseIRFile(path, error, module->getContext());
        if (!libModule) {
          std::string message = "Failed to parse library at " + path;
          throw std::invalid_argument(message);
        }
        libModule->setTargetTriple(module->getTargetTriple());
        libModule->setDataLayout(module->getDataLayout());

        std::unordered_set<std::string> funcNames;
        for (auto &func : libModule->functions())
          if (!func.isDeclaration())
            funcNames.insert(func.getName().str());

        if (linker.linkInModule(std::move(libModule),
                                llvm::Linker::Flags::LinkOnlyNeeded)) {
          std::string message = "Failed to link library at " + path;
          throw std::invalid_argument(message);
        }

        // Mark linked-in functions as internal because backends use external
        // linkage as a signifier of kernel functions.
        for (auto &func : module->functions())
          if (funcNames.count(func.getName().str()))
            func.setLinkage(llvm::GlobalValue::InternalLinkage);
      }
    }

    llvm::LoopAnalysisManager lam;
    llvm::FunctionAnalysisManager fam;
    llvm::CGSCCAnalysisManager cgam;
    llvm::ModuleAnalysisManager mam;

    PipelineTuningOptions tuningOptions;
    tuningOptions.LoopUnrolling = true;
    tuningOptions.LoopInterleaving = true;
    tuningOptions.LoopVectorization = true;
    tuningOptions.SLPVectorization = true;

    PassBuilder pb(targetMachine.get(), tuningOptions, std::nullopt, nullptr);
    pb.registerModuleAnalyses(mam);
    pb.registerCGSCCAnalyses(cgam);
    pb.registerFunctionAnalyses(fam);
    pb.registerLoopAnalyses(lam);
    pb.crossRegisterProxies(lam, fam, cgam, mam);

    llvm::ModulePassManager mpm;
    pb.registerVectorizerStartEPCallback(
        [&](llvm::FunctionPassManager &fpm, llvm::OptimizationLevel) {
          fpm.addPass(BreakStructPhiNodesPass());
          fpm.addPass(InstCombinePass());
        });
    mpm.addPass(pb.buildPerModuleDefaultPipeline(llvm::OptimizationLevel::O3));
    mpm.run(*module, mam);

    auto options = cl::getRegisteredOptions();
    for (auto flag : flags) {
      auto *option = static_cast<cl::opt<bool> *>(options[flag]);
      assert(option);
      option->setValue(true);
    }

    for (auto &func : module->functions())
      if (!func.hasFnAttribute(llvm::Attribute::NoInline))
        func.addFnAttr(llvm::Attribute::AlwaysInline);

    llvm::legacy::PassManager lpm;
    lpm.add(createAlwaysInlinerLegacyPass());
    lpm.add(createVerifierPass());
    lpm.run(*module);

    std::string ptx;
    {
      llvm::raw_string_ostream ss(ptx);
      llvm::buffer_ostream bs(ss);
      llvm::legacy::PassManager lpm;
      auto fileType = llvm::CodeGenFileType::AssemblyFile;
      targetMachine->addPassesToEmitFile(lpm, bs, nullptr, fileType);
      lpm.run(*module);
    }
    return ptx;
  });
}
