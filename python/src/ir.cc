//===- ir.cc ----------------------------------------------------*- C++ -*-===//
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

#include "kapy/Dialect/Kapy/IR/Kapy.h"
#include "kapy/Dialect/Kgpu/IR/Kgpu.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/LLVMIR/NVVMDialect.h"
#include "mlir/IR/Verifier.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Target/LLVMIR/Dialect/Builtin/BuiltinToLLVMIRTranslation.h"
#include "mlir/Target/LLVMIR/Dialect/LLVMIR/LLVMToLLVMIRTranslation.h"
#include "mlir/Target/LLVMIR/Dialect/NVVM/NVVMToLLVMIRTranslation.h"
#include "mlir/Transforms/LocationSnapshot.h"
#include "pybind11/pybind11.h"

namespace {

namespace py = pybind11;
using namespace mlir;
using namespace mlir::kapy;

/// A custom operation builder that keeps track of the last location.
class KapyOpBuilder {
public:
  KapyOpBuilder(MLIRContext *context) {
    builder = std::make_unique<OpBuilder>(context);
    lastLoc = std::make_unique<Location>(builder->getUnknownLoc());
  }

  OpBuilder &getBuilder() { return *builder; }

  MLIRContext *getContext() { return builder->getContext(); }

  void setLastLoc(Location loc) { lastLoc = std::make_unique<Location>(loc); }

  void setLastLoc(const std::string &fileName, unsigned line, unsigned column) {
    setLastLoc(FileLineColLoc::get(getContext(), fileName, line, column));
  }

  Location getLastLoc() { return *lastLoc; }

  void setInsertionPointToStart(Block &block) {
    if (!block.empty())
      setLastLoc(block.begin()->getLoc());
    else
      setLastLoc(builder->getUnknownLoc());
    builder->setInsertionPointToStart(&block);
  }

  void setInsertionPointToEnd(Block &block) {
    if (!block.empty())
      setLastLoc(block.back().getLoc());
    else
      setLastLoc(builder->getUnknownLoc());
    builder->setInsertionPointToEnd(&block);
  }

  void setInsertionPointAfter(Operation &op) {
    setLastLoc(op.getLoc());
    builder->setInsertionPointAfter(&op);
  }

  void restoreInsertionPoint(OpBuilder::InsertPoint ip) {
    if (ip.isSet() && ip.getPoint() != ip.getBlock()->end())
      setLastLoc(ip.getPoint()->getLoc());
    else
      setLastLoc(builder->getUnknownLoc());
    builder->restoreInsertionPoint(ip);
  }

  template <typename OpT, typename... Ts> OpT create(Ts &&...args) {
    return builder->create<OpT>(getLastLoc(), std::forward<Ts>(args)...);
  }

private:
  std::unique_ptr<OpBuilder> builder;
  std::unique_ptr<Location> lastLoc;
};
} // namespace

static std::string locationToString(Location loc) {
  std::string str;
  llvm::raw_string_ostream os(str);
  loc.print(os);
  os.flush();
  return str;
}

static void outputWarning(Location loc, const std::string &msg) {
  std::string locStr = locationToString(loc);
  PyErr_WarnEx(PyExc_UserWarning, (locStr + ": " + msg).c_str(), 2);
}

void init_kapy_ir(py::module &&m) {
  using return_policy = py::return_value_policy;

  py::enum_<PaddingOption>(m, "PaddingOption", py::module_local())
      .value("ZERO", PaddingOption::ZERO)
      .value("QNAN", PaddingOption::QNAN)
      .value("PINF", PaddingOption::PINF)
      .value("NINF", PaddingOption::NINF);

  py::enum_<CacheModifier>(m, "CacheModifier", py::module_local())
      .value("NONE", CacheModifier::NONE)
      .value("CA", CacheModifier::CA)
      .value("CG", CacheModifier::CG)
      .value("CS", CacheModifier::CS)
      .value("WB", CacheModifier::WB)
      .value("WT", CacheModifier::WT)
      .value("LU", CacheModifier::LU)
      .value("CV", CacheModifier::CV);

  py::enum_<EvictPriority>(m, "EvictPriority", py::module_local())
      .value("EVICT_NORMAL", EvictPriority::EVICT_NORMAL)
      .value("EVICT_FIRST", EvictPriority::EVICT_FIRST)
      .value("EVICT_LAST", EvictPriority::EVICT_LAST)
      .value("EVICT_UNCHANGED", EvictPriority::EVICT_UNCHANGED)
      .value("NO_ALLOCATE", EvictPriority::NO_ALLOCATE);

  py::enum_<MatmulImplWay>(m, "MatmulImplWay", py::module_local())
      .value("MMA_M16N8K8_F16", MatmulImplWay::MMA_M16N8K8_F16)
      .value("MMA_M16N8K16_F16", MatmulImplWay::MMA_M16N8K16_F16)
      .value("MMA_M16N8K8_TF32", MatmulImplWay::MMA_M16N8K8_TF32)
      .value("MMA_M16N8K16_F8", MatmulImplWay::MMA_M16N8K16_F8);

  py::enum_<RoundingMode>(m, "RoundingMode", py::module_local())
      .value("RZ", RoundingMode::RZ)
      .value("RN", RoundingMode::RN);

  py::class_<MLIRContext>(m, "MLIRContext", py::module_local())
      .def(py::init<>());

  m.def("load_dialects", [](MLIRContext &context) {
    DialectRegistry registry;
    registry.insert<kapy::KapyDialect, kapy::KgpuDialect, //
                    LLVM::LLVMDialect, NVVM::NVVMDialect>();
    registerBuiltinDialectTranslation(registry);
    registerLLVMDialectTranslation(registry);
    registerNVVMDialectTranslation(registry);
    context.appendDialectRegistry(registry);
    context.loadAllAvailableDialects();
  });

  py::class_<Type>(m, "Type", py::module_local())
      .def("is_int", [](Type &self,
                        unsigned bitWidth) { return self.isInteger(bitWidth); })
      .def("is_float8e4m3", &Type::isFloat8E4M3)
      .def("is_float8e5m2", &Type::isFloat8E5M2)
      .def("is_bfloat16", &Type::isBF16)
      .def("is_float16", &Type::isF16)
      .def("is_float32", &Type::isF32)
      .def("is_float64", &Type::isF64)
      .def("__str__", [](Type &self) {
        std::string str;
        llvm::raw_string_ostream os(str);
        self.print(os);
        return os.str();
      });

  py::class_<FunctionType>(m, "FunctionType", py::module_local())
      .def("get_param_types", [](FunctionType &self) {
        return std::vector<Type>(self.getInputs().begin(),
                                 self.getInputs().end());
      });

  py::class_<Location>(m, "Location", py::module_local())
      .def("__str__", [](Location &self) {
        std::string str;
        llvm::raw_string_ostream os(str);
        self.print(os);
        return os.str();
      });

  py::class_<Value>(m, "Value", py::module_local())
      .def("set_attr",
           [](Value &self, std::string &name, Attribute &attr) {
             if (auto *defOp = self.getDefiningOp()) {
               defOp->setAttr(name, attr);
             } else {
               auto blockArg = cast<BlockArgument>(self);
               auto argIndex = blockArg.getArgNumber();
               auto attrName = name + "_arg" + std::to_string(argIndex);
               auto *block = blockArg.getOwner();
               if (block->isEntryBlock() && !isa<FuncOp>(block->getParentOp()))
                 block->getParentOp()->setAttr(attrName, attr);
             }
           })
      .def("get_context", &Value::getContext)
      .def("replace_all_uses_with",
           [](Value &self, Value &newValue) {
             self.replaceAllUsesWith(newValue);
           })
      .def("get_type", &Value::getType)
      .def_property_readonly(
          "id", [](Value &self) { return (uint64_t)self.getImpl(); });

  py::class_<OpResult, Value>(m, "OpResult", py::module_local());

  py::class_<BlockArgument, Value>(m, "BlockArgument", py::module_local());

  py::class_<Region>(m, "Region", py::module_local())
      .def("get_parent_region", &Region::getParentRegion,
           return_policy::reference)
      .def("size", [](Region &self) { return self.getBlocks().size(); })
      .def("empty", &Region::empty)
      .def_property_readonly("id",
                             [](Region &self) { return (uint64_t)&self; });

  py::class_<Block>(m, "Block", py::module_local())
      .def("get_argument",
           [](Block &self, unsigned index) {
             if (index >= self.getNumArguments())
               throw py::index_error("block argument index out of range");
             return self.getArgument(index);
           })
      .def("add_argument",
           [](Block &self, Type &type) {
             self.addArgument(type, UnknownLoc::get(type.getContext()));
           })
      .def("get_num_arguments", &Block::getNumArguments)
      .def("dump", &Block::dump)
      .def("move_before",
           [](Block &self, Block &other) { self.moveBefore(&other); })
      .def("insert_before", &Block::insertBefore)
      .def("get_parent", &Block::getParent, return_policy::reference)
      .def("merge_block_before",
           [](Block &self, Block &other) {
             if (self.getNumArguments() != 0)
               throw std::runtime_error(
                   "this block has argument, can not merge");
             other.getOperations().splice(other.begin(), self.getOperations());
             self.dropAllUses();
             self.erase();
           })
      .def("replace_use_in_block_with",
           [](Block &self, Value &oldValue, Value &newValue) {
             oldValue.replaceUsesWithIf(newValue, [&](OpOperand &operand) {
               auto *useOp = operand.getOwner();
               auto *block = useOp->getBlock();
               while (block) {
                 if (block == &self)
                   return true;
                 // Move up one level.
                 block = block->getParent()->getParentOp()->getBlock();
               }
               return false;
             });
           })
      .def("__str__",
           [](Block &self) {
             std::string str;
             llvm::raw_string_ostream os(str);
             self.print(os);
             return str;
           })
      .def("has_terminator",
           [](Block &self) {
             return !self.empty() &&
                    self.back().hasTrait<OpTrait::IsTerminator>();
           })
      .def("has_return",
           [](Block &self) {
             return !self.empty() &&
                    self.back().hasTrait<OpTrait::ReturnLike>();
           })
      .def("erase", [](Block &self) { self.erase(); })
      .def_property_readonly("id", [](Block &self) { return (uint64_t)&self; });

  py::class_<Attribute>(m, "Attribute", py::module_local());

  py::class_<IntegerAttr, Attribute>(m, "IntegerAttr", py::module_local());

  py::class_<BoolAttr, Attribute>(m, "BoolAttr", py::module_local());

  py::class_<OpState>(m, "OpState", py::module_local())
      .def("set_attr", [](OpState &self, std::string &name,
                          Attribute &attr) { self->setAttr(name, attr); })
      .def("get_num_results",
           [](OpState &self) { return self->getNumResults(); })
      .def("get_num_regions",
           [](OpState &self) { return self->getNumRegions(); })
      .def("get_result",
           [](OpState &self, unsigned index) {
             if (index >= self->getNumResults())
               throw py::index_error("operation result index out of range");
             return self->getResult(index);
           })
      .def("get_region",
           [](OpState &self, unsigned index) -> Region & {
             if (index >= self->getNumRegions())
               throw py::index_error("operation region index out of range");
             return self->getRegion(index);
           })
      .def("dump", [](OpState &self) { self->dump(); })
      .def("__str__",
           [](OpState &self) {
             std::string str;
             llvm::raw_string_ostream os(str);
             auto flags = OpPrintingFlags();
             flags.enableDebugInfo();
             self->print(os, flags);
             return str;
           })
      .def("append_operand",
           [](OpState &self, Value &operand) {
             self->insertOperands(self->getNumOperands(), operand);
           })
      .def("verify", [](OpState &self) {
        return succeeded(verify(self.getOperation()));
      });

  py::class_<scf::ForOp, OpState>(m, "ForOp", py::module_local())
      .def(
          "get_body",
          [](scf::ForOp &self) -> Block * { return self.getBody(); },
          return_policy::reference)
      .def("get_induction_var", &scf::ForOp::getInductionVar);

  py::class_<scf::IfOp, OpState>(m, "IfOp", py::module_local())
      .def("get_then_block", &scf::IfOp::thenBlock, return_policy::reference)
      .def("get_else_block", &scf::IfOp::elseBlock, return_policy::reference)
      .def("get_then_yield", &scf::IfOp::thenYield)
      .def("get_else_yield", &scf::IfOp::elseYield);

  py::class_<scf::WhileOp, OpState>(m, "WhileOp", py::module_local())
      .def("get_before", &scf::WhileOp::getBefore, return_policy::reference)
      .def("get_after", &scf::WhileOp::getAfter, return_policy::reference);

  py::class_<Operation, std::unique_ptr<Operation, py::nodelete>>(
      m, "Operation", py::module_local())
      .def("get_name",
           [](Operation &self) { return self.getName().getStringRef().str(); })
      .def("get_num_operands", &Operation::getNumOperands)
      .def("get_operand", &Operation::getOperand)
      .def("get_num_results", &Operation::getNumResults)
      .def("get_result", &Operation::getResult)
      .def("get_num_regions", &Operation::getNumRegions)
      .def("get_region", &Operation::getRegion, return_policy::reference)
      .def("get_block", &Operation::getBlock, return_policy::reference)
      .def("get_str_attr",
           [](Operation &self, std::string &name) -> py::object {
             auto strAttr = self.getAttrOfType<StringAttr>(name);
             if (!strAttr)
               return py::none();
             return py::str(strAttr.getValue().str());
           })
      .def("get_flat_symbol_ref_attr",
           [](Operation &self, std::string &name) -> py::object {
             auto symbolAttr = self.getAttrOfType<FlatSymbolRefAttr>(name);
             if (!symbolAttr)
               return py::none();
             return py::str(symbolAttr.getValue().str());
           });

  // dynamic_attr is used to transfer onwership of the mlir context to the
  // module.
  py::class_<ModuleOp, OpState>(m, "ModuleOp", py::module_local(),
                                py::dynamic_attr())
      .def("dump", &ModuleOp::dump)
      .def("__str__",
           [](ModuleOp &self) {
             std::string str;
             llvm::raw_string_ostream os(str);
             auto flags = OpPrintingFlags();
             flags.enableDebugInfo();
             self.print(os, flags);
             return str;
           })
      .def("push_back",
           [](ModuleOp &self, FuncOp &funcOp) { self.push_back(funcOp); })
      .def("has_function",
           [](ModuleOp &self, std::string &funcName) {
             if (self.lookupSymbol(funcName))
               return true;
             return false;
           })
      .def("get_function",
           [](ModuleOp &self, std::string &funcName) {
             return self.lookupSymbol<FuncOp>(funcName);
           })
      .def("get_int_attr",
           [](ModuleOp &self, std::string &name) -> py::object {
             auto intAttr = self->getAttrOfType<IntegerAttr>(name);
             if (!intAttr)
               return py::none();
             return py::int_(intAttr.getInt());
           })
      .def("create_location_snapshot",
           [](ModuleOp &self, std::string &fileName) {
             generateLocationsFromIR(llvm::nulls(), fileName, self,
                                     std::nullopt);
           })
      .def("walk",
           [](ModuleOp &self, std::function<void(Operation *)> &callback) {
             self.walk(callback);
           });

  m.def(
      "parse_mlir_module",
      [](std::string &fileName, MLIRContext &context) {
        auto module = parseSourceFile<ModuleOp>(fileName, &context);
        if (!module)
          throw std::runtime_error("parse mlir module file failed.");
        return module->clone();
      },
      return_policy::take_ownership);

  py::class_<FuncOp, OpState>(m, "FuncOp", py::module_local())
      .def("get_argument",
           [](FuncOp &self, unsigned index) {
             if (index >= self.getNumArguments())
               throw py::index_error("function argument index out of range");
             return self.getArgument(index);
           })
      .def(
          "add_entry_block",
          [](FuncOp &self) -> Block * { return self.addEntryBlock(); },
          return_policy::reference)
      .def("set_arg_attr",
           [](FuncOp &self, unsigned index, std::string &name, int64_t value) {
             auto i64Type = IntegerType::get(self.getContext(), 64);
             self.setArgAttr(index, name, IntegerAttr::get(i64Type, value));
           })
      .def("finalize",
           [](FuncOp &self) {
             // Remove unreachable code after return.
             self.walk([&](Block *block) {
               Operation *returnOp = nullptr;
               // It's better to not use walk here because we only want to check
               // operations in the current block.
               for (auto &op : block->getOperations()) {
                 if (isa<ReturnOp>(op)) {
                   if (!returnOp) {
                     returnOp = &op;
                     break;
                   }
                 }
               }
               if (returnOp && returnOp != &block->back()) {
                 auto it = returnOp->getIterator();
                 it++;
                 auto *deadBlock = block->splitBlock(it);
                 deadBlock->erase();
               }
             });
           })
      .def_property_readonly("function_type", &FuncOp::getFunctionType)
      .def("set_type", &FuncOp::setType);

  py::class_<OpBuilder::InsertPoint>(m, "InsertPoint", py::module_local());

  py::class_<KapyOpBuilder>(m, "Builder", py::module_local(),
                            py::dynamic_attr())
      .def(py::init<MLIRContext *>())
      .def("create_module",
           [](KapyOpBuilder &self) { return self.create<ModuleOp>(); })
      .def("set_insertion_point_to_start",
           [](KapyOpBuilder &self, Block &block) {
             self.setInsertionPointToStart(block);
           })
      .def("set_insertion_point_to_end",
           [](KapyOpBuilder &self, Block &block) {
             self.setInsertionPointToEnd(block);
           })
      .def("set_insertion_point_after",
           [](KapyOpBuilder &self, Operation &op) {
             self.setInsertionPointAfter(op);
           })
      .def(
          "get_insertion_block",
          [](KapyOpBuilder &self) -> Block * {
            return self.getBuilder().getInsertionBlock();
          },
          return_policy::reference)
      .def("save_insertion_point",
           [](KapyOpBuilder &self) {
             return self.getBuilder().saveInsertionPoint();
           })
      .def("restore_insertion_point",
           [](KapyOpBuilder &self, OpBuilder::InsertPoint ip) {
             self.restoreInsertionPoint(ip);
           })
      .def("get_bool_attr",
           [](KapyOpBuilder &self, bool value) {
             return self.getBuilder().getBoolAttr(value);
           })
      .def("get_int64_attr",
           [](KapyOpBuilder &self, int64_t value) {
             return self.getBuilder().getI64IntegerAttr(value);
           })
      .def("create_constant_int1",
           [](KapyOpBuilder &self, bool value) -> Value {
             return self.create<arith::ConstantIntOp>(
                 value, self.getBuilder().getI1Type());
           })
      .def("create_constant_int8",
           [](KapyOpBuilder &self, int64_t value) -> Value {
             return self.create<arith::ConstantIntOp>(
                 value, self.getBuilder().getI8Type());
           })
      .def("create_constant_int16",
           [](KapyOpBuilder &self, int64_t value) -> Value {
             return self.create<arith::ConstantIntOp>(
                 value, self.getBuilder().getI16Type());
           })
      .def("create_constant_int32",
           [](KapyOpBuilder &self, int64_t value) -> Value {
             return self.create<arith::ConstantIntOp>(
                 value, self.getBuilder().getI32Type());
           })
      .def("create_constant_int64",
           [](KapyOpBuilder &self, int64_t value) -> Value {
             return self.create<arith::ConstantIntOp>(
                 value, self.getBuilder().getI64Type());
           })
      .def("create_constant_float8e4m3",
           [](KapyOpBuilder &self, float value) -> Value {
             auto f8E4M3Type = self.getBuilder().getFloat8E4M3Type();
             return self.create<arith::ConstantFloatOp>(
                 APFloat(f8E4M3Type.getFloatSemantics(), std::to_string(value)),
                 f8E4M3Type);
           })
      .def("create_constant_float8e5m2",
           [](KapyOpBuilder &self, float value) -> Value {
             auto f8E5M2Type = self.getBuilder().getFloat8E5M2Type();
             return self.create<arith::ConstantFloatOp>(
                 APFloat(f8E5M2Type.getFloatSemantics(), std::to_string(value)),
                 f8E5M2Type);
           })
      .def("create_constant_bfloat16",
           [](KapyOpBuilder &self, float value) -> Value {
             auto bf16Type = self.getBuilder().getBF16Type();
             return self.create<arith::ConstantFloatOp>(
                 APFloat(bf16Type.getFloatSemantics(), std::to_string(value)),
                 bf16Type);
           })
      .def("create_constant_float16",
           [](KapyOpBuilder &self, float value) -> Value {
             return self.create<arith::ConstantOp>(
                 self.getBuilder().getF16FloatAttr(value));
           })
      .def("create_constant_float32",
           [](KapyOpBuilder &self, float value) -> Value {
             return self.create<arith::ConstantOp>(
                 self.getBuilder().getF32FloatAttr(value));
           })
      .def("create_constant_float64",
           [](KapyOpBuilder &self, double value) -> Value {
             return self.create<arith::ConstantOp>(
                 self.getBuilder().getF64FloatAttr(value));
           })
      .def("create_constant_zero",
           [](KapyOpBuilder &self, Type &type) -> Value {
             if (auto floatType = dyn_cast<FloatType>(type))
               return self.create<arith::ConstantFloatOp>(
                   APFloat(floatType.getFloatSemantics(), 0), floatType);
             else if (auto intType = dyn_cast<IntegerType>(type))
               return self.create<arith::ConstantIntOp>(0, intType);
             else
               throw std::runtime_error("not implemented");
           })
      .def("create_constant_all_ones",
           [](KapyOpBuilder &self, Type &type) -> Value {
             uint64_t value = 0xFFFFFFFFFFFFFFFF;
             if (auto intType = dyn_cast<IntegerType>(type))
               return self.create<arith::ConstantIntOp>(value, intType);
             else
               throw std::runtime_error("not implemented");
           })
      .def("get_void_type",
           [](KapyOpBuilder &self) -> Type {
             return self.getBuilder().getNoneType();
           })
      .def("get_int1_type",
           [](KapyOpBuilder &self) -> Type {
             return self.getBuilder().getI1Type();
           })
      .def("get_int8_type",
           [](KapyOpBuilder &self) -> Type {
             return self.getBuilder().getI8Type();
           })
      .def("get_int16_type",
           [](KapyOpBuilder &self) -> Type {
             return self.getBuilder().getI16Type();
           })
      .def("get_int32_type",
           [](KapyOpBuilder &self) -> Type {
             return self.getBuilder().getI32Type();
           })
      .def("get_int64_type",
           [](KapyOpBuilder &self) -> Type {
             return self.getBuilder().getI64Type();
           })
      .def("get_float8e4m3_type",
           [](KapyOpBuilder &self) -> Type {
             return self.getBuilder().getFloat8E4M3Type();
           })
      .def("get_float8e5m2_type",
           [](KapyOpBuilder &self) -> Type {
             return self.getBuilder().getFloat8E5M2Type();
           })
      .def("get_bfloat16_type",
           [](KapyOpBuilder &self) -> Type {
             return self.getBuilder().getBF16Type();
           })
      .def("get_float16_type",
           [](KapyOpBuilder &self) -> Type {
             return self.getBuilder().getF16Type();
           })
      .def("get_float32_type",
           [](KapyOpBuilder &self) -> Type {
             return self.getBuilder().getF32Type();
           })
      .def("get_pointer_type",
           [](KapyOpBuilder &self, unsigned space = 1) -> Type {
             return KapyPointerType::get(self.getContext(), space);
           })
      .def("get_tensor_type",
           [](KapyOpBuilder &self, std::vector<int64_t> &shape,
              Type &elementType, MemorySpace memory) -> Type {
             auto encoding = EncodingAttr::get(self.getContext(), memory);
             return RankedTensorType::get(shape, elementType, encoding);
           })
      .def("get_function_type",
           [](KapyOpBuilder &self, std::vector<Type> &inputs,
              std::vector<Type> &results) -> Type {
             return self.getBuilder().getFunctionType(inputs, results);
           })
      .def("set_loc",
           [](KapyOpBuilder &self, Location loc) { self.setLastLoc(loc); })
      .def("set_loc",
           [](KapyOpBuilder &self, std::string &fileName, unsigned line,
              unsigned column) { self.setLastLoc(fileName, line, column); })
      .def("get_loc", [](KapyOpBuilder &self) { return self.getLastLoc(); })
      .def("get_or_create_function",
           [](KapyOpBuilder &self, ModuleOp &module, std::string &name,
              Type &type, std::string &visibility, bool noinline) -> FuncOp {
             if (auto *funcOp = module.lookupSymbol(name))
               return dyn_cast<FuncOp>(funcOp);
             if (auto funcType = dyn_cast<FunctionType>(type)) {
               SmallVector<NamedAttribute, 2> attrs = {
                   NamedAttribute(
                       self.getBuilder().getStringAttr("sym_visibility"),
                       self.getBuilder().getStringAttr(visibility)),
                   NamedAttribute(self.getBuilder().getStringAttr("noinline"),
                                  self.getBuilder().getBoolAttr(noinline))};
               return self.create<FuncOp>(name, funcType, attrs);
             }
             throw std::invalid_argument("invalid function type");
           })
      .def(
          "create_block",
          [](KapyOpBuilder &self) -> Block * {
            auto *parent = self.getBuilder().getBlock()->getParent();
            return self.getBuilder().createBlock(parent);
          },
          return_policy::reference)
      .def(
          "create_block_with_parent",
          [](KapyOpBuilder &self, Region &parent,
             std::vector<Type> &argTypes) -> Block * {
            // TODO: Update argument location.
            auto loc = self.getBuilder().getUnknownLoc();
            SmallVector<Location, 8> locs(argTypes.size(), loc);
            return self.getBuilder().createBlock(&parent, {}, argTypes, locs);
          },
          return_policy::reference)
      .def(
          "create_block",
          [](KapyOpBuilder &self) -> Block * { return new Block(); },
          return_policy::reference)
      .def("create_return",
           [](KapyOpBuilder &self, std::vector<Value> &operands) -> OpState {
             return self.create<ReturnOp>(operands);
           })
      .def("create_call",
           [](KapyOpBuilder &self, FuncOp &funcOp, std::vector<Value> &operands)
               -> OpState { return self.create<CallOp>(funcOp, operands); })
      .def("create_for",
           [](KapyOpBuilder &self, Value &lb, Value &ub, Value &step,
              std::vector<Value> &operands) -> scf::ForOp {
             return self.create<scf::ForOp>(lb, ub, step, operands);
           })
      .def("create_if",
           [](KapyOpBuilder &self, std::vector<Type> &resultTypes,
              Value &condition, bool withElse) -> scf::IfOp {
             return self.create<scf::IfOp>(resultTypes, condition, withElse);
           })
      .def("create_while",
           [](KapyOpBuilder &self, std::vector<Type> &resultTypes,
              std::vector<Value> operands) -> scf::WhileOp {
             return self.create<scf::WhileOp>(resultTypes, operands);
           })
      .def("create_condition",
           [](KapyOpBuilder &self, Value &condition,
              std::vector<Value> &operands) -> scf::ConditionOp {
             return self.create<scf::ConditionOp>(condition, operands);
           })
      .def("create_yield",
           [](KapyOpBuilder &self, std::vector<Value> &operands)
               -> scf::YieldOp { return self.create<scf::YieldOp>(operands); })
      .def("create_arange",
           [](KapyOpBuilder &self, unsigned axis, int32_t start,
              int32_t end) -> Value {
             if (axis > 1)
               throw py::index_error("program axis must be 0, 1");
             std::array<int64_t, 2> shape;
             if (axis == 0)
               shape = {end - start, 1};
             else
               shape = {1, end - start};
             return self.create<ArangeOp>(
                 RankedTensorType::get(shape, self.getBuilder().getI32Type()),
                 axis, start, end);
           })
      .def("create_fptofp",
           [](KapyOpBuilder &self, Type &resultType, Value &source,
              std::optional<RoundingMode> roundingMode) -> Value {
             if (roundingMode.has_value())
               return self.create<FPToFPOp>(
                   resultType, source,
                   RoundingModeAttr::get(self.getContext(),
                                         roundingMode.value()));
             else
               return self.create<FPToFPOp>(resultType, source);
           })
      .def("create_bitcast",
           [](KapyOpBuilder &self, Type &resultType, Value &source) -> Value {
             return self.create<arith::BitcastOp>(resultType, source);
           })
      .def("create_sitofp",
           [](KapyOpBuilder &self, Type &resultType, Value &source) -> Value {
             return self.create<arith::SIToFPOp>(resultType, source);
           })
      .def("create_fptosi",
           [](KapyOpBuilder &self, Type &resultType, Value &source) -> Value {
             return self.create<arith::FPToSIOp>(resultType, source);
           })
      .def("create_extf",
           [](KapyOpBuilder &self, Type &resultType, Value &source) -> Value {
             return self.create<arith::ExtFOp>(resultType, source);
           })
      .def("create_truncf",
           [](KapyOpBuilder &self, Type &resultType, Value &source) -> Value {
             return self.create<arith::TruncFOp>(resultType, source);
           })
      .def("create_intcast",
           [](KapyOpBuilder &self, Type &resultType, Value &source,
              bool isSigned) -> Value {
             auto oldBitWidth = getIntOrFloatBitWidth(source.getType());
             auto newBitWidth = getIntOrFloatBitWidth(resultType);
             if (oldBitWidth == newBitWidth)
               return self.create<arith::BitcastOp>(resultType, source);
             else if (oldBitWidth > newBitWidth)
               return self.create<arith::TruncIOp>(resultType, source);
             else if (isSigned)
               return self.create<arith::ExtSIOp>(resultType, source);
             else
               return self.create<arith::ExtUIOp>(resultType, source);
           })
      .def("create_mulf",
           [](KapyOpBuilder &self, Value &lhs, Value &rhs) -> Value {
             return self.create<arith::MulFOp>(lhs, rhs);
           })
      .def("create_divf",
           [](KapyOpBuilder &self, Value &lhs, Value &rhs) -> Value {
             return self.create<arith::DivFOp>(lhs, rhs);
           })
      .def("create_remf",
           [](KapyOpBuilder &self, Value &lhs, Value &rhs) -> Value {
             return self.create<arith::RemFOp>(lhs, rhs);
           })
      .def("create_addf",
           [](KapyOpBuilder &self, Value &lhs, Value &rhs) -> Value {
             return self.create<arith::AddFOp>(lhs, rhs);
           })
      .def("create_subf",
           [](KapyOpBuilder &self, Value &lhs, Value &rhs) -> Value {
             return self.create<arith::SubFOp>(lhs, rhs);
           })
      .def("create_muli",
           [](KapyOpBuilder &self, Value &lhs, Value &rhs) -> Value {
             return self.create<arith::MulIOp>(lhs, rhs);
           })
      .def("create_divsi",
           [](KapyOpBuilder &self, Value &lhs, Value &rhs) -> Value {
             return self.create<arith::DivSIOp>(lhs, rhs);
           })
      .def("create_divui",
           [](KapyOpBuilder &self, Value &lhs, Value &rhs) -> Value {
             return self.create<arith::DivUIOp>(lhs, rhs);
           })
      .def("create_remsi",
           [](KapyOpBuilder &self, Value &lhs, Value &rhs) -> Value {
             return self.create<arith::RemSIOp>(lhs, rhs);
           })
      .def("create_remui",
           [](KapyOpBuilder &self, Value &lhs, Value &rhs) -> Value {
             return self.create<arith::RemUIOp>(lhs, rhs);
           })
      .def("create_addi",
           [](KapyOpBuilder &self, Value &lhs, Value &rhs) -> Value {
             return self.create<arith::AddIOp>(lhs, rhs);
           })
      .def("create_subi",
           [](KapyOpBuilder &self, Value &lhs, Value &rhs) -> Value {
             return self.create<arith::SubIOp>(lhs, rhs);
           })
      .def(
          "create_fma",
          [](KapyOpBuilder &self, Value &lhs, Value &rhs, Value &acc) -> Value {
            return self.create<math::FmaOp>(lhs, rhs, acc);
          })
      .def("create_shli",
           [](KapyOpBuilder &self, Value &lhs, Value &rhs) -> Value {
             return self.create<arith::ShLIOp>(lhs, rhs);
           })
      .def("create_shrui",
           [](KapyOpBuilder &self, Value &lhs, Value &rhs) -> Value {
             return self.create<arith::ShRUIOp>(lhs, rhs);
           })
      .def("create_shrsi",
           [](KapyOpBuilder &self, Value &lhs, Value &rhs) -> Value {
             return self.create<arith::ShRSIOp>(lhs, rhs);
           })
      .def("create_minsi",
           [](KapyOpBuilder &self, Value &lhs, Value &rhs) -> Value {
             return self.create<arith::MinSIOp>(lhs, rhs);
           })
      .def("create_minui",
           [](KapyOpBuilder &self, Value &lhs, Value &rhs) -> Value {
             return self.create<arith::MinUIOp>(lhs, rhs);
           })
      .def("create_minimumf",
           [](KapyOpBuilder &self, Value &lhs, Value &rhs) -> Value {
             return self.create<arith::MinimumFOp>(lhs, rhs);
           })
      .def("create_minnumf",
           [](KapyOpBuilder &self, Value &lhs, Value &rhs) -> Value {
             return self.create<arith::MinNumFOp>(lhs, rhs);
           })
      .def("create_maxsi",
           [](KapyOpBuilder &self, Value &lhs, Value &rhs) -> Value {
             return self.create<arith::MaxSIOp>(lhs, rhs);
           })
      .def("create_maxui",
           [](KapyOpBuilder &self, Value &lhs, Value &rhs) -> Value {
             return self.create<arith::MaxUIOp>(lhs, rhs);
           })
      .def("create_maximumf",
           [](KapyOpBuilder &self, Value &lhs, Value &rhs) -> Value {
             return self.create<arith::MaximumFOp>(lhs, rhs);
           })
      .def("create_maxnumf",
           [](KapyOpBuilder &self, Value &lhs, Value &rhs) -> Value {
             return self.create<arith::MaxNumFOp>(lhs, rhs);
           })
      .def("create_clampf",
           [](KapyOpBuilder &self, Value &source, Value &low, Value &high,
              bool propagateNan) -> Value {
             return self.create<ClampFOp>(source, low, high, propagateNan);
           })
      .def("create_cmpi_sle",
           [](KapyOpBuilder &self, Value &lhs, Value &rhs) -> Value {
             return self.create<arith::CmpIOp>(arith::CmpIPredicate::sle, lhs,
                                               rhs);
           })
      .def("create_cmpi_slt",
           [](KapyOpBuilder &self, Value &lhs, Value &rhs) -> Value {
             return self.create<arith::CmpIOp>(arith::CmpIPredicate::slt, lhs,
                                               rhs);
           })
      .def("create_cmpi_sge",
           [](KapyOpBuilder &self, Value &lhs, Value &rhs) -> Value {
             return self.create<arith::CmpIOp>(arith::CmpIPredicate::sge, lhs,
                                               rhs);
           })
      .def("create_cmpi_sgt",
           [](KapyOpBuilder &self, Value &lhs, Value &rhs) -> Value {
             return self.create<arith::CmpIOp>(arith::CmpIPredicate::sgt, lhs,
                                               rhs);
           })
      .def("create_cmpi_ule",
           [](KapyOpBuilder &self, Value &lhs, Value &rhs) -> Value {
             return self.create<arith::CmpIOp>(arith::CmpIPredicate::ule, lhs,
                                               rhs);
           })
      .def("create_cmpi_ult",
           [](KapyOpBuilder &self, Value &lhs, Value &rhs) -> Value {
             return self.create<arith::CmpIOp>(arith::CmpIPredicate::ult, lhs,
                                               rhs);
           })
      .def("create_cmpi_uge",
           [](KapyOpBuilder &self, Value &lhs, Value &rhs) -> Value {
             return self.create<arith::CmpIOp>(arith::CmpIPredicate::uge, lhs,
                                               rhs);
           })
      .def("create_cmpi_ugt",
           [](KapyOpBuilder &self, Value &lhs, Value &rhs) -> Value {
             return self.create<arith::CmpIOp>(arith::CmpIPredicate::ugt, lhs,
                                               rhs);
           })
      .def("create_cmpi_eq",
           [](KapyOpBuilder &self, Value &lhs, Value &rhs) -> Value {
             return self.create<arith::CmpIOp>(arith::CmpIPredicate::eq, lhs,
                                               rhs);
           })
      .def("create_cmpi_ne",
           [](KapyOpBuilder &self, Value &lhs, Value &rhs) -> Value {
             return self.create<arith::CmpIOp>(arith::CmpIPredicate::ne, lhs,
                                               rhs);
           })
      .def("create_cmpf_olt",
           [](KapyOpBuilder &self, Value &lhs, Value &rhs) -> Value {
             return self.create<arith::CmpFOp>(arith::CmpFPredicate::OLT, lhs,
                                               rhs);
           })
      .def("create_cmpf_ogt",
           [](KapyOpBuilder &self, Value &lhs, Value &rhs) -> Value {
             return self.create<arith::CmpFOp>(arith::CmpFPredicate::OGT, lhs,
                                               rhs);
           })
      .def("create_cmpf_ole",
           [](KapyOpBuilder &self, Value &lhs, Value &rhs) -> Value {
             return self.create<arith::CmpFOp>(arith::CmpFPredicate::OLE, lhs,
                                               rhs);
           })
      .def("create_cmpf_oge",
           [](KapyOpBuilder &self, Value &lhs, Value &rhs) -> Value {
             return self.create<arith::CmpFOp>(arith::CmpFPredicate::OGE, lhs,
                                               rhs);
           })
      .def("create_cmpf_oeq",
           [](KapyOpBuilder &self, Value &lhs, Value &rhs) -> Value {
             return self.create<arith::CmpFOp>(arith::CmpFPredicate::OEQ, lhs,
                                               rhs);
           })
      .def("create_cmpf_one",
           [](KapyOpBuilder &self, Value &lhs, Value &rhs) -> Value {
             return self.create<arith::CmpFOp>(arith::CmpFPredicate::ONE, lhs,
                                               rhs);
           })
      .def("create_cmpf_ult",
           [](KapyOpBuilder &self, Value &lhs, Value &rhs) -> Value {
             return self.create<arith::CmpFOp>(arith::CmpFPredicate::ULT, lhs,
                                               rhs);
           })
      .def("create_cmpf_ugt",
           [](KapyOpBuilder &self, Value &lhs, Value &rhs) -> Value {
             return self.create<arith::CmpFOp>(arith::CmpFPredicate::UGT, lhs,
                                               rhs);
           })
      .def("create_cmpf_ule",
           [](KapyOpBuilder &self, Value &lhs, Value &rhs) -> Value {
             return self.create<arith::CmpFOp>(arith::CmpFPredicate::ULE, lhs,
                                               rhs);
           })
      .def("create_cmpf_uge",
           [](KapyOpBuilder &self, Value &lhs, Value &rhs) -> Value {
             return self.create<arith::CmpFOp>(arith::CmpFPredicate::UGE, lhs,
                                               rhs);
           })
      .def("create_cmpf_ueq",
           [](KapyOpBuilder &self, Value &lhs, Value &rhs) -> Value {
             return self.create<arith::CmpFOp>(arith::CmpFPredicate::UEQ, lhs,
                                               rhs);
           })
      .def("create_cmpf_une",
           [](KapyOpBuilder &self, Value &lhs, Value &rhs) -> Value {
             return self.create<arith::CmpFOp>(arith::CmpFPredicate::UNE, lhs,
                                               rhs);
           })
      .def("create_andi",
           [](KapyOpBuilder &self, Value &lhs, Value &rhs) -> Value {
             return self.create<arith::AndIOp>(lhs, rhs);
           })
      .def("create_ori",
           [](KapyOpBuilder &self, Value &lhs, Value &rhs) -> Value {
             return self.create<arith::OrIOp>(lhs, rhs);
           })
      .def("create_xori",
           [](KapyOpBuilder &self, Value &lhs, Value &rhs) -> Value {
             return self.create<arith::XOrIOp>(lhs, rhs);
           })
      .def("create_mk_global",
           [](KapyOpBuilder &self, Value &globalAddress, Value &dynamicOffset,
              Value &size0, Value &size1, Value &stride0, Value &stride1,
              Type &elementType) -> Value {
             auto encoding = EncodingAttr::get(self.getBuilder().getContext(),
                                               MemorySpace::GLOBAL_MEMORY);
             SmallVector<int64_t, 2> shape(2, ShapedType::kDynamic);
             auto resultType =
                 RankedTensorType::get(shape, elementType, encoding);
             return self.create<MkGlobalOp>(resultType, globalAddress,
                                            dynamicOffset, size0, size1,
                                            stride0, stride1);
           })
      .def("create_sv_global",
           [](KapyOpBuilder &self, Value &source, Value &start0, Value &end0,
              Value &start1, Value &end1,
              std::vector<int64_t> &shape) -> Value {
             auto sourceType = cast<RankedTensorType>(source.getType());
             auto resultType = cloneWithShape(sourceType, shape);
             return self.create<SvGlobalOp>(resultType, source, start0, end0,
                                            start1, end1);
           })
      .def("create_ld_global",
           [](KapyOpBuilder &self, Value &source, PaddingOption paddingOption,
              CacheModifier cacheModifier, EvictPriority evictPriority,
              bool isVolatile) -> Value {
             return self.create<LdGlobalOp>(source.getType(), source,
                                            paddingOption, cacheModifier,
                                            evictPriority, isVolatile);
           })
      .def("create_st_global",
           [](KapyOpBuilder &self, Value &source, Value &target,
              CacheModifier cacheModifier, EvictPriority evictPriority,
              bool isVolatile) {
             self.create<StGlobalOp>(source, target, cacheModifier,
                                     evictPriority, isVolatile);
           })
      .def("create_mk_shared",
           [](KapyOpBuilder &self, std::vector<int64_t> &shape,
              Type &elementType, bool rowMajor) -> Value {
             auto encoding = EncodingAttr::get(self.getContext(),
                                               MemorySpace::SHARED_MEMORY);
             auto resultType =
                 RankedTensorType::get(shape, elementType, encoding);
             auto op = self.create<MkSharedOp>(resultType);
             op->setAttr("kapy.row_major",
                         self.getBuilder().getBoolAttr(rowMajor));
             return op;
           })
      .def("create_sv_shared",
           [](KapyOpBuilder &self, Value &source, Value &start0, Value &end0,
              Value &start1, Value &end1,
              std::vector<int64_t> &shape) -> Value {
             auto sourceType = cast<RankedTensorType>(source.getType());
             auto resultType = cloneWithShape(sourceType, shape);
             return self.create<SvSharedOp>(resultType, source, start0, end0,
                                            start1, end1);
           })
      .def("create_splat",
           [](KapyOpBuilder &self, Value &source,
              std::vector<int64_t> &shape) -> Value {
             return self.create<SplatOp>(
                 RankedTensorType::get(shape, source.getType()), source);
           })
      .def("create_broadcast",
           [](KapyOpBuilder &self, Value &source,
              std::vector<int64_t> &shape) -> Value {
             if (auto sourceType =
                     dyn_cast<RankedTensorType>(source.getType())) {
               auto resultType = cloneWithShape(sourceType, shape);
               return self.create<BroadcastOp>(resultType, source);
             }
             throw std::invalid_argument("source must be a tensor");
           })
      .def("create_transpose",
           [](KapyOpBuilder &self, Value &source) -> Value {
             return self.create<TransposeOp>(source);
           })
      .def("create_program_id",
           [](KapyOpBuilder &self, unsigned axis) -> Value {
             if (axis > 2)
               throw py::index_error("program axis must be 0, 1, 2");
             return self.create<ProgramIdOp>(axis);
           })
      .def("create_warp_id",
           [](KapyOpBuilder &self) -> Value { return self.create<WarpIdOp>(); })
      .def("create_lane_id",
           [](KapyOpBuilder &self) -> Value { return self.create<LaneIdOp>(); })
      .def("create_matmul",
           [](KapyOpBuilder &self, Value &lhs, Value &rhs, Value &acc,
              MatmulImplWay implWay) -> Value {
             return self.create<MatmulOp>(lhs, rhs, acc, implWay);
           })
      .def("create_floor",
           [](KapyOpBuilder &self, Value &source) -> Value {
             return self.create<math::FloorOp>(source);
           })
      .def("create_ceil",
           [](KapyOpBuilder &self, Value &source) -> Value {
             return self.create<math::CeilOp>(source);
           })
      .def("create_exp",
           [](KapyOpBuilder &self, Value &source) -> Value {
             return self.create<math::ExpOp>(source);
           })
      .def("create_exp2",
           [](KapyOpBuilder &self, Value &source) -> Value {
             return self.create<math::Exp2Op>(source);
           })
      .def("create_cos",
           [](KapyOpBuilder &self, Value &source) -> Value {
             return self.create<math::CosOp>(source);
           })
      .def("create_sin",
           [](KapyOpBuilder &self, Value &source) -> Value {
             return self.create<math::SinOp>(source);
           })
      .def("create_log",
           [](KapyOpBuilder &self, Value &source) -> Value {
             return self.create<math::LogOp>(source);
           })
      .def("create_log2",
           [](KapyOpBuilder &self, Value &source) -> Value {
             return self.create<math::Log2Op>(source);
           })
      .def("create_erf",
           [](KapyOpBuilder &self, Value &source) -> Value {
             return self.create<math::ErfOp>(source);
           })
      .def("create_sqrt",
           [](KapyOpBuilder &self, Value &source) -> Value {
             return self.create<math::SqrtOp>(source);
           })
      .def("create_rsqrt",
           [](KapyOpBuilder &self, Value &source) -> Value {
             return self.create<math::RsqrtOp>(source);
           })
      .def("create_absf",
           [](KapyOpBuilder &self, Value &source) -> Value {
             return self.create<math::AbsFOp>(source);
           })
      .def("create_absi",
           [](KapyOpBuilder &self, Value &source) -> Value {
             return self.create<math::AbsIOp>(source);
           })
      .def("create_reduce",
           [](KapyOpBuilder &self, Value &source, unsigned axis) -> OpState {
             return self.create<ReduceOp>(source, axis);
           })
      .def("create_select",
           [](KapyOpBuilder &self, Value &condition, Value &trueValue,
              Value &falseValue) -> Value {
             return self.create<arith::SelectOp>(condition, trueValue,
                                                 falseValue);
           })
      .def("create_elementwise_extern_lib",
           [](KapyOpBuilder &self, Type &resultType,
              std::vector<Value> &operands, std::string &libName,
              std::string &libPath, std::string &symName, bool pure) -> Value {
             return self.create<ElementwiseExternLibOp>(
                 resultType, operands, libName, libPath, symName, pure);
           })
      .def("create_elementwise_inline_asm",
           [](KapyOpBuilder &self, Type &resultType,
              std::vector<Value> &operands, std::string &asmStr,
              std::string &constraints, bool pure) -> OpState {
             return self.create<ElementwiseInlineAsmOp>(
                 resultType, operands, asmStr, constraints, pure);
           });
}
