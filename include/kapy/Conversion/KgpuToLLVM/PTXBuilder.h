//===- PTXBuilder.h ---------------------------------------------*- C++ -*-===//
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

#ifndef KAPY_CONVERSION_KGPUTOLLVM_PTXBUILDER_H
#define KAPY_CONVERSION_KGPUTOLLVM_PTXBUILDER_H

#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/Value.h"

namespace mlir {
namespace kapy {

class PTXInstruction;
class PTXInstructionCommon;
class PTXInstructionExecution;

/// PTXBuilder helps to manage a PTX program consists of single or multiple
/// instructions.
///
/// A helper for building an ASM program, the objective of PTXBuilder is to give
/// a thin encapsulation and make the ASM code for MLIR LLVM Dialect more clear.
/// Currently, several factors are introduced to reduce the need for mixing
/// string and C++ if-else code.
///
/// To build:
///   @$3 asm("@%3 add.s32 %0, %1, %2;" : "=r"(i) : "r"(j), "r"(k), "b"(p));
///
///   PTXBuilder builder;
///   auto &add = builder.create<>();
///   add.predicate(pValue).o("lo").o("u32"); // add any suffix
///   // predicate here binds %0 to pValue which is a mlir::Value
///
///   auto *iOperand = builder.newOperand(iValue, "r"); // %0 bind to iValue
///   auto *jOperand = builder.newOperand(jValue, "r"); // %1 bind to jValue
///   auto *kOperand = builder.newOperand(kValue, "r"); // %2 bind to kValue
///   // set operands and predicate
///   add(iOperand, jOperand, kOperand).predicate(pValue);
///
/// To get the asm code:
///   builder.dump()
///
/// To get all the mlir::Value used in the PTX code:
///   builder.getMLIRValues() // get {pValue, iValue, jValue, kValue}
///
/// To get the string containing all the constraints with ',' separated:
///   builder.getConstraints() // get "=r,r,k"
///
/// PTXBuilder can build PTX code with multiple instructions, sample code:
///   PTXBuilder builder;
///   auto &mov = builder.create("mov");
///   auto &cp = builder.create("cp");
///   mov(...);
///   cp(...);
///
/// This will get PTX code with two instructions.
///
/// Similar to a C function, a declared PTXInstruction instance can be launched
/// multiple times with different operands, for example:
///   auto &mov = builder.create("mov");
///   mov(... some operands ...);
///   mov(... some different operands ...);
///
/// Finally, we will get PTX code with two mov instructions.
class PTXBuilder {
public:
  struct Operand {
    Value value;
    std::string constraint;
    unsigned index;
    SmallVector<Operand *> list;
    std::function<std::string(unsigned)> formatter;

    Operand() = default;
    Operand(const Operand &) = delete;
    Operand(Value value, StringRef constraint)
        : value(value), constraint(constraint) {}

    bool isList() const { return !value && constraint.empty(); }

    Operand *listPushBack(Operand *operand) {
      list.push_back(operand);
      return this;
    }

    Operand *listAt(unsigned index) const {
      assert(index < list.size());
      return list[index];
    }

    std::string dump() const;
  };

  template <typename InstructionT = PTXInstruction, typename... Ts>
  InstructionT *create(Ts &&...args) {
    instructions.emplace_back(std::make_unique<InstructionT>(this, args...));
    return static_cast<InstructionT *>(instructions.back().get());
  }

  /// Create a list of operands.
  Operand *newListOperand() { return newOperand(); }

  Operand *newListOperand(ArrayRef<std::pair<Value, std::string>> items) {
    auto *list = newOperand();
    for (auto &item : items)
      list->listPushBack(newOperand(item.first, item.second));
    return list;
  }

  Operand *newListOperand(unsigned numOperands, Value value,
                          const std::string &constraint) {
    auto *list = newOperand();
    for (unsigned i = 0; i < numOperands; ++i)
      list->listPushBack(newOperand(value, constraint));
    return list;
  }

  Operand *newListOperand(unsigned numOperands, const std::string &constraint) {
    auto *list = newOperand();
    for (unsigned i = 0; i < numOperands; ++i)
      list->listPushBack(newOperand(constraint));
    return list;
  }

  /// Create a new operand. It will not add to operand list.
  Operand *newOperand(Value value, StringRef constraint,
                      std::function<std::string(unsigned)> formatter = nullptr);

  /// Create a new operand which is written to, that is, the constraint starts
  /// with "=", e.g. "=r".
  /// If the operand will be used in predicated execution, users may want to
  /// initialize it before use. Otherwise if the register is undefined and ptxas
  /// can perform aggressive optimizations that may lead to incorrect results.
  Operand *newOperand(StringRef constraint, bool init = false);

  /// Create a new operand that is tied to a previous operand. In this case the
  /// asm would be premitted to write an input register. Instead of providing
  /// constraint code for this operand, the constraint code of the tied operand
  /// is used.
  Operand *newOperand(unsigned index);

  /// Create a constant integer operand.
  Operand *newConstantOperand(int64_t value);
  /// Create a constant operand with explicit code specified.
  Operand *newConstantOperand(const std::string &value);

  Operand *newAddressOperand(Value value, StringRef constraint, int offset = 0);

  SmallVector<Operand *> getOperands() const;

  SmallVector<Value> getMLIRValues() const;

  std::string getConstraints() const;

  std::string dump() const;

  Value launch(RewriterBase &rewriter, Location loc, Type resultType,
               bool hasSideEffect = true, bool isAlignStack = false,
               ArrayRef<Attribute> operandAttrs = {}) const;

protected:
  SmallVector<std::unique_ptr<Operand>> operands;
  SmallVector<std::unique_ptr<PTXInstructionCommon>, 2> instructions;
  SmallVector<std::unique_ptr<PTXInstructionExecution>, 4> executions;
  unsigned numOperands = 0;

private:
  Operand *newOperand() {
    operands.emplace_back(std::make_unique<Operand>());
    return operands.back().get();
  }

  void initOperand(Operand *operand);

  /// Make the operands follow the provided order.
  void reorderOperands(ArrayRef<Operand *> order) {
    assert(order.size() == operands.size());
    // The order in `operands` is unnecessary when onlyAttachMLIRValues = false,
    // but it does necessary when onlyAttachMLIRValues = true.
    // The $0, $1, ... are determined by PTX code snippet passed from external.
    std::sort(
        operands.begin(), operands.end(),
        [&](std::unique_ptr<Operand> &lhs, std::unique_ptr<Operand> &rhs) {
          auto lhsIndex = std::find(order.begin(), order.end(), lhs.get());
          auto rhsIndex = std::find(order.begin(), order.end(), rhs.get());
          assert(lhsIndex != order.end());
          assert(rhsIndex != order.end());
          return lhsIndex < rhsIndex;
        });
  }

  friend class PTXInstruction;
  friend class PTXInstructionCommon;
};

/// PTX instruction common interface. Put the logic for all the instructions
/// here.
class PTXInstructionCommon {
public:
  explicit PTXInstructionCommon(PTXBuilder *builder) : builder(builder) {}

  using Operand = PTXBuilder::Operand;

  PTXInstructionExecution &operator()() { return call({}); }
  PTXInstructionExecution &operator()(Operand *a) { return call({a}); }
  PTXInstructionExecution &operator()(Operand *a, Operand *b) {
    return call({a, b});
  }
  PTXInstructionExecution &operator()(Operand *a, Operand *b, Operand *c) {
    return call({a, b, c});
  }
  PTXInstructionExecution &operator()(Operand *a, Operand *b, Operand *c,
                                      Operand *d) {
    return call({a, b, c, d});
  }
  PTXInstructionExecution &operator()(Operand *a, Operand *b, Operand *c,
                                      Operand *d, Operand *e) {
    return call({a, b, c, d, e});
  }
  PTXInstructionExecution &operator()(Operand *a, Operand *b, Operand *c,
                                      Operand *d, Operand *e, Operand *f) {
    return call({a, b, c, d, e, f});
  }
  PTXInstructionExecution &operator()(Operand *a, Operand *b, Operand *c,
                                      Operand *d, Operand *e, Operand *f,
                                      Operand *g) {
    return call({a, b, c, d, e, f, g});
  }
  PTXInstructionExecution &operator()(Operand *a, Operand *b, Operand *c,
                                      Operand *d, Operand *e, Operand *f,
                                      Operand *g, Operand *h) {
    return call({a, b, c, d, e, f, g, h});
  }

  /// Set operands for this instruction.
  PTXInstructionExecution &operator()(ArrayRef<Operand *> operands,
                                      bool onlyAttachMLIRValues = false);

protected:
  PTXBuilder *builder;
  SmallVector<std::string, 4> instructionParts;

  // Call the instruction with operands, `onlyAttachMLIRValues` indicate that it
  // simply attach the MLIR values to the PTX without generating the operand ids
  // (such as $0, $1) in PTX code.
  PTXInstructionExecution &call(ArrayRef<Operand *> operands,
                                bool onlyAttachMLIRValues = false);

  friend class PTXInstructionExecution;
};

template <typename ConcreteT>
class PTXInstructionBase : public PTXInstructionCommon {
public:
  using Operand = PTXBuilder::Operand;

  explicit PTXInstructionBase(PTXBuilder *builder, const std::string &name)
      : PTXInstructionCommon(builder) {
    o(name);
  }

  /// Append a suffix to the instruction, for example:
  ///   PTXInstruction("add").o("s32");
  /// will get "add.s32".
  ///
  /// A predicate is used to tell whether to apply suffix, so that no if-else
  /// code needed, for example:
  ///   PTXInstruction("add").o("s32", isS32).o("u32", !isS32);
  /// will get "add.s32" if isS32 is true.
  ConcreteT &o(const std::string &suffix, bool predicate = true) {
    if (predicate)
      instructionParts.push_back(suffix);
    return *static_cast<ConcreteT *>(this);
  }
};

class PTXInstruction : public PTXInstructionBase<PTXInstruction> {
public:
  using PTXInstructionBase<PTXInstruction>::PTXInstructionBase;

  /// Append a ".global" to the instruction.
  PTXInstruction &global();

  /// Append a ".shared" to the instruction.
  PTXInstruction &shared();

  /// Append a ".v[0-9]+" to the instruction.
  PTXInstruction &v(unsigned vecWidth, bool predicate = true);

  /// Append a ".b[0-9]+" to the instruction.
  PTXInstruction &b(unsigned bitWidth);
};

/// Record the operands and context for launching a PTXInstruction.
class PTXInstructionExecution {
public:
  using Operand = PTXBuilder::Operand;

  PTXInstructionExecution() = default;
  explicit PTXInstructionExecution(PTXInstructionCommon *instruction,
                                   ArrayRef<Operand *> operands,
                                   bool onlyAttachMLIRValues)
      : operands(operands), instruction(instruction),
        onlyAttachMLIRValues(onlyAttachMLIRValues) {}

  /// Prefix a predicate to the instruction.
  PTXInstructionExecution &predicate(Value value, StringRef constraint = "b") {
    predOperand = instruction->builder->newOperand(value, constraint);
    return *this;
  }

  /// Prefix a !predicate to the instruction.
  PTXInstructionExecution &predicateNot(Value value, StringRef constraint) {
    predOperand = instruction->builder->newOperand(value, constraint);
    predOperand->formatter = [](unsigned index) {
      return "@!$" + std::to_string(index);
    };
    return *this;
  }

  std::string dump() const;

  SmallVector<Operand *> getOperands() const;

private:
  SmallVector<Operand *> operands;
  PTXInstructionCommon *instruction;
  Operand *predOperand;
  bool onlyAttachMLIRValues = true;
};

/*
class PTXCpAsyncInstruction : public PTXInstructionBase<PTXCpAsyncInstruction> {
public:
  explicit PTXCpAsyncInstruction(PTXBuilder *builder,
                                 CacheModifier cacheModifier)
      : PTXInstructionBase(builder, "cp.async") {
    o(stringifyCacheModifier(cacheModifier).str());
    o("shared");
    o("global");
  }
};
*/

} // namespace kapy
} // namespace mlir

#endif // KAPY_CONVERSION_KGPUTOLLVM_PTXBUILDER_H
