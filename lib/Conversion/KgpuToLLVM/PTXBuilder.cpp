//===- PTXBuilder.cpp -------------------------------------------*- C++ -*-===//
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

#include "kapy/Conversion/KgpuToLLVM/PTXBuilder.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "llvm/ADT/StringExtras.h"

using namespace mlir;
using namespace mlir::kapy;

PTXBuilder::Operand *PTXBuilder::newOperand(Value value, StringRef constraint) {
  operands.emplace_back(std::make_unique<Operand>(value, constraint));
  auto *operand = operands.back().get();
  operand->index = numOperands++;
  return operand;
}

void PTXBuilder::initOperand(Operand *operand, int64_t value) {
  unsigned bitWidth = 0;
  if (operand->constraint[1] == 'c' || operand->constraint[1] == 'h')
    bitWidth = 16;
  else if (operand->constraint[1] == 'r')
    bitWidth = 32;
  else if (operand->constraint[1] == 'l')
    bitWidth = 64;
  else
    llvm_unreachable("unsupported constraint");
  auto &mov = create("mov")->o("u" + std::to_string(bitWidth));
  mov(operand, newConstantOperand(value));
}

PTXBuilder::Operand *PTXBuilder::newOperand(StringRef constraint,
                                            std::optional<int64_t> initValue) {
  // Constraint should be something like "=r".
  assert(constraint.size() == 2 && constraint[0] == '=');
  auto *operand = newOperand();
  operand->index = numOperands++;
  operand->constraint = constraint;
  if (initValue.has_value())
    initOperand(operand, initValue.value());
  return operand;
}

PTXBuilder::Operand *PTXBuilder::newOperand(unsigned index) {
  assert(index < numOperands);
  auto *operand = newOperand();
  operand->index = numOperands++;
  operand->constraint = std::to_string(index);
  return operand;
}

PTXBuilder::Operand *PTXBuilder::newConstantOperand(const std::string &value) {
  operands.emplace_back(std::make_unique<Operand>());
  operands.back()->formatter = [value](unsigned index) { return value; };
  return operands.back().get();
}

PTXBuilder::Operand *PTXBuilder::newConstantOperand(int64_t value) {
  std::stringstream ss;
  ss << "0x" << std::hex << value;
  return newConstantOperand(ss.str());
}

SmallVector<PTXBuilder::Operand *> PTXBuilder::getOperands() const {
  SmallVector<Operand *> operands;
  for (auto &operand : this->operands)
    if (!operand->isList())
      operands.push_back(operand.get());
  return operands;
}

SmallVector<Value> PTXBuilder::getMLIRValues() const {
  SmallVector<Value> values;
  for (auto *operand : getOperands())
    if (!operand->isList() && operand->value)
      values.push_back(operand->value);
  return values;
}

std::string PTXBuilder::getConstraints() const {
  SmallVector<std::string, 4> constraints;
  for (auto *operand : getOperands())
    constraints.push_back(operand->constraint);
  return join(constraints, ",");
}

Value PTXBuilder::launch(OpBuilder &rewriter, Location loc, Type resultType,
                         bool hasSideEffect, bool isAlignStack,
                         ArrayRef<Attribute> operandAttrs) const {
  auto *context = rewriter.getContext();
  auto inlineAsmOp = rewriter.create<LLVM::InlineAsmOp>(
      loc, resultType, getMLIRValues(), dump(), getConstraints(),   //
      hasSideEffect, isAlignStack,                                  //
      LLVM::AsmDialectAttr::get(context, LLVM::AsmDialect::AD_ATT), //
      ArrayAttr::get(context, operandAttrs));
  return inlineAsmOp.getRes();
}

std::string PTXBuilder::Operand::dump() const {
  if (formatter)
    return formatter(index);
  if (!isList())
    return "$" + std::to_string(index);
  SmallVector<std::string> operands;
  for (auto *operand : list)
    operands.push_back(operand->dump());
  return "{ " + join(operands, ", ") + " }";
}

PTXBuilder::Operand *
PTXBuilder::newAddressOperand(Value address, StringRef constraint, int offset) {
  auto *operand = newOperand(address, constraint);
  operand->formatter = [offset](unsigned index) {
    std::stringstream ss;
    ss << "[ $" << index << " + " << offset << " ]";
    return ss.str();
  };
  return operand;
}

std::string PTXBuilder::dump() const {
  SmallVector<std::string> lines;
  for (auto &execution : executions)
    lines.push_back(execution->dump());
  return join(lines, "\n\t");
}

PTXInstructionExecution &
PTXInstructionCommon::call(ArrayRef<Operand *> operands,
                           bool onlyAttachMLIRValues) {
  if (onlyAttachMLIRValues) {
    // Nearly impossible to make the $0, $1 in two PTX code snippets to point to
    // the same MLIR values in onlyAttachMLIRValues mode.
    assert(builder->executions.empty());
    builder->reorderOperands(operands);
  }
  builder->executions.emplace_back(std::make_unique<PTXInstructionExecution>(
      this, operands, onlyAttachMLIRValues));
  return *builder->executions.back();
}

PTXInstructionExecution &
PTXInstructionCommon::operator()(ArrayRef<Operand *> operands,
                                 bool onlyAttachMLIRValues) {
  return call(operands, onlyAttachMLIRValues);
}

std::string PTXInstructionExecution::dump() const {
  auto instruction = join(this->instruction->keywords, ".");
  if (onlyAttachMLIRValues)
    return instruction;

  std::string string;
  llvm::raw_string_ostream os(string);

  if (predOperand) {
    assert(predOperand->formatter);
    os << predOperand->formatter(predOperand->index) << " ";
  }

  SmallVector<std::string, 4> operands;
  for (auto *operand : this->operands)
    operands.push_back(operand->dump());

  os << instruction << " " << join(operands, ", ") << ";";
  os.flush();
  return string;
}

SmallVector<PTXBuilder::Operand *>
PTXInstructionExecution::getOperands() const {
  SmallVector<Operand *> operands;
  for (auto *operand : this->operands)
    if (operand->isList())
      operands.append(operand->list.begin(), operand->list.end());
    else
      operands.push_back(operand);
  return operands;
}

PTXInstruction &PTXInstruction::global() {
  o("global");
  return *this;
}

PTXInstruction &PTXInstruction::shared() {
  o("shared");
  return *this;
}

PTXInstruction &PTXInstruction::v(unsigned vectorSize, bool predicate) {
  if (vectorSize > 1)
    o("v" + std::to_string(vectorSize), predicate);
  return *this;
}

PTXInstruction &PTXInstruction::b(unsigned bitWidth) {
  o("b" + std::to_string(bitWidth));
  return *this;
}
