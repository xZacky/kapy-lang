//===- Integer.cpp ----------------------------------------------*- C++ -*-===//
//
// This file implements classes and functions used by data flow analysis for
// integers.
//
//===----------------------------------------------------------------------===//

#include "kapy/Analysis/Integer.h"
#include "kapy/Analysis/Utils.h"
#include "kapy/Dialect/Kapy/IR/Kapy.h"
#include "kapy/Dialect/Kapy/IR/Utils.h"
#include "mlir/Analysis/DataFlow/SparseAnalysis.h"

using namespace mlir;
using namespace mlir::kapy;
using namespace mlir::dataflow;

/// Implementation of euclidean algorithm.
template <typename I> static I gcdImpl(I a, I b, I &x, I &y) {
  if (a == 0) {
    x = 0;
    y = 1;
    return b;
  }
  I x1, y1; // to store results of recursive call
  I g = gcdImpl(b % a, a, x1, y1);
  // update `x` and `y` using results of recursive call
  x = y1 - (b / a) * x1;
  y = x1;
  return g;
}

/// Greatest common divisor.
template <typename I> static I gcd(I a, I b) {
  static_assert(std::is_integral_v<I>);
  if (a == 0)
    return b;
  if (b == 0)
    return a;
  I x, y;
  return gcdImpl(a, b, x, y);
}

/// Greatest power of two divisor. If `x == 0`, return the greatest power of two
/// for type `I`.
template <typename I> static I gpd(I x) {
  static_assert(std::is_integral_v<I>);
  if (x == 0)
    return static_cast<I>(1) << (sizeof(I) * 8 - 2);
  return x & (~(x - 1));
}

/// If `a * b` overflows, return greatest the power of two for type `I`.
template <typename I> static I mul(I a, I b) {
  static_assert(std::is_integral_v<I>);
  I g = gpd<I>(0);
  return a > g / b ? g : a * b;
}

namespace {

class IntegerInfoVisitor {
public:
  IntegerInfoVisitor() = default;
  virtual ~IntegerInfoVisitor() = default;

  virtual bool match(Operation *op) = 0;

  virtual IntegerInfo
  getIntegerInfo(Operation *op,
                 ArrayRef<const Lattice<IntegerInfo> *> operands) = 0;
};

class IntegerInfoVisitorList {
public:
  template <typename... Ts, typename = std::enable_if_t<sizeof...(Ts) != 0>>
  void append() {
    (visitors.emplace_back(std::make_unique<Ts>()), ...);
  }

  IntegerInfo apply(Operation *op,
                    ArrayRef<const Lattice<IntegerInfo> *> operands) {
    for (auto &visitor : visitors)
      if (visitor->match(op))
        return visitor->getIntegerInfo(op, operands);
    return IntegerInfo();
  }

private:
  std::vector<std::unique_ptr<IntegerInfoVisitor>> visitors;
};

template <typename OpT> class OpIntegerInfoVisitor : public IntegerInfoVisitor {
public:
  using IntegerInfoVisitor::IntegerInfoVisitor;

  virtual bool match(Operation *op) override { return isa<OpT>(op); }

  virtual IntegerInfo
  getIntegerInfo(Operation *op,
                 ArrayRef<const Lattice<IntegerInfo> *> operands) override {
    return getIntegerInfo(cast<OpT>(op), operands);
  }

  virtual IntegerInfo
  getIntegerInfo(OpT op, //
                 ArrayRef<const Lattice<IntegerInfo> *> operands) = 0;
};

template <typename OpT>
class BinOpIntegerInfoVisitor : public OpIntegerInfoVisitor<OpT> {
public:
  using OpIntegerInfoVisitor<OpT>::OpIntegerInfoVisitor;

  virtual IntegerInfo
  getIntegerInfo(OpT op,
                 ArrayRef<const Lattice<IntegerInfo> *> operands) override {
    assert(operands.size() == 2);
    auto lhs = operands[0]->getValue();
    auto rhs = operands[1]->getValue();
    int64_t divisibility = 1;
    std::optional<int64_t> constant = getConstant(op, lhs, rhs);
    if (constant.has_value())
      divisibility = gpd(constant.value());
    else
      divisibility = getDivisibility(op, lhs, rhs);
    return IntegerInfo(divisibility, constant);
  }

protected:
  virtual int64_t getDivisibility(OpT op, //
                                  const IntegerInfo &lhs,
                                  const IntegerInfo &rhs) {
    return 1;
  }

  virtual std::optional<int64_t> getConstant(OpT op, //
                                             const IntegerInfo &lhs,
                                             const IntegerInfo &rhs) {
    return std::nullopt;
  }
};

template <typename OpT>
class CastOpIntegerInfoVisitor : public OpIntegerInfoVisitor<OpT> {
public:
  using OpIntegerInfoVisitor<OpT>::OpIntegerInfoVisitor;

  virtual IntegerInfo
  getIntegerInfo(OpT op,
                 ArrayRef<const Lattice<IntegerInfo> *> operands) override {
    return operands[0]->getValue();
  }
};

class ConstantOpIntegerInfoVisitor
    : public OpIntegerInfoVisitor<arith::ConstantOp> {
public:
  using OpIntegerInfoVisitor<arith::ConstantOp>::OpIntegerInfoVisitor;

  virtual IntegerInfo
  getIntegerInfo(arith::ConstantOp op,
                 ArrayRef<const Lattice<IntegerInfo> *> operands) override {
    auto intAttr = dyn_cast<IntegerAttr>(op.getValue());
    auto boolAttr = dyn_cast<BoolAttr>(op.getValue());
    if (intAttr || boolAttr) {
      int64_t value;
      if (intAttr)
        value = intAttr.getInt();
      else
        value = boolAttr.getValue() ? 1 : 0;
      return IntegerInfo(gpd(value), value);
    }
    return IntegerInfo();
  }
};

template <typename OpT>
class AddSubOpIntegerInfoVisitor : public BinOpIntegerInfoVisitor<OpT> {
public:
  using BinOpIntegerInfoVisitor<OpT>::BinOpIntegerInfoVisitor;

private:
  virtual int64_t getDivisibility(OpT op, //
                                  const IntegerInfo &lhs,
                                  const IntegerInfo &rhs) override {
    return gcd(lhs.getDivisibility(), rhs.getDivisibility());
  }

  virtual std::optional<int64_t> getConstant(OpT op, //
                                             const IntegerInfo &lhs,
                                             const IntegerInfo &rhs) override {
    if (lhs.hasConstant() && rhs.hasConstant()) {
      if constexpr (std::is_same_v<OpT, arith::AddIOp>)
        return lhs.getConstant() + rhs.getConstant();
      if constexpr (std::is_same_v<OpT, arith::SubIOp>)
        return lhs.getConstant() - rhs.getConstant();
    }
    return std::nullopt;
  }
};

class MulIOpIntegerInfoVisitor : public BinOpIntegerInfoVisitor<arith::MulIOp> {
public:
  using BinOpIntegerInfoVisitor<arith::MulIOp>::BinOpIntegerInfoVisitor;

private:
  virtual int64_t getDivisibility(arith::MulIOp op, //
                                  const IntegerInfo &lhs,
                                  const IntegerInfo &rhs) override {
    return mul(lhs.getDivisibility(), rhs.getDivisibility());
  }

  virtual std::optional<int64_t> getConstant(arith::MulIOp op, //
                                             const IntegerInfo &lhs,
                                             const IntegerInfo &rhs) override {
    if (lhs.hasConstantEqualTo(0) || rhs.hasConstantEqualTo(0))
      return 0;
    if (lhs.hasConstant() && rhs.hasConstant())
      return mul(lhs.getConstant(), rhs.getConstant());
    return std::nullopt;
  }
};

template <typename OpT>
class DivOpIntegerInfoVisitor : public BinOpIntegerInfoVisitor<OpT> {
public:
  using BinOpIntegerInfoVisitor<OpT>::BinOpIntegerInfoVisitor;

private:
  virtual int64_t getDivisibility(OpT op, //
                                  const IntegerInfo &lhs,
                                  const IntegerInfo &rhs) override {
    if (lhs.hasConstantEqualTo(0))
      return lhs.getDivisibility();
    if (rhs.hasConstantEqualTo(1))
      return lhs.getDivisibility();
    return std::max<int64_t>(lhs.getDivisibility() / rhs.getDivisibility(), 1);
  }

  virtual std::optional<int64_t> getConstant(OpT op, //
                                             const IntegerInfo &lhs,
                                             const IntegerInfo &rhs) override {
    if (lhs.hasConstantEqualTo(0))
      return 0;
    if (lhs.hasConstant() && rhs.hasConstant())
      return lhs.getConstant() / rhs.getConstant();
    return std::nullopt;
  }
};

template <typename OpT>
class RemOpIntegerInfoVisitor : public BinOpIntegerInfoVisitor<OpT> {
public:
  using BinOpIntegerInfoVisitor<OpT>::BinOpIntegerInfoVisitor;

private:
  virtual int64_t getDivisibility(OpT op, //
                                  const IntegerInfo &lhs,
                                  const IntegerInfo &rhs) override {
    return gcd(lhs.getDivisibility(), rhs.getDivisibility());
  }

  virtual std::optional<int64_t> getConstant(OpT op, //
                                             const IntegerInfo &lhs,
                                             const IntegerInfo &rhs) override {
    if (rhs.hasConstantEqualTo(1))
      return 0;
    if (lhs.hasConstant() && rhs.hasConstant())
      return lhs.getConstant() % rhs.getConstant();
    return std::nullopt;
  }
};

class CmpIOpIntegerInfoVisitor : public OpIntegerInfoVisitor<arith::CmpIOp> {
public:
  using OpIntegerInfoVisitor<arith::CmpIOp>::OpIntegerInfoVisitor;

  virtual IntegerInfo
  getIntegerInfo(arith::CmpIOp op,
                 ArrayRef<const Lattice<IntegerInfo> *> operands) override {
    assert(operands.size() == 2);
    auto lhs = operands[0]->getValue();
    auto rhs = operands[1]->getValue();
    int64_t divisibility = 1;
    std::optional<int64_t> constant;
    if (lhs.hasConstant() && rhs.hasConstant()) {
      auto predicate = op.getPredicate();
      constant =
          compare(predicate, lhs.getConstant(), rhs.getConstant()) ? 1 : 0;
      divisibility = gpd(constant.value());
    }
    return IntegerInfo(divisibility, constant);
  }

private:
  static bool isGt(arith::CmpIPredicate predicate) {
    return predicate == arith::CmpIPredicate::sgt ||
           predicate == arith::CmpIPredicate::ugt;
  }
  static bool isGe(arith::CmpIPredicate predicate) {
    return predicate == arith::CmpIPredicate::sge ||
           predicate == arith::CmpIPredicate::uge;
  }
  static bool isLt(arith::CmpIPredicate predicate) {
    return predicate == arith::CmpIPredicate::slt ||
           predicate == arith::CmpIPredicate::ult;
  }
  static bool isLe(arith::CmpIPredicate predicate) {
    return predicate == arith::CmpIPredicate::sle ||
           predicate == arith::CmpIPredicate::ule;
  }
  static bool compare(arith::CmpIPredicate predicate, //
                      int64_t lhs, int64_t rhs) {
    switch (predicate) {
    case arith::CmpIPredicate::eq:
      return lhs == rhs;
    case arith::CmpIPredicate::ne:
      return lhs != rhs;
    case arith::CmpIPredicate::slt:
      return lhs < rhs;
    case arith::CmpIPredicate::sle:
      return lhs <= rhs;
    case arith::CmpIPredicate::sgt:
      return lhs > rhs;
    case arith::CmpIPredicate::sge:
      return lhs >= rhs;
    case arith::CmpIPredicate::ult:
      return (uint64_t)lhs < (uint64_t)rhs;
    case arith::CmpIPredicate::ule:
      return (uint64_t)lhs <= (uint64_t)rhs;
    case arith::CmpIPredicate::ugt:
      return (uint64_t)lhs > (uint64_t)rhs;
    case arith::CmpIPredicate::uge:
      return (uint64_t)lhs >= (uint64_t)rhs;
    default:
      break;
    }
    llvm_unreachable("unknown comparison predicate");
  }
};

class SelectOpIntegerInfoVisitor
    : public OpIntegerInfoVisitor<arith::SelectOp> {
public:
  using OpIntegerInfoVisitor<arith::SelectOp>::OpIntegerInfoVisitor;

  virtual IntegerInfo
  getIntegerInfo(arith::SelectOp op,
                 ArrayRef<const Lattice<IntegerInfo> *> operands) override {
    auto condition = operands[0]->getValue();
    auto lhs = operands[1]->getValue();
    auto rhs = operands[2]->getValue();
    int64_t divisibility = 1;
    std::optional<int64_t> constant;
    if (condition.hasConstant()) {
      if (condition.getConstant() == 0) {
        divisibility = rhs.getDivisibility();
        constant = rhs.getConstant();
      } else {
        divisibility = lhs.getDivisibility();
        constant = lhs.getConstant();
      }
    } else {
      divisibility = std::min(lhs.getDivisibility(), rhs.getDivisibility());
    }
    if (lhs.hasConstant() && rhs.hasConstant() &&
        lhs.getConstant() == rhs.getConstant())
      constant = lhs.getConstant();
    return IntegerInfo(divisibility, constant);
  }
};

template <typename OpT>
class BitOpIntegerInfoVisitor : public BinOpIntegerInfoVisitor<OpT> {
public:
  virtual int64_t getDivisibility(OpT op, //
                                  const IntegerInfo &lhs,
                                  const IntegerInfo &rhs) override {
    std::optional<int64_t> constant = getConstant(op, lhs, rhs);
    if (constant.has_value())
      return constant.value() == 0 ? gpd<int64_t>(0) : 1;
    return 1;
  }

  virtual std::optional<int64_t> getConstant(OpT op, //
                                             const IntegerInfo &lhs,
                                             const IntegerInfo &rhs) override {
    if (lhs.hasConstant() && rhs.hasConstant()) {
      if constexpr (std::is_same_v<OpT, arith::AndIOp>)
        return lhs.getConstant() & rhs.getConstant();
      if constexpr (std::is_same_v<OpT, arith::OrIOp>)
        return lhs.getConstant() | rhs.getConstant();
      if constexpr (std::is_same_v<OpT, arith::XOrIOp>)
        return lhs.getConstant() ^ rhs.getConstant();
    }
    return std::nullopt;
  }
};

class ShLIOpIntegerInfoVisitor : public BinOpIntegerInfoVisitor<arith::ShLIOp> {
public:
  using BinOpIntegerInfoVisitor<arith::ShLIOp>::BinOpIntegerInfoVisitor;

private:
  virtual int64_t getDivisibility(arith::ShLIOp op, //
                                  const IntegerInfo &lhs,
                                  const IntegerInfo &rhs) override {
    auto shift = rhs.hasConstant() ? rhs.getConstant() : rhs.getDivisibility();
    return mul<int64_t>(lhs.getDivisibility(), 1 << shift);
  }

  virtual std::optional<int64_t> getConstant(arith::ShLIOp op,
                                             const IntegerInfo &lhs,
                                             const IntegerInfo &rhs) override {
    if (lhs.hasConstant() && rhs.hasConstant())
      return lhs.getConstant() << rhs.getConstant();
    return std::nullopt;
  }
};

template <typename OpT>
class ShROpIntegerInfoVisitor : public BinOpIntegerInfoVisitor<OpT> {
public:
  using BinOpIntegerInfoVisitor<OpT>::BinOpIntegerInfoVisitor;

private:
  virtual int64_t getDivisibility(OpT op, //
                                  const IntegerInfo &lhs,
                                  const IntegerInfo &rhs) override {
    auto shift = rhs.hasConstant() ? rhs.getConstant() : rhs.getDivisibility();
    return std::max<int64_t>(lhs.getDivisibility() / (1 << shift), 1);
  }

  virtual std::optional<int64_t> getConstant(OpT op, //
                                             const IntegerInfo &lhs,
                                             const IntegerInfo &rhs) override {
    if (lhs.hasConstant() && rhs.hasConstant())
      return lhs.getConstant() >> rhs.getConstant();
    return std::nullopt;
  }
};

template <typename OpT>
class MaxMinOpIntegerInfoVisitor : public OpIntegerInfoVisitor<OpT> {
public:
  using OpIntegerInfoVisitor<OpT>::OpIntegerInfoVisitor;

  virtual IntegerInfo
  getIntegerInfo(OpT op,
                 ArrayRef<const Lattice<IntegerInfo> *> operands) override {
    auto lhs = operands[0]->getValue();
    auto rhs = operands[1]->getValue();
    int64_t divisibility = 1;
    std::optional<int64_t> constant;
    if (lhs.hasConstant() && rhs.hasConstant()) {
      if constexpr (std::is_same_v<OpT, arith::MaxSIOp> ||
                    std::is_same_v<OpT, arith::MaxUIOp>)
        constant = std::max(lhs.getConstant(), rhs.getConstant());
      if constexpr (std::is_same_v<OpT, arith::MinSIOp> ||
                    std::is_same_v<OpT, arith::MaxUIOp>)
        constant = std::min(lhs.getConstant(), rhs.getConstant());
      divisibility = gpd(constant.value());
    } else {
      divisibility = std::min(lhs.getDivisibility(), rhs.getDivisibility());
    }
    return IntegerInfo(divisibility, constant);
  }
};

class MakeMemRefOpIntegerInfoVisitor
    : public OpIntegerInfoVisitor<MakeMemRefOp> {
public:
  using OpIntegerInfoVisitor<MakeMemRefOp>::OpIntegerInfoVisitor;

  virtual IntegerInfo
  getIntegerInfo(MakeMemRefOp op,
                 ArrayRef<const Lattice<IntegerInfo> *> operands) override {
    return operands[0]->getValue();
  }
};

class MoveMemRefOpIntegerInfoVisitor
    : public BinOpIntegerInfoVisitor<MoveMemRefOp> {
public:
  using BinOpIntegerInfoVisitor<MoveMemRefOp>::BinOpIntegerInfoVisitor;

private:
  virtual int64_t getDivisibility(MoveMemRefOp op, //
                                  const IntegerInfo &lhs,
                                  const IntegerInfo &rhs) override {
    auto bitWidth = getIntOrFloatBitWidth(op.getSource().getType());
    auto rhsDivisibility =
        mul<int64_t>(rhs.getDivisibility(), ceilDiv<unsigned>(bitWidth, 8));
    return gcd(lhs.getDivisibility(), rhsDivisibility);
  }
};

class IntegerInfoAnalysis
    : public SparseForwardDataFlowAnalysis<Lattice<IntegerInfo>> {
public:
  IntegerInfoAnalysis(DataFlowSolver &solver)
      : SparseForwardDataFlowAnalysis<Lattice<IntegerInfo>>(solver) {
    // UnrealizedConversionCastOp - This is needed by KgpuToLLVM, to get
    // IntegerInfo when the graph is in the process of a PartialConversion,
    // where UnrealizedConversionCastOp may exist.
    visitors.append<CastOpIntegerInfoVisitor<arith::ExtSIOp>,
                    CastOpIntegerInfoVisitor<arith::ExtUIOp>,
                    CastOpIntegerInfoVisitor<arith::TruncIOp>,
                    CastOpIntegerInfoVisitor<arith::BitcastOp>,
                    CastOpIntegerInfoVisitor<UnrealizedConversionCastOp>>();
    visitors.append<ConstantOpIntegerInfoVisitor>();
    visitors.append<AddSubOpIntegerInfoVisitor<arith::AddIOp>,
                    AddSubOpIntegerInfoVisitor<arith::SubIOp>>();
    visitors.append<MulIOpIntegerInfoVisitor>();
    visitors.append<DivOpIntegerInfoVisitor<arith::DivSIOp>,
                    DivOpIntegerInfoVisitor<arith::DivUIOp>>();
    visitors.append<RemOpIntegerInfoVisitor<arith::RemSIOp>,
                    RemOpIntegerInfoVisitor<arith::RemUIOp>>();
    visitors.append<CmpIOpIntegerInfoVisitor>();
    visitors.append<BitOpIntegerInfoVisitor<arith::AndIOp>,
                    BitOpIntegerInfoVisitor<arith::OrIOp>,
                    BitOpIntegerInfoVisitor<arith::XOrIOp>>();
    visitors.append<SelectOpIntegerInfoVisitor>();
    visitors.append<ShLIOpIntegerInfoVisitor>();
    visitors.append<ShROpIntegerInfoVisitor<arith::ShRSIOp>,
                    ShROpIntegerInfoVisitor<arith::ShRUIOp>>();
    visitors.append<MaxMinOpIntegerInfoVisitor<arith::MaxSIOp>,
                    MaxMinOpIntegerInfoVisitor<arith::MaxUIOp>,
                    MaxMinOpIntegerInfoVisitor<arith::MinSIOp>,
                    MaxMinOpIntegerInfoVisitor<arith::MinUIOp>>();
    visitors.append<MakeMemRefOpIntegerInfoVisitor,
                    MoveMemRefOpIntegerInfoVisitor>();
  }

  using SparseForwardDataFlowAnalysis<Lattice<IntegerInfo>>::getLatticeElement;

  void visitOperation(Operation *op,
                      ArrayRef<const Lattice<IntegerInfo> *> operands,
                      ArrayRef<Lattice<IntegerInfo> *> results) override {
    // TODO: For sure not the right way to do this but why is scf::IfOp not
    // initialized otherwise?
    for (auto *operand : operands)
      if (operand->getValue().isEntryState())
        setToEntryState(const_cast<Lattice<IntegerInfo> *>(operand));

    auto info = visitors.apply(op, operands);
    if (info.isEntryState())
      return setAllToEntryStates(results);

    if (auto attr = op->getDiscardableAttr(divisibilityAttrName))
      info.setDivisibility(cast<IntegerAttr>(attr).getInt());

    // Join all lattice elements.
    for (auto *result : results)
      propagateIfChanged(result, result->join(info));
  }

  void visitForOpInductionVar(scf::ForOp forOp,
                              ArrayRef<Lattice<IntegerInfo> *> lattices) {
    auto lb = getLatticeElementFor(forOp, forOp.getLowerBound())->getValue();
    auto step = getLatticeElementFor(forOp, forOp.getStep())->getValue();
    auto iv = IntegerInfo(gcd(lb.getDivisibility(), step.getDivisibility()));
    (void)lattices[0]->join(iv);
  }

private:
  IntegerInfoVisitorList visitors;

  void setToEntryState(Lattice<IntegerInfo> *lattice) override {
    auto info = IntegerInfo::getPessimisticState(lattice->getPoint());
    propagateIfChanged(lattice, lattice->join(info));
  }

  void visitNonControlFlowArguments(Operation *op,
                                    const RegionSuccessor &successor,
                                    ArrayRef<Lattice<IntegerInfo> *> lattices,
                                    unsigned firstIndex) override {
    if (auto forOp = dyn_cast<scf::ForOp>(op)) {
      visitForOpInductionVar(forOp, lattices);
    } else {
      setAllToEntryStates(lattices.take_front(firstIndex));
      setAllToEntryStates(lattices.drop_front(
          firstIndex + successor.getSuccessorInputs().size()));
    }
  }
};

} // namespace

IntegerInfo IntegerInfo::getPessimisticState(FunctionOpInterface funcOp,
                                             unsigned argIndex) {
  auto attr = funcOp.getArgAttr(argIndex, divisibilityAttrName);
  if (auto intAttr = dyn_cast_or_null<IntegerAttr>(attr))
    return IntegerInfo(intAttr.getInt());
  return IntegerInfo();
}

IntegerInfo IntegerInfo::getPessimisticState(Value value) {
  IntegerInfo info(1);
  auto blockArg = dyn_cast<BlockArgument>(value);
  if (blockArg && blockArg.getOwner()->isEntryBlock()) {
    auto *op = blockArg.getOwner()->getParentOp();
    if (auto funcOp = dyn_cast<FunctionOpInterface>(op))
      info = getPessimisticState(funcOp, blockArg.getArgNumber());
    // Control flow operations are initialized with unknown state - the maximum
    // possible divisibility.
    else if (isa<RegionBranchOpInterface>(op))
      info.setDivisibility(gpd<int64_t>(0));
  } else if (auto *op = value.getDefiningOp()) {
    if (isa<RegionBranchOpInterface>(op))
      info.setDivisibility(gpd<int64_t>(0));
    // Other operations are conservatively initialized with the lowest possible
    // divisibility, unless been specified.
    else if (auto attr = op->getDiscardableAttr(divisibilityAttrName))
      info.setDivisibility(cast<IntegerAttr>(attr).getInt());
  }
  return info;
}

IntegerInfo IntegerInfo::join(const IntegerInfo &lhs, const IntegerInfo &rhs) {
  if (lhs.isEntryState())
    return rhs;
  if (rhs.isEntryState())
    return lhs;
  auto divisibility = gcd(lhs.getDivisibility(), rhs.getDivisibility());
  std::optional<int64_t> constant;
  if (lhs.hasConstant() && rhs.hasConstant() &&
      lhs.getConstant() == rhs.getConstant())
    constant = lhs.getConstant();
  return IntegerInfo(divisibility, constant);
}

void ModuleIntegerInfoAnalysis::initialize(FunctionOpInterface funcOp) {
  auto solver = createDataFlowSolver();
  auto *analysis = solver->load<IntegerInfoAnalysis>();
  if (failed(solver->initializeAndRun(funcOp)))
    return;

  auto *valueToInfo = getData(funcOp);
  auto updateInfo = [&](Value value) {
    auto info = analysis->getLatticeElement(value)->getValue();
    if (valueToInfo->contains(value))
      info = IntegerInfo::join(info, valueToInfo->lookup(value));
    (*valueToInfo)[value] = info;
  };

  funcOp.walk([&](Operation *op) {
    for (auto value : op->getResults())
      updateInfo(value);
  });
  funcOp.walk([&](Block *block) {
    for (auto value : block->getArguments())
      updateInfo(value);
  });
}

void ModuleIntegerInfoAnalysis::update(CallOpInterface caller,
                                       FunctionOpInterface callee) {
  auto i64Type = IntegerType::get(callee.getContext(), 64);
  auto *valueToInfo = getData(caller->getParentOfType<FunctionOpInterface>());
  for (auto it : llvm::enumerate(caller->getOperands())) {
    auto updateAttr = [&](StringRef name, int64_t newValue) {
      auto curValue = gpd<int64_t>(0);
      if (auto curAttr = callee.getArgAttrOfType<IntegerAttr>(it.index(), name))
        curValue = curAttr.getInt();
      auto newAttr = IntegerAttr::get(i64Type, gcd(curValue, newValue));
      callee.setArgAttr(it.index(), name, newAttr);
    };
    auto info = valueToInfo->lookup(it.value());
    updateAttr(divisibilityAttrName, info.getDivisibility());
  }
}
