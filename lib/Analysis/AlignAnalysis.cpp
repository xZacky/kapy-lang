//===- AlignAnalysis.cpp --------------------------------------*- C++ -*-===//
//
// This file implements classes and functions used by data flow analysis for
// alignment.
//
//===----------------------------------------------------------------------===//

#include "kapy/Analysis/AlignAnalysis.h"
#include "kapy/Analysis/AnalysisUtils.h"
#include "kapy/Dialect/Kapy/IR/Kapy.h"
#include "mlir/Analysis/DataFlow/SparseAnalysis.h"

using namespace mlir;
using namespace mlir::kapy;
using namespace mlir::dataflow;

/// Implementation of euclidean algorithm.
template <typename T> static T gcdImpl(T a, T b, T &x, T &y) {
  if (a == 0) {
    x = 0;
    y = 1;
    return b;
  }
  T x1, y1; // to store results of recursive call
  T g = gcdImpl(b % a, a, x1, y1);
  // update `x` and `y` using results of recursive call
  x = y1 - (b / a) * x1;
  y = x1;
  return g;
}

/// Greatest common divisor.
template <typename T> static T gcd(T a, T b) {
  static_assert(std::is_integral_v<T>);
  if (a == 0)
    return b;
  if (b == 0)
    return a;
  T x, y;
  return gcdImpl(a, b, x, y);
}

/// Greatest power of two divisor. If `x == 0`, return the greatest power of two
/// for type `I`.
template <typename T> static T gpd(T x) {
  static_assert(std::is_integral_v<T>);
  if (x == 0)
    return static_cast<T>(1) << (sizeof(T) * 8 - 2);
  return x & (~(x - 1));
}

/// If `a * b` overflows, return greatest the power of two for type `I`.
template <typename T> static T mul(T a, T b) {
  static_assert(std::is_integral_v<T>);
  T g = gpd<T>(0);
  return a > g / b ? g : a * b;
}

namespace {

class AlignInfoVisitor {
public:
  AlignInfoVisitor() = default;
  virtual ~AlignInfoVisitor() = default;

  virtual bool match(Operation *op) = 0;

  virtual AlignInfo
  getAlignInfo(Operation *op,
               ArrayRef<const Lattice<AlignInfo> *> operands) = 0;
};

class AlignInfoVisitorList {
public:
  template <typename... Ts, typename = std::enable_if_t<sizeof...(Ts) != 0>>
  void append() {
    (visitors.emplace_back(std::make_unique<Ts>()), ...);
  }

  AlignInfo apply(Operation *op,
                  ArrayRef<const Lattice<AlignInfo> *> operands) {
    for (auto &visitor : visitors)
      if (visitor->match(op))
        return visitor->getAlignInfo(op, operands);
    return AlignInfo();
  }

private:
  std::vector<std::unique_ptr<AlignInfoVisitor>> visitors;
};

template <typename OpT> class OpAlignInfoVisitor : public AlignInfoVisitor {
public:
  using AlignInfoVisitor::AlignInfoVisitor;

  virtual bool match(Operation *op) override { return isa<OpT>(op); }

  virtual AlignInfo
  getAlignInfo(Operation *op,
               ArrayRef<const Lattice<AlignInfo> *> operands) override {
    return getAlignInfo(cast<OpT>(op), operands);
  }

  virtual AlignInfo
  getAlignInfo(OpT op, //
               ArrayRef<const Lattice<AlignInfo> *> operands) = 0;
};

template <typename OpT>
class BinOpAlignInfoVisitor : public OpAlignInfoVisitor<OpT> {
public:
  using OpAlignInfoVisitor<OpT>::OpAlignInfoVisitor;

  virtual AlignInfo
  getAlignInfo(OpT op, //
               ArrayRef<const Lattice<AlignInfo> *> operands) override {
    assert(operands.size() == 2);
    auto lhs = operands[0]->getValue();
    auto rhs = operands[1]->getValue();
    int64_t alignment = 1;
    std::optional<int64_t> constant = getConstant(op, lhs, rhs);
    if (constant.has_value())
      alignment = gpd(constant.value());
    else
      alignment = getAlignment(op, lhs, rhs);
    return AlignInfo(alignment, constant);
  }

protected:
  virtual int64_t getAlignment(OpT op,               //
                               const AlignInfo &lhs, //
                               const AlignInfo &rhs) {
    return 1;
  }

  virtual std::optional<int64_t> getConstant(OpT op, //
                                             const AlignInfo &lhs,
                                             const AlignInfo &rhs) {
    return std::nullopt;
  }
};

template <typename OpT>
class CastOpAlignInfoVisitor : public OpAlignInfoVisitor<OpT> {
public:
  using OpAlignInfoVisitor<OpT>::OpAlignInfoVisitor;

  virtual AlignInfo
  getAlignInfo(OpT op, //
               ArrayRef<const Lattice<AlignInfo> *> operands) override {
    return operands[0]->getValue();
  }
};

class ConstantOpAlignInfoVisitor
    : public OpAlignInfoVisitor<arith::ConstantOp> {
public:
  using OpAlignInfoVisitor<arith::ConstantOp>::OpAlignInfoVisitor;

  virtual AlignInfo
  getAlignInfo(arith::ConstantOp op,
               ArrayRef<const Lattice<AlignInfo> *> operands) override {
    auto intAttr = dyn_cast<IntegerAttr>(op.getValue());
    auto boolAttr = dyn_cast<BoolAttr>(op.getValue());
    if (intAttr || boolAttr) {
      int64_t value;
      if (intAttr)
        value = intAttr.getInt();
      else
        value = boolAttr.getValue() ? 1 : 0;
      return AlignInfo(gpd(value), value);
    }
    return AlignInfo();
  }
};

template <typename OpT>
class AddSubOpAlignInfoVisitor : public BinOpAlignInfoVisitor<OpT> {
public:
  using BinOpAlignInfoVisitor<OpT>::BinOpAlignInfoVisitor;

private:
  virtual int64_t getAlignment(OpT op, //
                               const AlignInfo &lhs,
                               const AlignInfo &rhs) override {
    return gcd(lhs.getAlignment(), rhs.getAlignment());
  }

  virtual std::optional<int64_t> getConstant(OpT op, //
                                             const AlignInfo &lhs,
                                             const AlignInfo &rhs) override {
    if (lhs.hasConstant() && rhs.hasConstant()) {
      if constexpr (std::is_same_v<OpT, arith::AddIOp>)
        return lhs.getConstant() + rhs.getConstant();
      if constexpr (std::is_same_v<OpT, arith::SubIOp>)
        return lhs.getConstant() - rhs.getConstant();
    }
    return std::nullopt;
  }
};

class MulIOpAlignInfoVisitor : public BinOpAlignInfoVisitor<arith::MulIOp> {
public:
  using BinOpAlignInfoVisitor<arith::MulIOp>::BinOpAlignInfoVisitor;

private:
  virtual int64_t getAlignment(arith::MulIOp op, //
                               const AlignInfo &lhs,
                               const AlignInfo &rhs) override {
    return mul(lhs.getAlignment(), rhs.getAlignment());
  }

  virtual std::optional<int64_t> getConstant(arith::MulIOp op, //
                                             const AlignInfo &lhs,
                                             const AlignInfo &rhs) override {
    if (lhs.hasConstantEqualTo(0) || rhs.hasConstantEqualTo(0))
      return 0;
    if (lhs.hasConstant() && rhs.hasConstant())
      return mul(lhs.getConstant(), rhs.getConstant());
    return std::nullopt;
  }
};

template <typename OpT>
class DivOpAlignInfoVisitor : public BinOpAlignInfoVisitor<OpT> {
public:
  using BinOpAlignInfoVisitor<OpT>::BinOpAlignInfoVisitor;

private:
  virtual int64_t getAlignment(OpT op, //
                               const AlignInfo &lhs,
                               const AlignInfo &rhs) override {
    if (lhs.hasConstantEqualTo(0) || rhs.hasConstantEqualTo(1))
      return lhs.getAlignment();
    return 1;
  }

  virtual std::optional<int64_t> getConstant(OpT op, //
                                             const AlignInfo &lhs,
                                             const AlignInfo &rhs) override {
    if (lhs.hasConstantEqualTo(0))
      return 0;
    if (lhs.hasConstant() && rhs.hasConstant())
      return lhs.getConstant() / rhs.getConstant();
    return std::nullopt;
  }
};

template <typename OpT>
class RemOpAlignInfoVisitor : public BinOpAlignInfoVisitor<OpT> {
public:
  using BinOpAlignInfoVisitor<OpT>::BinOpAlignInfoVisitor;

private:
  virtual int64_t getAlignment(OpT op, //
                               const AlignInfo &lhs,
                               const AlignInfo &rhs) override {
    if (rhs.hasConstant() && lhs.getAlignment() % rhs.getConstant() == 0)
      return gpd<int64_t>(0);
    return gcd(lhs.getAlignment(), rhs.getAlignment());
  }

  virtual std::optional<int64_t> getConstant(OpT op, //
                                             const AlignInfo &lhs,
                                             const AlignInfo &rhs) override {
    if (rhs.hasConstantEqualTo(1))
      return 0;
    if (lhs.hasConstant() && rhs.hasConstant())
      return lhs.getConstant() % rhs.getConstant();
    return std::nullopt;
  }
};

class CmpIOpAlignInfoVisitor : public OpAlignInfoVisitor<arith::CmpIOp> {
public:
  using OpAlignInfoVisitor<arith::CmpIOp>::OpAlignInfoVisitor;

  virtual AlignInfo
  getAlignInfo(arith::CmpIOp op,
               ArrayRef<const Lattice<AlignInfo> *> operands) override {
    assert(operands.size() == 2);
    auto lhs = operands[0]->getValue();
    auto rhs = operands[1]->getValue();
    int64_t alignment = 1;
    std::optional<int64_t> constant;
    if (lhs.hasConstant() && rhs.hasConstant()) {
      auto predicate = op.getPredicate();
      constant = cmp(predicate, lhs.getConstant(), rhs.getConstant()) ? 1 : 0;
      alignment = gpd(constant.value());
    }
    return AlignInfo(alignment, constant);
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
  static bool cmp(arith::CmpIPredicate predicate, int64_t lhs, int64_t rhs) {
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

class SelectOpAlignInfoVisitor : public OpAlignInfoVisitor<arith::SelectOp> {
public:
  using OpAlignInfoVisitor<arith::SelectOp>::OpAlignInfoVisitor;

  virtual AlignInfo
  getAlignInfo(arith::SelectOp op,
               ArrayRef<const Lattice<AlignInfo> *> operands) override {
    auto condition = operands[0]->getValue();
    auto lhs = operands[1]->getValue();
    auto rhs = operands[2]->getValue();
    int64_t alignment = 1;
    std::optional<int64_t> constant;
    if (condition.hasConstant()) {
      if (condition.getConstant() == 0) {
        alignment = rhs.getAlignment();
        constant = rhs.getConstant();
      } else {
        alignment = lhs.getAlignment();
        constant = lhs.getConstant();
      }
    } else {
      alignment = std::min(lhs.getAlignment(), rhs.getAlignment());
    }
    if (lhs.hasConstant() && rhs.hasConstant() &&
        lhs.getConstant() == rhs.getConstant())
      constant = lhs.getConstant();
    return AlignInfo(alignment, constant);
  }
};

template <typename OpT>
class BitOpAlignInfoVisitor : public BinOpAlignInfoVisitor<OpT> {
public:
  using BinOpAlignInfoVisitor<OpT>::BinOpAlignInfoVisitor;

private:
  virtual std::optional<int64_t> getConstant(OpT op, //
                                             const AlignInfo &lhs,
                                             const AlignInfo &rhs) override {
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

class ShLIOpAlignInfoVisitor : public BinOpAlignInfoVisitor<arith::ShLIOp> {
public:
  using BinOpAlignInfoVisitor<arith::ShLIOp>::BinOpAlignInfoVisitor;

private:
  virtual int64_t getAlignment(arith::ShLIOp op, //
                               const AlignInfo &lhs,
                               const AlignInfo &rhs) override {
    auto shift = rhs.hasConstant() ? rhs.getConstant() : rhs.getAlignment();
    return mul<int64_t>(lhs.getAlignment(), 1 << shift);
  }

  virtual std::optional<int64_t> getConstant(arith::ShLIOp op,
                                             const AlignInfo &lhs,
                                             const AlignInfo &rhs) override {
    if (lhs.hasConstant() && rhs.hasConstant())
      return lhs.getConstant() << rhs.getConstant();
    return std::nullopt;
  }
};

template <typename OpT>
class ShROpAlignInfoVisitor : public BinOpAlignInfoVisitor<OpT> {
public:
  using BinOpAlignInfoVisitor<OpT>::BinOpAlignInfoVisitor;

private:
  virtual int64_t getAlignment(OpT op, //
                               const AlignInfo &lhs,
                               const AlignInfo &rhs) override {
    auto shift = rhs.hasConstant() ? rhs.getConstant() : rhs.getAlignment();
    return std::max<int64_t>(lhs.getAlignment() / (1 << shift), 1);
  }

  virtual std::optional<int64_t> getConstant(OpT op, const AlignInfo &lhs,
                                             const AlignInfo &rhs) override {
    if (lhs.hasConstant() && rhs.hasConstant())
      return lhs.getConstant() >> rhs.getConstant();
    return std::nullopt;
  }
};

template <typename OpT>
class MaxMinOpAlignInfoVisitor : public OpAlignInfoVisitor<OpT> {
public:
  using OpAlignInfoVisitor<OpT>::OpAlignInfoVisitor;

  virtual AlignInfo
  getAlignInfo(OpT op, //
               ArrayRef<const Lattice<AlignInfo> *> operands) override {
    auto lhs = operands[0]->getValue();
    auto rhs = operands[1]->getValue();
    int64_t alignment = 1;
    std::optional<int64_t> constant;
    if (lhs.hasConstant() && rhs.hasConstant()) {
      if constexpr (std::is_same_v<OpT, arith::MaxSIOp> ||
                    std::is_same_v<OpT, arith::MaxUIOp>)
        constant = std::max(lhs.getConstant(), rhs.getConstant());
      if constexpr (std::is_same_v<OpT, arith::MinSIOp> ||
                    std::is_same_v<OpT, arith::MaxUIOp>)
        constant = std::min(lhs.getConstant(), rhs.getConstant());
      alignment = gpd(constant.value());
    } else {
      alignment = std::min(lhs.getAlignment(), rhs.getAlignment());
    }
    return AlignInfo(alignment, constant);
  }
};

class MkGlobalOpAlignInfoVisitor : public OpAlignInfoVisitor<MkGlobalOp> {
public:
  using OpAlignInfoVisitor<MkGlobalOp>::OpAlignInfoVisitor;

  virtual AlignInfo
  getAlignInfo(MkGlobalOp op,
               ArrayRef<const Lattice<AlignInfo> *> operands) override {
    return operands[0]->getValue();
  }
};

class AlignAnalysis : public SparseForwardDataFlowAnalysis<Lattice<AlignInfo>> {
public:
  AlignAnalysis(DataFlowSolver &solver)
      : SparseForwardDataFlowAnalysis<Lattice<AlignInfo>>(solver) {
    visitors.append<CastOpAlignInfoVisitor<arith::ExtSIOp>,
                    CastOpAlignInfoVisitor<arith::ExtUIOp>,
                    CastOpAlignInfoVisitor<arith::TruncIOp>,
                    CastOpAlignInfoVisitor<arith::BitcastOp>>();
    visitors.append<ConstantOpAlignInfoVisitor>();
    visitors.append<AddSubOpAlignInfoVisitor<arith::AddIOp>,
                    AddSubOpAlignInfoVisitor<arith::SubIOp>>();
    visitors.append<MulIOpAlignInfoVisitor>();
    visitors.append<DivOpAlignInfoVisitor<arith::DivSIOp>,
                    DivOpAlignInfoVisitor<arith::DivUIOp>>();
    visitors.append<RemOpAlignInfoVisitor<arith::RemSIOp>,
                    RemOpAlignInfoVisitor<arith::RemUIOp>>();
    visitors.append<CmpIOpAlignInfoVisitor>();
    visitors.append<BitOpAlignInfoVisitor<arith::AndIOp>,
                    BitOpAlignInfoVisitor<arith::OrIOp>,
                    BitOpAlignInfoVisitor<arith::XOrIOp>>();
    visitors.append<SelectOpAlignInfoVisitor>();
    visitors.append<ShLIOpAlignInfoVisitor>();
    visitors.append<ShROpAlignInfoVisitor<arith::ShRSIOp>,
                    ShROpAlignInfoVisitor<arith::ShRUIOp>>();
    visitors.append<MaxMinOpAlignInfoVisitor<arith::MaxSIOp>,
                    MaxMinOpAlignInfoVisitor<arith::MaxUIOp>,
                    MaxMinOpAlignInfoVisitor<arith::MinSIOp>,
                    MaxMinOpAlignInfoVisitor<arith::MinUIOp>>();
    visitors.append<MkGlobalOpAlignInfoVisitor>();
  }

  using SparseForwardDataFlowAnalysis<Lattice<AlignInfo>>::getLatticeElement;

  void visitOperation(Operation *op,
                      ArrayRef<const Lattice<AlignInfo> *> operands,
                      ArrayRef<Lattice<AlignInfo> *> results) override {
    // TODO: For sure not the right way to do this but why is scf::IfOp not
    // initialized otherwise?
    for (auto *operand : operands)
      if (operand->getValue().isEntryState())
        setToEntryState(const_cast<Lattice<AlignInfo> *>(operand));

    auto info = visitors.apply(op, operands);
    if (info.isEntryState())
      return setAllToEntryStates(results);

    if (auto attr = op->getAttr(alignmentAttrName))
      info.setAlignment(cast<IntegerAttr>(attr).getInt());

    for (auto *result : results)
      propagateIfChanged(result, result->join(info));
  }

  void visitForOpInductionVar(scf::ForOp forOp,
                              ArrayRef<Lattice<AlignInfo> *> lattices) {
    auto lb = getLatticeElementFor(forOp, forOp.getLowerBound())->getValue();
    auto step = getLatticeElementFor(forOp, forOp.getStep())->getValue();
    auto iv = AlignInfo(gcd(lb.getAlignment(), step.getAlignment()));
    (void)lattices[0]->join(iv);
  }

private:
  AlignInfoVisitorList visitors;

  void setToEntryState(Lattice<AlignInfo> *lattice) override {
    auto info = AlignInfo::getPessimisticState(lattice->getPoint());
    propagateIfChanged(lattice, lattice->join(info));
  }

  void visitNonControlFlowArguments(Operation *op,
                                    const RegionSuccessor &successor,
                                    ArrayRef<Lattice<AlignInfo> *> lattices,
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

AlignInfo AlignInfo::getPessimisticState(FunctionOpInterface funcOp,
                                         unsigned argIndex) {
  auto attr = funcOp.getArgAttr(argIndex, alignmentAttrName);
  if (auto intAttr = dyn_cast_if_present<IntegerAttr>(attr))
    return AlignInfo(intAttr.getInt());
  return AlignInfo();
}

AlignInfo AlignInfo::getPessimisticState(Value value) {
  AlignInfo info(1);
  auto blockArg = dyn_cast<BlockArgument>(value);
  if (blockArg && blockArg.getOwner()->isEntryBlock()) {
    auto *op = blockArg.getOwner()->getParentOp();
    if (auto funcOp = dyn_cast<FunctionOpInterface>(op))
      info = getPessimisticState(funcOp, blockArg.getArgNumber());
    // Control flow operations are initialized with unknown state - the maximum
    // possible alignment.
    else if (isa<RegionBranchOpInterface>(op))
      info.setAlignment(gpd<int64_t>(0));
  } else if (auto *op = value.getDefiningOp()) {
    if (isa<RegionBranchOpInterface>(op))
      info.setAlignment(gpd<int64_t>(0));
    // Other operations are conservatively initialized with the lowest possible
    // alignment, unless been specified.
    else if (auto attr = op->getAttr(alignmentAttrName))
      info.setAlignment(cast<IntegerAttr>(attr).getInt());
  }
  return info;
}

AlignInfo AlignInfo::join(const AlignInfo &lhs, const AlignInfo &rhs) {
  if (lhs.isEntryState())
    return rhs;
  if (rhs.isEntryState())
    return lhs;
  auto alignment = gcd(lhs.getAlignment(), rhs.getAlignment());
  std::optional<int64_t> constant;
  if (lhs.hasConstant() && rhs.hasConstant() &&
      lhs.getConstant() == rhs.getConstant())
    constant = lhs.getConstant();
  return AlignInfo(alignment, constant);
}

void ModuleAlignAnalysis::initialize(FunctionOpInterface funcOp) {
  auto solver = createDataFlowSolver();
  auto *analysis = solver->load<AlignAnalysis>();
  if (failed(solver->initializeAndRun(funcOp)))
    return;

  auto *valueToInfo = getData(funcOp);
  auto updateInfo = [&](Value value) {
    auto info = analysis->getLatticeElement(value)->getValue();
    if (valueToInfo->contains(value))
      info = AlignInfo::join(info, valueToInfo->lookup(value));
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

void ModuleAlignAnalysis::update(CallOpInterface caller,
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
    updateAttr(alignmentAttrName, info.getAlignment());
  }
}
