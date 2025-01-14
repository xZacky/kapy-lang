//===- Affine.h -------------------------------------------------*- C++ -*-===//
//
// This file defines affine structures and functions to to analysis and compute
// affine maps in kapy.
//
//===----------------------------------------------------------------------===//

#ifndef KAPY_ANALYSIS_AFFINE_H
#define KAPY_ANALYSIS_AFFINE_H

#include "mlir/Analysis/FlatLinearValueConstraints.h"
#include "mlir/Analysis/Presburger/IntegerRelation.h"

namespace mlir {
namespace kapy {

/// Reimplement FlatAffineRelation from mlir::affine::FlatAffineRelation.
///
/// A FlatAffineRelation represents a set of ordered pairs (domain -> range)
/// where "domain" and "range" are tuples of variables. The relation is
/// represented as a FlatLinearConstraints with separation of dimension
/// variables into domain and range. The variables are stored as:
///
///   [domainVars, rangeVars, symbolVars, localVars, constant].
///
class FlatAffineRelation : public FlatLinearConstraints {
public:
  FlatAffineRelation(unsigned numReservedInequalities,
                     unsigned numReservedEqualities,
                     unsigned numReservedColumns, unsigned numDomainVars,
                     unsigned numRangeVars, unsigned numSymbolVars,
                     unsigned numLocalVars)
      : FlatLinearConstraints(numReservedInequalities, numReservedEqualities,
                              numReservedColumns, numDomainVars + numRangeVars,
                              numSymbolVars, numLocalVars),
        numDomainVars(numDomainVars), numRangeVars(numRangeVars) {}

  FlatAffineRelation(unsigned numDomainVars = 0, unsigned numRangeVars = 0,
                     unsigned numSymbolVars = 0, unsigned numLocalVars = 0)
      : FlatLinearConstraints(numDomainVars + numRangeVars, numSymbolVars,
                              numLocalVars),
        numDomainVars(numDomainVars), numRangeVars(numRangeVars) {}

  FlatAffineRelation(unsigned numDomainVars, unsigned numRangeVars,
                     FlatLinearConstraints &flc)
      : FlatLinearConstraints(flc), numDomainVars(numDomainVars),
        numRangeVars(numRangeVars) {
    space = presburger::PresburgerSpace::getRelationSpace(
        numDomainVars, numRangeVars, flc.getNumSymbolVars(),
        flc.getNumLocalVars());
  }

  FlatAffineRelation(unsigned numDomainVars, unsigned numRangeVars,
                     presburger::IntegerPolyhedron &flc)
      : FlatLinearConstraints(flc), numDomainVars(numDomainVars),
        numRangeVars(numRangeVars) {
    space = presburger::PresburgerSpace::getRelationSpace(
        numDomainVars, numRangeVars, flc.getNumSymbolVars(),
        flc.getNumLocalVars());
  }

  virtual Kind getKind() const override { return Kind::FlatAffineRelation; }

  static bool classof(const presburger::IntegerRelation *rel) {
    return rel->getKind() == Kind::FlatAffineRelation;
  }

  /// Get a set represented by a FlatLinearConstraints to the domain of this
  /// relation.
  FlatLinearConstraints getDomainSet() const;
  /// Get a set represented by a FlatLinearConstraints to the range of this
  /// relation.
  FlatLinearConstraints getRangeSet() const;

  inline unsigned getNumDomainVars() const { return numDomainVars; }
  inline unsigned getNumRangeVars() const { return numRangeVars; }

  /// Swap domain and range of this relation.
  void inverse();

  /// Insert `num` variables of the specified kind after the `pos` variable of
  /// that kind. The coefficient columns corresponding to the added variables
  /// are initialized to zero.
  void insertDomainVar(unsigned pos, unsigned num = 1);
  void insertRangeVar(unsigned pos, unsigned num = 1);

  /// Append `num` variables of the specified kind after the last variable of
  /// that kind. The coefficient columns corresponding to the added variables
  /// are initialized to zero.
  void appendDomainVar(unsigned num = 1);
  void appendRangeVar(unsigned num = 1);

  /// Remove variables in the column range `[varStart, varLimit)`, and copy any
  /// remaining valid data into place, updates member variables, and resize
  /// arrays as needed.
  virtual void removeVarRange(VarKind kind, unsigned varStart,
                              unsigned varLimit) override;
  using presburger::IntegerRelation::removeVarRange;

protected:
  // Number of dimension variables corresponding to domain variables.
  unsigned numDomainVars;
  // Number of dimension variables corresponding to range variables.
  unsigned numRangeVars;
};

/// Build a relation from the given AffineMap `map`, containing all pairs of the
/// form `operands -> result` that satisify `map`, `rel` is set to the relation
/// built. For example, given the AffineMap:
///
///   (d0, d1)[s0] -> (d0 + s0, d0 - s0)
///
/// the resulting relation formed is:
///
///   (d0, d1) -> (r0, r1)
///   [d0  d1  r0  r1  s0  const]
///     1   0  -1   0   1      0  = 0
///     0   1   0  -1  -1      0  = 0
///
/// Return failure if the AffineMap could not be flattened (i.e. semi-affine is
/// not yet handled).
LogicalResult getRelationFromMap(AffineMap map, FlatAffineRelation &rel);

} // namespace kapy
} // namespace mlir

#endif // KAPY_ANALYSIS_AFFINE_H
