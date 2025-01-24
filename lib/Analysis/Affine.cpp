//===- Affine.cpp -----------------------------------------------*- C++ -*-===//
//
// This file implements the affine structures and functions.
//
//===----------------------------------------------------------------------===//

#include "kapy/Analysis/Affine.h"

using namespace mlir;
using namespace mlir::kapy;
using namespace mlir::presburger;

FlatLinearConstraints FlatAffineRelation::getDomainSet() const {
  FlatLinearConstraints domainSet = *this;
  // Convert all range variables to local variables.
  domainSet.convertToLocal(VarKind::SetDim, getNumDomainVars(),
                           getNumDimVars());
  return domainSet;
}

FlatLinearConstraints FlatAffineRelation::getRangeSet() const {
  FlatLinearConstraints rangeSet = *this;
  // Convert all domain variables to local variables.
  rangeSet.convertToLocal(VarKind::SetDim, 0, getNumDomainVars());
  return rangeSet;
}

void FlatAffineRelation::inverse() {
  auto oldNumDomainVars = getNumDomainVars();
  auto oldNumRangeVars = getNumRangeVars();
  // Add new range vars.
  appendRangeVar(oldNumDomainVars);
  // Swap new vars with domain.
  for (unsigned i = 0; i < oldNumDomainVars; ++i)
    swapVar(i, oldNumDomainVars + oldNumRangeVars + i);
  // Remove the swapped domain.
  removeVarRange(0, oldNumDomainVars);
  // Set domain and range as inverse.
  numDomainVars = oldNumRangeVars;
  numRangeVars = oldNumDomainVars;
}

void FlatAffineRelation::insertDomainVar(unsigned pos, unsigned num) {
  assert(pos <= getNumDomainVars());
  insertDimVar(pos, num);
  numDomainVars += num;
}

void FlatAffineRelation::insertRangeVar(unsigned pos, unsigned num) {
  assert(pos <= getNumRangeVars());
  insertDimVar(getNumDomainVars() + pos, num);
  numRangeVars += num;
}

void FlatAffineRelation::appendDomainVar(unsigned num) {
  insertDimVar(getNumDomainVars(), num);
  numDomainVars += num;
}

void FlatAffineRelation::appendRangeVar(unsigned num) {
  insertDimVar(getNumDimVars(), num);
  numRangeVars += num;
}

void FlatAffineRelation::removeVarRange(VarKind kind, unsigned varStart,
                                        unsigned varLimit) {
  assert(varLimit <= getNumVarKind(kind));
  if (varStart >= varLimit)
    return;

  FlatLinearConstraints::removeVarRange(kind, varStart, varLimit);

  // If kind is not SetDim, domain and range don't need to be updated.
  if (kind != VarKind::SetDim)
    return;

  // Compute number of domain and range variables to remove. This is done by
  // intersecting the range of domain/range variables with range of variables
  // to remove.
  auto intersectDomainLHS = std::min(varLimit, getNumDomainVars());
  auto intersectDomainRHS = varStart;
  auto intersectRangeLHS = std::min(varLimit, getNumDimVars());
  auto intersectRangeRHS = std::max(varStart, getNumDomainVars());

  if (intersectDomainLHS > intersectDomainRHS)
    numDomainVars -= intersectDomainLHS - intersectDomainRHS;
  if (intersectRangeLHS > intersectRangeRHS)
    numRangeVars -= intersectRangeLHS - intersectRangeRHS;
}

LogicalResult kapy::getRelationFromMap(AffineMap map, FlatAffineRelation &rel) {
  // Get the flattened affine expressions.
  std::vector<SmallVector<int64_t, 8>> flatExprs;
  FlatLinearConstraints flc;
  if (failed(getFlattenedAffineExprs(map, &flatExprs, &flc)))
    return failure();

  auto oldNumDims = flc.getNumDimVars();
  auto oldNumCols = flc.getNumCols();
  auto numRangeVars = map.getNumResults();
  auto numDomainVars = map.getNumDims();

  // Add range as the new expressions.
  flc.appendDimVar(numRangeVars);

  // Add equalities between domain and range.
  SmallVector<int64_t, 8> eq(flc.getNumCols());
  for (unsigned i = 0; i < map.getNumResults(); ++i) {
    // Zero fill.
    std::fill(eq.begin(), eq.end(), 0);
    // Fill equality.
    for (unsigned j = 0; j < oldNumDims; ++j)
      eq[j] = flatExprs[i][j];
    for (unsigned j = oldNumDims; j < oldNumCols; ++j)
      eq[j + numRangeVars] = flatExprs[i][j];
    // Set this dimension to -1 to equate lhs and rhs and add equality.
    eq[numDomainVars + i] = -1;
    flc.addEquality(eq);
  }

  // Create relation and return success.
  rel = FlatAffineRelation(numDomainVars, numRangeVars, flc);
  return success();
}
