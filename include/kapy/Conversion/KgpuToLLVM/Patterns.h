//===- Passes.h -------------------------------------------------*- C++ -*-===//
//
// This file defines functions to populate patterns.
//
//===----------------------------------------------------------------------===//

#ifndef KAPY_CONVERSION_KGPUTOLLVM_PATTERNS
#define KAPY_CONVERSION_KGPUTOLLVM_PATTERNS

#include "mlir/Conversion/LLVMCommon/TypeConverter.h"

namespace mlir {
namespace kapy {

void populateElementwiseOpToLLVMConversionPatterns(
    const LLVMTypeConverter &typeConverter, RewritePatternSet &patterns);

void populateSelectOpToLLVMConversionPattern(
    const LLVMTypeConverter &typeConverter, RewritePatternSet &patterns);

void populateMkGlobalOpToLLVMConversionPattern(
    const LLVMTypeConverter &typeConverter, RewritePatternSet &patterns);

void populateSvGlobalOpToLLVMConversionPattern(
    const LLVMTypeConverter &typeConverter, RewritePatternSet &patterns);

void populateLdGlobalOpToLLVMConversionPattern(
    const LLVMTypeConverter &typeConverter, RewritePatternSet &patterns);

void populateStGlobalOpToLLVMConversionPattern(
    const LLVMTypeConverter &typeConverter, RewritePatternSet &patterns);

void populateMkSharedOpToLLVMConversionPattern(
    const LLVMTypeConverter &typeConverter, RewritePatternSet &patterns);

void populateSvSharedOpToLLVMConversionPattern(
    const LLVMTypeConverter &typeConverter, RewritePatternSet &patterns);

void populateLdSharedOpToLLVMConversionPattern(
    const LLVMTypeConverter &typeConverter, RewritePatternSet &patterns);

void populateStSharedOpToLLVMConversionPattern(
    const LLVMTypeConverter &typeConverter, RewritePatternSet &patterns);

void populateLdMatrixOpToLLVMConversionPattern(
    const LLVMTypeConverter &typeConverter, RewritePatternSet &patterns);

void populateCpAsyncOpToLLVMConversionPatterns(
    const LLVMTypeConverter &typeConverter, RewritePatternSet &patterns);

void populateSplatLikeOpToLLVMConversionPatterns(
    const LLVMTypeConverter &typeConverter, RewritePatternSet &patterns);

void populateBroadcastOpToLLVMConversionPattern(
    const LLVMTypeConverter &typeConverter, RewritePatternSet &patterns);

void populateTransposeOpToLLVMConversionPattern(
    const LLVMTypeConverter &typeConverter, RewritePatternSet &patterns);

void populateArangeOpToLLVMConversionPattern(
    const LLVMTypeConverter &typeConverter, RewritePatternSet &patterns);

void populateMatmulOpToLLVMConversionPattern(
    const LLVMTypeConverter &typeConverter, RewritePatternSet &patterns);

void populateReduceOpToLLVMConversionPattern(
    const LLVMTypeConverter &typeConverter, RewritePatternSet &patterns);

void populateChangeOpToLLVMConversionPattern(
    const LLVMTypeConverter &typeConverter, RewritePatternSet &patterns);

void populateFuncOpToLLVMConversionPattern(
    const LLVMTypeConverter &typeConverter, RewritePatternSet &patterns);

void populateCallReturnOpToLLVMConversionPatterns(
    const LLVMTypeConverter &typeConverter, RewritePatternSet &patterns);

void populateParallelIdOpToLLVMConversionPatterns(
    const LLVMTypeConverter &typeConverter, RewritePatternSet &patterns);

} // namespace kapy
} // namespace mlir

#endif // KAPY_CONVERSION_KGPUTOLLVM_PATTERNS
