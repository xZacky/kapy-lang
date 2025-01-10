// RUN: kapy-opt %s -kapy-test-integer | FileCheck %s

#glmem = #kapy.glmem<[?, ?]>
module {
  kapy.func @matmul_kernel(%arg0: !kapy.ptr<1> {kapy.divisibility = 128 : i64}, %arg1: !kapy.ptr<1> {kapy.divisibility = 128 : i64}, %arg2: !kapy.ptr<1> {kapy.divisibility = 128 : i64}) {
    %c0_i32 = arith.constant 0 : i32
    %c1_i32 = arith.constant 1 : i32
    %c2_i32 = arith.constant 2 : i32
    %c64_i32 = arith.constant 64 : i32
    %c128_i32 = arith.constant 128 : i32
    %c8192_i32 = arith.constant 8192 : i32
    %0 = kapy.program_id x : i32
    %1 = kapy.program_id y : i32
    %2 = kapy.make_memref %arg0, [%c128_i32, %c128_i32], [%c128_i32, %c1_i32] : !kapy.ptr<1> -> !kapy.memref<64x64xf16, #glmem>
    %3 = arith.muli %0, %c8192_i32 : i32
    %4 = kapy.move_memref %2, %3 : !kapy.memref<64x64xf16, #glmem>
    %5 = kapy.make_memref %arg1, [%c128_i32, %c128_i32], [%c128_i32, %c1_i32] : !kapy.ptr<1> -> !kapy.memref<64x64xf16, #glmem>
    %6 = arith.muli %1, %c64_i32 : i32
    %7 = kapy.move_memref %5, %6 : !kapy.memref<64x64xf16, #glmem>
    %cst = arith.constant dense<0.000000e+00> : tensor<64x64xf32>
    %8:3 = scf.for %arg3 = %c0_i32 to %c2_i32 step %c1_i32 iter_args(%arg4 = %cst, %arg5 = %4, %arg6 = %7) -> (tensor<64x64xf32>, !kapy.memref<64x64xf16, #glmem>, !kapy.memref<64x64xf16, #glmem>) : i32 {
      %15 = kapy.load %arg5 : !kapy.memref<64x64xf16, #glmem> -> tensor<64x64xf16>
      %16 = kapy.load %arg6 : !kapy.memref<64x64xf16, #glmem> -> tensor<64x64xf16>
      %17 = kapy.matmul %15, %16, %arg4 : tensor<64x64xf16>, tensor<64x64xf16> -> tensor<64x64xf32>
      %18 = kapy.move_memref %arg5, %c64_i32 : !kapy.memref<64x64xf16, #glmem>
      %19 = kapy.move_memref %arg6, %c8192_i32 : !kapy.memref<64x64xf16, #glmem>
      scf.yield %17, %18, %19 : tensor<64x64xf32>, !kapy.memref<64x64xf16, #glmem>, !kapy.memref<64x64xf16, #glmem>
    }
    %9 = arith.truncf %8#0 : tensor<64x64xf32> to tensor<64x64xf16>
    %10 = kapy.make_memref %arg2, [%c128_i32, %c128_i32], [%c128_i32, %c1_i32] : !kapy.ptr<1> -> !kapy.memref<64x64xf16, #glmem>
    %11 = arith.muli %0, %c8192_i32 : i32
    %12 = arith.muli %1, %c64_i32 : i32
    %13 = arith.addi %11, %12 : i32
    %14 = kapy.move_memref %10, %13 : !kapy.memref<64x64xf16, #glmem>
    kapy.store %14, %9 : !kapy.memref<64x64xf16, #glmem>, tensor<64x64xf16>
    kapy.return
  }
}
