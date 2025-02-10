// RUN: kapy-opt %s -convert-kapy-to-kgpu | FileCheck %s

// CHECK: #[[A:.*]] = #kgpu.fragments<[2, 2], [1, 1], [1, 32], [1, 1], (0, 1)>
// CHECK: #[[B:.*]] = #kgpu.fragments<[4, 1], [1, 1], [2, 16], [4, 4], (0, 1)>
// CHECK: #[[C:.*]] = #kgpu.fragments<[4, 1], [1, 1], [2, 16], [4, 64], (0, 1)>
// CHECK: #[[D:.*]] = #kgpu.fragments<[4, 1], [1, 1], [2, 16], [64, 4], (0, 1)>
// CHECK: module attributes
// CHECK-SAME: kgpu.num_warps = 4
// CHECK-SAME: kgpu.nvidia_cc = 80
#strided2d = #kapy.strided2d<[?, ?]>
module {
  kapy.func @matmul_kernel(%arg0: i64 {kapy.divisibility = 128 : i64}, %arg1: i64 {kapy.divisibility = 128 : i64}, %arg2: i64 {kapy.divisibility = 128 : i64}) {
    %c0_i32 = arith.constant 0 : i32
    %c1_i32 = arith.constant 1 : i32
    %c2_i32 = arith.constant 2 : i32
    %c14_i32 = arith.constant 14 : i32
    %c64_i32 = arith.constant 64 : i32
    %c1024_i32 = arith.constant 1024 : i32
    %cst = arith.constant dense<0.0> : tensor<64x64xf16>
    %0 = kapy.program_id x : i32
    %1 = kapy.program_id y : i32
    %2 = arith.muli %0, %c64_i32 : i32
    %3 = arith.muli %1, %c64_i32 : i32
    %4 = kapy.make_global %arg0[%c1024_i32, %c1024_i32][%c1024_i32, %c1_i32] : !kapy.global<?x?xf16, #strided2d>
    %5 = kapy.make_global %arg1[%c1024_i32, %c1024_i32][%c1024_i32, %c1_i32] : !kapy.global<?x?xf16, #strided2d>
    %6 = kapy.load_global cp_async %4[%2, %c0_i32], %cst : !kapy.global<?x?xf16, #strided2d> -> tensor<64x64xf16>
    %7 = kapy.load_global cp_async %5[%c0_i32, %3], %cst : !kapy.global<?x?xf16, #strided2d> -> tensor<64x64xf16>
    %8 = kapy.load_global cp_async %4[%2, %c0_i32], %cst : !kapy.global<?x?xf16, #strided2d> -> tensor<64x64xf16>
    %9 = kapy.load_global cp_async %5[%c0_i32, %3], %cst : !kapy.global<?x?xf16, #strided2d> -> tensor<64x64xf16>
    %10:5 = scf.for %arg3 = %c0_i32 to %c14_i32 step %c1_i32 iter_args(%arg4 = %cst, %arg5 = %6, %arg6 = %7, %arg7 = %8, %arg8 = %9) -> (tensor<64x64xf16>, tensor<64x64xf16>, tensor<64x64xf16>, tensor<64x64xf16>, tensor<64x64xf16>) : i32 {
      kapy.cp_async_wait {num_pending = 2 : i32}
      %11 = kapy.matmul mma_m16n8k8_f16 %arg5, %arg6, %arg4 : tensor<64x64xf16>, tensor<64x64xf16> -> tensor<64x64xf16>
      %12 = arith.addi %arg3, %c2_i32 : i32
      %13 = arith.muli %12, %c64_i32 : i32
      %14 = kapy.load_global cp_async %4[%2, %13], %cst : !kapy.global<?x?xf16, #strided2d> -> tensor<64x64xf16>
      %15 = kapy.load_global cp_async %5[%13, %3], %cst : !kapy.global<?x?xf16, #strided2d> -> tensor<64x64xf16>
      scf.yield %11, %arg7, %arg8, %14, %15 : tensor<64x64xf16>, tensor<64x64xf16>, tensor<64x64xf16>, tensor<64x64xf16>, tensor<64x64xf16>
    }
    kapy.cp_async_wait {num_pending = 2 : i32}
    %14 = kapy.matmul fma %10#1, %10#2, %10#0 : tensor<64x64xf16>, tensor<64x64xf16> -> tensor<64x64xf16>
    kapy.cp_async_wait {num_pending = 2 : i32}
    %15 = kapy.matmul fma %10#3, %10#4, %14 : tensor<64x64xf16>, tensor<64x64xf16> -> tensor<64x64xf16>
    %16 = kapy.make_global %arg2[%c1024_i32, %c1024_i32][%c1024_i32, %c1_i32] : !kapy.global<?x?xf16, #strided2d>
    kapy.store_global %16[%2, %3], %15 : !kapy.global<?x?xf16, #strided2d>, tensor<64x64xf16>
    kapy.return
  }
}
