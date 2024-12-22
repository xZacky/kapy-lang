// RUN: kapy-opt %s -convert-kapy-to-kgpu | FileCheck %s

#gmem = #kapy.gmem<map = (d0, d1)[s0] -> ((d0 + s0 * 64) * 128 + d1)>
#gmem1 = #kapy.gmem<map = (d0, d1)[s0, s1] -> ((d0 + s0 * 64) * 128 + d1 + s1 * 64)>
// CHECK: #[[REGS0:.*]] = #kgpu.regs<map = (d0, d1, d2) -> (d1 floordiv 32 + d2 floordiv 2 + d0 * 2, d1 mod 32 + (d2 mod 2) * 32)>
// CHECK: #[[REGS1:.*]] = #kgpu.regs<map = (d0, d1, d2) -> ((d0 floordiv 4) mod 4 + (d1 floordiv 16) * 4 + d2 * 8 + ((d0 floordiv 4) floordiv 4) * 32, d0 mod 4 + (d1 mod 16) * 4 + ((d0 mod 4) floordiv 4) * 64)>
// CHECK: #[[DOTLD0:.*]] = #kgpu.dotld<parent = #[[REGS1]], operand_index = 0, bit_width = 16>
// CHECK: #[[DOTLD1:.*]] = #kgpu.dotld<parent = #[[REGS1]], operand_index = 1, bit_width = 16>
// CHECK: module attributes
// CHECK-SAME: kgpu.num_warps = 4
// CHECK-SAME: kgpu.nvidia_cc = 80
module {
  kapy.func @matmul_kernel(%arg0: !kapy.ptr<1>, %arg1: !kapy.ptr<1>, %arg2: !kapy.ptr<1>) {
    %c0_i32 = arith.constant 0 : i32
    %c2_i32 = arith.constant 2 : i32
    %c1_i32 = arith.constant 1 : i32
    %c128_i32 = arith.constant 128 : i32
    %c8192_i32 = arith.constant 8192 : i32
    %0 = kapy.program_id x : i32
    %1 = kapy.program_id y : i32
    %2 = kapy.get_memref %arg0, [%0] : !kapy.ptr<1> -> !kapy.memref<64x64xf16, #gmem>
    %3 = kapy.get_memref %arg1, [%1] : !kapy.ptr<1> -> !kapy.memref<64x64xf16, #gmem>
    %cst = arith.constant dense<0.000000e+00> : tensor<64x64xf32>
    %4:3 = scf.for %arg3 = %c0_i32 to %c2_i32 step %c1_i32 iter_args(%arg4 = %2, %arg5 = %3, %arg6 = %cst) -> (!kapy.memref<64x64xf16, #gmem>, !kapy.memref<64x64xf16, #gmem>, tensor<64x64xf32>)  : i32 {
      %7 = kapy.load %arg4 : !kapy.memref<64x64xf16, #gmem> -> tensor<64x64xf16>
      %8 = kapy.load %arg5 : !kapy.memref<64x64xf16, #gmem> -> tensor<64x64xf16>
      // CHECK: kapy.dot
      // CHECK-SAME: #[[DOTLD0]]
      // CHECK-SAME: #[[DOTLD1]]
      // CHECK-SAME: #[[REGS1]]
      %9 = kapy.dot %7, %8, %arg6 : tensor<64x64xf16>, tensor<64x64xf16> -> tensor<64x64xf32>
      %10 = kapy.mov_memref %arg4, %c128_i32 : !kapy.memref<64x64xf16, #gmem>
      %11 = kapy.mov_memref %arg5, %c8192_i32 : !kapy.memref<64x64xf16, #gmem>
      scf.yield %10, %11, %9 : !kapy.memref<64x64xf16, #gmem>, !kapy.memref<64x64xf16, #gmem>, tensor<64x64xf32>
    }
    %5 = arith.truncf %4#2 : tensor<64x64xf32> to tensor<64x64xf16>
    %6 = kapy.get_memref %arg2, [%0, %1] : !kapy.ptr<1> -> !kapy.memref<64x64xf16, #gmem1>
    kapy.store %6, %5 : !kapy.memref<64x64xf16, #gmem1>, tensor<64x64xf16>
    kapy.return
  }
}
