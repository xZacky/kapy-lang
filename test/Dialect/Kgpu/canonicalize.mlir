// RUN: kapy-opt %s -canonicalize | FileCheck %s

#gmem = #kapy.gmem<map = (d0, d1) -> (d0 * 128 + d1)>
#regs = #kgpu.regs<map = (d0, d1, d2) -> (d2 + (d0 floordiv 4) * 4, d1 + (d0 mod 4) * 32)>
#regs1 = #kgpu.regs<map = (d0, d1, d2) -> (d2 + (d0 floordiv 4) * 4, d0 mod 4 + d1 * 4)>
#regs2 = #kgpu.regs<map = (d0, d1, d2) -> (d0 + d1 * 2 + d2 * 64)>
#regs3 = #kgpu.regs<map = (d0, d1, d2) -> (d1 + d2 * 32 + d0 * 64)>
#smem = #kgpu.smem<map = (d0, d1) -> (d0 * 128 + d1)>
module {
  // CHECK-LABEL: canonicalize_change_op
  kapy.func @canonicalize_change_op(%arg0: tensor<32x128xf32, #regs>) -> tensor<32x128xf32, #regs> {
    // CHECK-NOT: kgpu.change
    %0 = kgpu.change %arg0 : tensor<32x128xf32, #regs> to tensor<32x128xf32, #regs>
    kapy.return %0 : tensor<32x128xf32, #regs>
  }
  // CHECK-LABEL: combine_change_op_and_local_load_op
  kapy.func @combine_change_op_and_local_load_op(%arg0: !kapy.memref<32x128xf32, #smem>) -> tensor<32x128xf32, #regs1> {
    // CHECK: kgpu.local_load
    %0 = kgpu.local_load %arg0 : (!kapy.memref<32x128xf32, #smem>) -> tensor<32x128xf32, #regs>
    // CHECK-NOT: kgpu.change
    %1 = kgpu.change %0 : tensor<32x128xf32, #regs> to tensor<32x128xf32, #regs1>
    kapy.return %1 : tensor<32x128xf32, #regs1>
  }
  // CHECK-LABEL: combine_change_op_and_change_op
  kapy.func @combine_change_op_and_change_op(%arg0: tensor<32x128xf32, #regs>) -> tensor<32x128xf32, #regs> {
    // CHECK-NOT: kgpu.change
    %0 = kgpu.change %arg0 : tensor<32x128xf32, #regs> to tensor<32x128xf32, #regs1>
    %1 = kgpu.change %0 : tensor<32x128xf32, #regs1> to tensor<32x128xf32, #regs>
    kapy.return %1 : tensor<32x128xf32, #regs>
  }
  // CHECK-LABEL: combine_change_op_and_splat_op
  kapy.func @combine_change_op_and_splat_op(%arg0: f32) -> tensor<32x128xf32, #regs1> {
    // CHECK: kapy.splat
    %0 = kapy.splat %arg0 : f32 -> tensor<32x128xf32, #regs>
    // CHECK-NOT: kgpu.change
    %1 = kgpu.change %0 : tensor<32x128xf32, #regs> to tensor<32x128xf32, #regs1>
    kapy.return %1 : tensor<32x128xf32, #regs1>
  }
  // CHECK-LABEL: combine_change_op_and_arange_op
  kapy.func @combine_change_op_and_arange_op() -> tensor<128xi32, #regs2> {
    // CHECK: kapy.arange
    %0 = kapy.arange {end = 128 : i32, start = 0 : i32} : tensor<128xi32, #regs3>
    // CHECK-NOT: kgpu.change
    %1 = kgpu.change %0 : tensor<128xi32, #regs3> to tensor<128xi32, #regs2>
    kapy.return %1 : tensor<128xi32, #regs2>
  }
  // CHECK-LABEL: combine_change_op_and_constant_op
  kapy.func @combine_change_op_and_constant_op() -> tensor<32x128xf32, #regs1> {
    // CHECK: arith.constant
    %cst = arith.constant dense<0.000000e+00> : tensor<32x128xf32, #regs>
    // CHECK-NOT: kgpu.change
    %0 = kgpu.change %cst : tensor<32x128xf32, #regs> to tensor<32x128xf32, #regs1>
    // CHECK: kapy.return
    kapy.return %0 : tensor<32x128xf32, #regs1>
  }
  // CHECK-LABEL: verify_async_copy_op
  kapy.func @verify_async_copy_op(%arg0: !kapy.memref<32x128xf32, #gmem>, %arg1: !kapy.memref<32x128xf32, #smem>) -> !kgpu.tok<0> {
    %cst = arith.constant dense<true> : tensor<32x128xi1, #regs1>
    // CHECK: kgpu.async_copy
    %0 = kgpu.async_copy %arg0, %arg1, %cst : (!kapy.memref<32x128xf32, #gmem>, !kapy.memref<32x128xf32, #smem>, tensor<32x128xi1, #regs1>) -> !kgpu.tok<0>
    kapy.return %0 : !kgpu.tok<0>
  }
}
