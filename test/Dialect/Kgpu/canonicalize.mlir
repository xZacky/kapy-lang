// RUN: kapy-opt %s -canonicalize | FileCheck %s

#glmem = #kapy.glmem<[?, 1]>
#frags = #kgpu.frags<[1, 4], [1, 1], [1, 32], [1, 1], 1>
#frags1 = #kgpu.frags<[4, 1], [1, 1], [1, 32], [1, 4], 1>
#frags2 = #kgpu.frags<[4], [1], [32], [1], 0>
#frags3 = #kgpu.frags<[4], [1], [32], [4], 0>
#shmem = #kgpu.shmem<[128, 1], 32, 4>
module {
  // CHECK-LABEL: canonicalize_change_op
  kapy.func @canonicalize_change_op(%arg0: tensor<32x128xf32, #frags>) -> tensor<32x128xf32, #frags> {
    // CHECK-NOT: kgpu.change
    %0 = kgpu.change %arg0 : tensor<32x128xf32, #frags> to tensor<32x128xf32, #frags>
    kapy.return %0 : tensor<32x128xf32, #frags>
  }
  // CHECK-LABEL: combine_change_op_and_change_op
  kapy.func @combine_change_op_and_change_op(%arg0: tensor<32x128xf32, #frags>) -> tensor<32x128xf32, #frags> {
    // CHECK-NOT: kgpu.change
    %0 = kgpu.change %arg0 : tensor<32x128xf32, #frags> to tensor<32x128xf32, #frags1>
    %1 = kgpu.change %0 : tensor<32x128xf32, #frags1> to tensor<32x128xf32, #frags>
    kapy.return %1 : tensor<32x128xf32, #frags>
  }
  // CHECK-LABEL: combine_change_op_and_constant_op
  kapy.func @combine_change_op_and_constant_op() -> tensor<32x128xf32, #frags1> {
    // CHECK: arith.constant
    %cst = arith.constant dense<0.000000e+00> : tensor<32x128xf32, #frags>
    // CHECK-NOT: kgpu.change
    %0 = kgpu.change %cst : tensor<32x128xf32, #frags> to tensor<32x128xf32, #frags1>
    // CHECK: kapy.return
    kapy.return %0 : tensor<32x128xf32, #frags1>
  }
  // CHECK-LABEL: combine_change_op_and_arange_op
  kapy.func @combine_change_op_and_arange_op() -> tensor<128xi32, #frags2> {
    // CHECK: kapy.arange
    %0 = kapy.arange {end = 128 : i32, start = 0 : i32} : tensor<128xi32, #frags3>
    // CHECK-NOT: kgpu.change
    %1 = kgpu.change %0 : tensor<128xi32, #frags3> to tensor<128xi32, #frags2>
    kapy.return %1 : tensor<128xi32, #frags2>
  }
  // CHECK-LABEL: combine_change_op_and_splat_op
  kapy.func @combine_change_op_and_splat_op(%arg0: f32) -> tensor<32x128xf32, #frags1> {
    // CHECK: kapy.splat
    %0 = kapy.splat %arg0 : f32 -> tensor<32x128xf32, #frags>
    // CHECK-NOT: kgpu.change
    %1 = kgpu.change %0 : tensor<32x128xf32, #frags> to tensor<32x128xf32, #frags1>
    kapy.return %1 : tensor<32x128xf32, #frags1>
  }
  // CHECK-LABEL: combine_change_op_and_local_load_op
  kapy.func @combine_change_op_and_local_load_op(%arg0: !kapy.memref<32x128xf32, #shmem>) -> tensor<32x128xf32, #frags1> {
    // CHECK: kgpu.local_load
    %0 = kgpu.local_load %arg0 : (!kapy.memref<32x128xf32, #shmem>) -> tensor<32x128xf32, #frags>
    // CHECK-NOT: kgpu.change
    %1 = kgpu.change %0 : tensor<32x128xf32, #frags> to tensor<32x128xf32, #frags1>
    kapy.return %1 : tensor<32x128xf32, #frags1>
  }
  // CHECK-LABEL: verify_async_copy_op
  kapy.func @verify_async_copy_op(%arg0: !kapy.memref<32x128xf32, #glmem>, %arg1: !kapy.memref<32x128xf32, #shmem>) -> !kgpu.tok<0> {
    // CHECK: kgpu.async_copy
    %0 = kgpu.async_copy %arg0, %arg1 {layout = #frags1} : (!kapy.memref<32x128xf32, #glmem>, !kapy.memref<32x128xf32, #shmem>) -> !kgpu.tok<0>
    kapy.return %0 : !kgpu.tok<0>
  }
}
