// RUN: kapy-opt %s -kapy-combine | FileCheck %s

#glmem = #kapy.glmem<[?]>
module {
  // CHECK-LABEL: combine_matmul_op_as_add_op_lhs
  kapy.func @combine_matmul_op_as_add_op_lhs() -> tensor<128x128xf32> {
    %cst = arith.constant dense<1.000000e+00> : tensor<128x128xf32>
    %cst_0 = arith.constant dense<2.000000e+00> : tensor<128x128xf32>
    %cst_1 = arith.constant dense<0.000000e+00> : tensor<128x128xf32>
    %cst_2 = arith.constant dense<3.000000e+00> : tensor<128x128xf32>
    %0 = kapy.matmul %cst, %cst_0, %cst_1 : tensor<128x128xf32>, tensor<128x128xf32> -> tensor<128x128xf32>
    // CHECK-NOT: arith.addf
    %1 = arith.addf %0, %cst_2 : tensor<128x128xf32>
    kapy.return %1 : tensor<128x128xf32>
  }
  // CHECK-LABEL: combine_matmul_op_as_add_op_rhs
  kapy.func @combine_matmul_op_as_add_op_rhs() -> tensor<128x128xf32> {
    %cst = arith.constant dense<1.000000e+00> : tensor<128x128xf32>
    %cst_0 = arith.constant dense<2.000000e+00> : tensor<128x128xf32>
    %cst_1 = arith.constant dense<0.000000e+00> : tensor<128x128xf32>
    %cst_2 = arith.constant dense<3.000000e+00> : tensor<128x128xf32>
    %0 = kapy.matmul %cst, %cst_0, %cst_1 : tensor<128x128xf32>, tensor<128x128xf32> -> tensor<128x128xf32>
    // CHECK-NOT: arith.addf
    %1 = arith.addf %cst_2, %0 : tensor<128x128xf32>
    kapy.return %1 : tensor<128x128xf32>
  }
  // CHECK-LABEL: combine_matmul_op_multi_uses_failed
  kapy.func @combine_matmul_op_multi_uses_failed() -> (tensor<128x128xf32>, tensor<128x128xf32>) {
    %cst = arith.constant dense<1.000000e+00> : tensor<128x128xf32>
    %cst_0 = arith.constant dense<2.000000e+00> : tensor<128x128xf32>
    %cst_1 = arith.constant dense<0.000000e+00> : tensor<128x128xf32>
    %cst_2 = arith.constant dense<3.000000e+00> : tensor<128x128xf32>
    %cst_3 = arith.constant dense<4.000000e+00> : tensor<128x128xf32>
    %0 = kapy.matmul %cst, %cst_0, %cst_1 : tensor<128x128xf32>, tensor<128x128xf32> -> tensor<128x128xf32>
    // CHECK: arith.addf
    %1 = arith.addf %0, %cst_2 : tensor<128x128xf32>
    // CHECK: arith.addf
    %2 = arith.addf %0, %cst_3 : tensor<128x128xf32>
    kapy.return %1, %2 : tensor<128x128xf32>, tensor<128x128xf32>
  }
  // CHECK-LABEL: combine_two_move_memref_ops
  kapy.func @combine_two_move_memref_ops(%arg0: !kapy.memref<128xf32, #glmem>) -> !kapy.memref<128xf32, #glmem> {
    // CHECK: arith.constant 80
    %c16_i32 = arith.constant 16 : i32
    %c64_i32 = arith.constant 64 : i32
    // CHECK: kapy.move_memref
    %0 = kapy.move_memref %arg0, %c16_i32 : !kapy.memref<128xf32, #glmem>
    // CHECK-NOT: kapy.move_memref
    %1 = kapy.move_memref %0, %c64_i32 : !kapy.memref<128xf32, #glmem>
    kapy.return %1 : !kapy.memref<128xf32, #glmem>
  }
  // CHECK-LABEL: combine_select_op_and_if_op
  kapy.func @combine_select_op_and_if_op(%arg0: tensor<32x128xf32>, %arg1: !kapy.memref<32x128xf32, #glmem>, %arg2: i1) {
    %cst = arith.constant dense<0.000000e+00> : tensor<32x128xf32>
    %cst_0 = arith.constant dense<1.000000e+00> : tensor<32x128xf32>
    // CHECK-NOT: arith.select
    %0 = arith.select %arg2, %cst, %cst_0 : tensor<32x128xf32>
    scf.if %arg2 {
      kapy.store %arg1, %arg0 : !kapy.memref<32x128xf32, #glmem>, tensor<32x128xf32>
    }
    kapy.store %arg1, %0 : !kapy.memref<32x128xf32, #glmem>, tensor<32x128xf32>
    kapy.return
  }
  // CHECK-LABEL: combine_two_select_ops_and_if_op
  kapy.func @combine_two_select_ops_and_if_op(%arg0: i1, %arg1: i32, %arg2: i32, %arg3: f32, %arg4: f32) -> (i32, f32, i32) {
    // CHECK-NOT: arith.select
    %0 = arith.select %arg0, %arg1, %arg2 : i32
    %1 = arith.select %arg0, %arg3, %arg4 : f32
    %2 = scf.if %arg0 -> (i32) {
      %3 = arith.subi %arg1, %arg2 : i32
      scf.yield %3 : i32
    } else {
      scf.yield %arg1 : i32
    }
    // CHECK: kapy.return
    kapy.return %0, %1, %2 : i32, f32, i32
  }
}
