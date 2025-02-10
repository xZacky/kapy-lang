// RUN: kapy-opt %s -canonicalize | FileCheck %s

module {
  // CHECK-LABEL: canonicalize_splat_op
  kapy.func @canonicalize_splat_op() -> tensor<32x128xf32> {
    %cst = arith.constant 0.000000e+00 : f32
    // CHECK: arith.constant
    // CHECK-NOT: kapy.splat
    %0 = kapy.splat %cst : f32 -> tensor<32x128xf32>
    // CHECK: kapy.return
    kapy.return %0 : tensor<32x128xf32>
  }
  // CHECK-LABEL: canonicalize_broadcast_op
  kapy.func @canonicalize_broadcast_op(%arg0: tensor<1x1xf32>, %arg1: f32) -> (tensor<64x128xf32>, tensor<32x128xf32>, tensor<128x128xf32>, tensor<1x1xf32>) {
    %cst = arith.constant dense<0.000000e+00> : tensor<1x128xf32>
    // CHECK: arith.constant
    %0 = kapy.broadcast %cst : tensor<1x128xf32> -> tensor<64x128xf32>
    %1 = kapy.broadcast %arg0 : tensor<1x1xf32> -> tensor<1x128xf32>
    // CHECK-NEXT: kapy.broadcast
    %2 = kapy.broadcast %1 : tensor<1x128xf32> -> tensor<32x128xf32>
    %3 = kapy.splat %arg1 : f32 -> tensor<1x128xf32>
    // CHECK-NEXT: kapy.splat
    %4 = kapy.broadcast %3 : tensor<1x128xf32> -> tensor<128x128xf32>
    // CHECK-NOT: kapy.broadcast
    %5 = kapy.broadcast %arg0 : tensor<1x1xf32> -> tensor<1x1xf32>
    kapy.return %0, %2, %4, %5 : tensor<64x128xf32>, tensor<32x128xf32>, tensor<128x128xf32>, tensor<1x1xf32>
  }
  // CHECK-LABEL: canonicalize_transpose_op
  kapy.func @canonicalize_transpose_op(%arg0: f32, %arg1: tensor<32x128xf32>) -> (tensor<128x32xf32>, tensor<128x32xf32>, tensor<32x128xf32>) {
    // CHECK: arith.constant
    %cst = arith.constant dense<0.000000e+00> : tensor<32x128xf32>
    // CHECK-NOT: kapy.transpose
    %0 = kapy.transpose %cst : tensor<32x128xf32> -> tensor<128x32xf32>
    %1 = kapy.splat %arg0 : f32 -> tensor<32x128xf32>
    // CHECK: kapy.splat
    %2 = kapy.transpose %1 : tensor<32x128xf32> -> tensor<128x32xf32>
    // CHECK-NOT: kapy.transpose
    %3 = kapy.transpose %arg1 : tensor<32x128xf32> -> tensor<128x32xf32>
    %4 = kapy.transpose %3 : tensor<128x32xf32> -> tensor<32x128xf32>
    kapy.return %0, %2, %4 : tensor<128x32xf32>, tensor<128x32xf32>, tensor<32x128xf32>
  }
  // CHECK-LABEL: verify_reduce_op
  kapy.func @verify_reduce_op(%arg0: tensor<32x128xf32>) -> tensor<32x1xf32> {
    // CHECK: kapy.reduce
    %0 = kapy.reduce %arg0 {axis = 1 : i32} lambda(%arg1: f32, %arg2: f32) {
      %1 = arith.addf %arg1, %arg2 : f32
      kapy.return %1 : f32
    } : tensor<32x128xf32> -> tensor<32x1xf32>
    kapy.return %0 : tensor<32x1xf32>
  }
}
