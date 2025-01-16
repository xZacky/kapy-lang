// RUN: kapy-opt %s -canonicalize | FileCheck %s

module {
  // CHECK-LABEL: canonicalize_arange_op
  kapy.func @canonicalize_arange_op() -> (tensor<128x1xi32>, tensor<1xi32>) {
    // CHECK-NOT: kapy.arange
    %0 = kapy.arange {end = 1 : i32, start = 0 : i32} : tensor<1xi32>
    %1 = kapy.unsqueeze %0 {axis = 0 : i32} : tensor<1xi32> -> tensor<1x1xi32>
    // CHECK: arith.constant
    %2 = kapy.broadcast %1 : tensor<1x1xi32> -> tensor<128x1xi32>
    // CHECK-NEXT: arith.constant
    %3 = kapy.arange {end = 2 : i32, start = 1 : i32} : tensor<1xi32>
    kapy.return %2, %3 : tensor<128x1xi32>, tensor<1xi32>
  }
  // CHECK-LABEL: canonicalize_splat_op
  kapy.func @canonicalize_splat_op() -> tensor<32x128xf32> {
    %cst = arith.constant 0.000000e+00 : f32
    // CHECK: arith.constant
    // CHECK-NOT: kapy.splat
    %0 = kapy.splat %cst : f32 -> tensor<32x128xf32>
    // CHECK: kapy.return
    kapy.return %0 : tensor<32x128xf32>
  }
  // CHECK-LABEL: canonicalize_unsqueeze_op
  kapy.func @canonicalize_unsqueeze_op(%arg0: f32, %arg1: tensor<1xf32>) -> (tensor<128x1xf32>, tensor<1x128xf32>, tensor<32x128xf32>) {
    %cst = arith.constant dense<0.000000e+00> : tensor<128xf32>
    // CHECK: arith.constant
    %0 = kapy.unsqueeze %cst {axis = 1 : i32} : tensor<128xf32> -> tensor<128x1xf32>
    %1 = kapy.splat %arg0 : f32 -> tensor<128xf32>
    // CHECK-NEXT: kapy.splat
    %2 = kapy.unsqueeze %1 {axis = 0 : i32} : tensor<128xf32> -> tensor<1x128xf32>
    %3 = kapy.broadcast %arg1 : tensor<1xf32> -> tensor<128xf32>
    // CHECK-NEXT: kapy.unsqueeze
    %4 = kapy.unsqueeze %3 {axis = 0 : i32} : tensor<128xf32> -> tensor<1x128xf32>
    // CHECK-NEXT: kapy.broadcast
    %5 = kapy.broadcast %4 : tensor<1x128xf32> -> tensor<32x128xf32>
    kapy.return %0, %2, %5 : tensor<128x1xf32>, tensor<1x128xf32>, tensor<32x128xf32>
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
}
