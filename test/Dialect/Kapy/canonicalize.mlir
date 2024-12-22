// RUN: kapy-opt %s -canonicalize | FileCheck %s

#gmem = #kapy.gmem<map = (d0, d1) -> (d0 * 128 + d1)>
module {
  // CHECK-LABEL: canonicalize_load_op
  kapy.func @canonicalize_load_op(%arg0: !kapy.memref<32x128xf32, #gmem>) -> (tensor<32x128xf32>, tensor<32x128xf32>) {
    %cst = arith.constant dense<true> : tensor<32x128xi1>
    %cst_1 = arith.constant dense<false> : tensor<32x128xi1>
    // CHECK-NOT: kapy.load
    %0 = kapy.load %arg0, %cst : !kapy.memref<32x128xf32, #gmem> -> tensor<32x128xf32>
    // CHECK: kapy.load
    // CHECK-SAME: is_volatile = true
    %1 = kapy.load %arg0, %cst {is_volatile = true} : !kapy.memref<32x128xf32, #gmem> -> tensor<32x128xf32>
    // CHECK-NOT: kapy.load
    %2 = kapy.load %arg0, %cst : !kapy.memref<32x128xf32, #gmem> -> tensor<32x128xf32>
    // CHECK: kapy.load %{{.*}} :
    %3 = kapy.load %arg0, %cst_1 : !kapy.memref<32x128xf32, #gmem> -> tensor<32x128xf32>
    kapy.return %2, %3 : tensor<32x128xf32>, tensor<32x128xf32>
  }
  // CHECK-LABEL: canonicalize_store_op
  kapy.func @canonicalize_store_op(%arg0: !kapy.memref<32x128xf32, #gmem>, %arg1: tensor<32x128xf32>) {
    %cst = arith.constant dense<true> : tensor<32x128xi1>
    %cst_0 = arith.constant dense<false> : tensor<32x128xi1>
    // CHECK-NOT: kapy.store
    kapy.store %arg0, %arg1, %cst : !kapy.memref<32x128xf32, #gmem>, tensor<32x128xf32>
    // CHECK: kapy.store %{{.*}}, %{{.*}} :
    kapy.store %arg0, %arg1, %cst_0 : !kapy.memref<32x128xf32, #gmem>, tensor<32x128xf32>
    kapy.return
  }
  // CHECK-LABEL: canonicalize_arange_op
  kapy.func @canonicalize_arange_op() -> (tensor<128x1xi32>, tensor<1xi32>) {
    // CHECK-NOT: kapy.arange
    %0 = kapy.arange {end = 1 : i32, start = 0 : i32} : tensor<1xi32>
    %1 = kapy.unsqueeze %0 {axis = 0 : i32} : tensor<1xi32> -> tensor<1x1xi32>
    // CHECK: arith.constant
    %2 = kapy.broadcast %1 {shape = array<i64: 128, 1>}: tensor<1x1xi32> -> tensor<128x1xi32>
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
    %3 = kapy.broadcast %arg1 {shape = array<i64: 128>} : tensor<1xf32> -> tensor<128xf32>
    // CHECK-NEXT: kapy.unsqueeze
    %4 = kapy.unsqueeze %3 {axis = 0 : i32} : tensor<128xf32> -> tensor<1x128xf32>
    // CHECK-NEXT: kapy.broadcast
    %5 = kapy.broadcast %4 {shape = array<i64: 32, 128>} : tensor<1x128xf32> -> tensor<32x128xf32>
    kapy.return %0, %2, %5 : tensor<128x1xf32>, tensor<1x128xf32>, tensor<32x128xf32>
  }
  // CHECK-LABEL: canonicalize_broadcast_op
  kapy.func @canonicalize_broadcast_op(%arg0: tensor<1x1x128xf32>, %arg1: f32) -> (tensor<64x128xf32>, tensor<64x32x128xf32>, tensor<128x128xf32>, tensor<1x1x128xf32>) {
    %cst = arith.constant dense<0.000000e+00> : tensor<1x128xf32>
    // CHECK: arith.constant
    %0 = kapy.broadcast %cst {shape = array<i64: 64, 128>} : tensor<1x128xf32> -> tensor<64x128xf32>
    %1 = kapy.broadcast %arg0 {shape = array<i64: 1, 32, 128>} : tensor<1x1x128xf32> -> tensor<1x32x128xf32>
    // CHECK-NEXT: kapy.broadcast
    %2 = kapy.broadcast %1 {shape = array<i64: 64, 32, 128>} : tensor<1x32x128xf32> -> tensor<64x32x128xf32>
    %3 = kapy.splat %arg1 : f32 -> tensor<1x128xf32>
    // CHECK-NEXT: kapy.splat
    %4 = kapy.broadcast %3 {shape = array<i64: 128, 128>} : tensor<1x128xf32> -> tensor<128x128xf32>
    // CHECK-NOT: kapy.broadcast
    %5 = kapy.broadcast %arg0 {shape = array<i64: 1, 1, 128>} : tensor<1x1x128xf32> -> tensor<1x1x128xf32>
    kapy.return %0, %2, %4, %5 : tensor<64x128xf32>, tensor<64x32x128xf32>, tensor<128x128xf32>, tensor<1x1x128xf32>
  }
  // CHECK-LABEL: canonicalize_permute_op
  kapy.func @canonicalize_permute_op(%arg0: f32, %arg1: tensor<32x128xf32>) -> (tensor<128x32xf32>, tensor<128x32xf32>, tensor<32x128xf32>) {
    // CHECK: arith.constant
    %cst = arith.constant dense<0.000000e+00> : tensor<32x128xf32>
    // CHECK-NOT: kapy.permute
    %0 = kapy.permute %cst {order = array<i32: 1, 0>} : tensor<32x128xf32> -> tensor<128x32xf32>
    %1 = kapy.splat %arg0 : f32 -> tensor<32x128xf32>
    // CHECK: kapy.splat
    %2 = kapy.permute %1 {order = array<i32: 1, 0>} : tensor<32x128xf32> -> tensor<128x32xf32>
    // CHECK-NOT: kapy.permute
    %3 = kapy.permute %arg1 {order = array<i32: 1, 0>} : tensor<32x128xf32> -> tensor<128x32xf32>
    %4 = kapy.permute %3 {order = array<i32: 1, 0>} : tensor<128x32xf32> -> tensor<32x128xf32>
    kapy.return %0, %2, %4 : tensor<128x32xf32>, tensor<128x32xf32>, tensor<32x128xf32>
  }
  // CHECK-LABEL: canonicalize_reshape_op
  kapy.func @canonicalize_reshape_op(%arg0: f32, %arg1: tensor<32x128xf32>) -> (tensor<64x64xf32>, tensor<128x32xf32>, tensor<32x128xf32>) {
    // CHECK: arith.constant
    %cst = arith.constant dense<0.000000e+00> : tensor<32x128xf32>
    // CHECK-NOT: kapy.reshape
    %0 = kapy.reshape %cst {shape = array<i64: 64, 64>} : tensor<32x128xf32> -> tensor<64x64xf32>
    %1 = kapy.splat %arg0 : f32 -> tensor<32x128xf32>
    // CHECK: kapy.splat
    %2 = kapy.reshape %1 {shape = array<i64: 128, 32>} : tensor<32x128xf32> -> tensor<128x32xf32>
    // CHECK-NOT: kapy.reshape
    %3 = kapy.reshape %arg1 {shape = array<i64: 128, 32>} : tensor<32x128xf32> -> tensor<128x32xf32>
    %4 = kapy.reshape %3 {shape = array<i64: 32, 128>} : tensor<128x32xf32> -> tensor<32x128xf32>
    // CHECK: kapy.return
    kapy.return %0, %2, %4 : tensor<64x64xf32>, tensor<128x32xf32>, tensor<32x128xf32>
  }
}
