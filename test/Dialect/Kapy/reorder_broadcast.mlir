// RUN: kapy-opt %s -kapy-reorder-broadcast | FileCheck %s

module {
  // CHECK-LABEL: move_splat_op_after_elementwise_op
  kapy.func @move_splat_op_after_elementwise_op(%arg0: f32) -> tensor<128x128xf32> {
    %cst = arith.constant dense<1.000000e+00> : tensor<128x128xf32>
    // CHECK: arith.addf
    %0 = kapy.splat %arg0 : f32 -> tensor<128x128xf32>
    // CHECK-NEXT: kapy.splat
    %1 = arith.addf %cst, %0 : tensor<128x128xf32>
    kapy.return %1 : tensor<128x128xf32>
  }
  // CHECK-LABEL: move_broadcast_op_after_elementwise_op
  kapy.func @move_broadcast_op_after_elementwise_op(%arg0: tensor<128x1xf32>) -> (tensor<128x128xf32>, tensor<128x32xf32>) {
    %cst = arith.constant dense<1.000000e+00> : tensor<128x32xf32>
    // CHECK: math.absf
    %0 = kapy.broadcast %arg0 {shape = array<i64: 128, 128>} : tensor<128x1xf32> -> tensor<128x128xf32>
    // CHECK-NEXT: kapy.broadcast
    %1 = math.absf %0 : tensor<128x128xf32>
    // CHECK: arith.addf
    %2 = kapy.broadcast %arg0 {shape = array<i64: 128, 32>} : tensor<128x1xf32> -> tensor<128x32xf32>
    // CHECK-NEXT: kapy.broadcast
    %3 = arith.addf %2, %cst : tensor<128x32xf32>
    kapy.return %1, %3 : tensor<128x128xf32>, tensor<128x32xf32>
  }
}
