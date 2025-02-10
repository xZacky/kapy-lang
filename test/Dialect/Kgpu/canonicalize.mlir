// RUN: kapy-opt %s -canonicalize | FileCheck %s

#strided2d = #kapy.strided2d<[?, 1]>
#fragments = #kgpu.fragments<[1, 4], [1, 1], [1, 32], [1, 1], (0, 1)>
#fragments1 = #kgpu.fragments<[4, 1], [1, 1], [1, 32], [1, 4], (0, 1)>
#swizzling = #kgpu.swizzling<(4, 8)>
module {
  // CHECK-LABEL: canonicalize_change_op
  kapy.func @canonicalize_change_op(%arg0: tensor<32x128xf32, #fragments>) -> tensor<32x128xf32, #fragments> {
    // CHECK-NOT: kgpu.change
    %0 = kgpu.change %arg0 : tensor<32x128xf32, #fragments> to tensor<32x128xf32, #fragments>
    kapy.return %0 : tensor<32x128xf32, #fragments>
  }
  // CHECK-LABEL: combine_change_op_and_change_op
  kapy.func @combine_change_op_and_change_op(%arg0: tensor<32x128xf32, #fragments>) -> tensor<32x128xf32, #fragments> {
    // CHECK-NOT: kgpu.change
    %0 = kgpu.change %arg0 : tensor<32x128xf32, #fragments> to tensor<32x128xf32, #fragments1>
    %1 = kgpu.change %0 : tensor<32x128xf32, #fragments1> to tensor<32x128xf32, #fragments>
    kapy.return %1 : tensor<32x128xf32, #fragments>
  }
  // CHECK-LABEL: combine_change_op_and_constant_op
  kapy.func @combine_change_op_and_constant_op() -> tensor<32x128xf32, #fragments1> {
    // CHECK: arith.constant
    %cst = arith.constant dense<0.000000e+00> : tensor<32x128xf32, #fragments>
    // CHECK-NOT: kgpu.change
    %0 = kgpu.change %cst : tensor<32x128xf32, #fragments> to tensor<32x128xf32, #fragments1>
    // CHECK: kapy.return
    kapy.return %0 : tensor<32x128xf32, #fragments1>
  }
  // CHECK-LABEL: combine_change_op_and_splat_op
  kapy.func @combine_change_op_and_splat_op(%arg0: f32) -> tensor<32x128xf32, #fragments1> {
    // CHECK: kapy.splat
    %0 = kapy.splat %arg0 : f32 -> tensor<32x128xf32, #fragments>
    // CHECK-NOT: kgpu.change
    %1 = kgpu.change %0 : tensor<32x128xf32, #fragments> to tensor<32x128xf32, #fragments1>
    kapy.return %1 : tensor<32x128xf32, #fragments1>
  }
  // CHECK-LABEL: combine_change_op_and_load_shared_op
  kapy.func @combine_change_op_and_load_shared_op(%arg0: !kgpu.shared<32x128xf32, #swizzling>) -> tensor<32x128xf32, #fragments1> {
    %c0_i32 = arith.constant 0 : i32
    // CHECK: kgpu.load_shared
    %0 = kgpu.load_shared %arg0[%c0_i32, %c0_i32] : !kgpu.shared<32x128xf32, #swizzling> -> tensor<32x128xf32, #fragments>
    // CHECK-NOT: kgpu.change
    %1 = kgpu.change %0 : tensor<32x128xf32, #fragments> to tensor<32x128xf32, #fragments1>
    kapy.return %1 : tensor<32x128xf32, #fragments1>
  }
}
