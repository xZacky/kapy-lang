// RUN: kapy-opt %s -split-input-file -kgpu-optimize-matmul | FileCheck %s

// CHECK: #[[NVMMA:.*]] = #kgpu.nvmma<[4, 1], [2, 1]>
// CHECK: #[[MMOPD0:.*]] = #kgpu.mmopd<#nvmma, 0>
// CHECK: #[[MMOPD1:.*]] = #kgpu.mmopd<#nvmma, 1>
#frags = #kgpu.frags<[4, 1], [1, 1], [2, 16], [4, 4], 1>
#frags1 = #kgpu.frags<[4, 1], [1, 1], [1, 32], [4, 4], 1>
#mmopd = #kgpu.mmopd<#frags, 0>
#mmopd1 = #kgpu.mmopd<#frags, 1>
#mmopd2 = #kgpu.mmopd<#frags1, 1>
#mmopd3 = #kgpu.mmopd<#frags1, 0>
module attributes {kgpu.num_warps = 4 : i64, kgpu.nvidia_cc = 80 : i64} {
  kapy.func @optimize_chained_matmul_ops(%arg0: tensor<64x128xf16, #mmopd>, %arg1: tensor<128x64xf16, #mmopd1>, %arg2: tensor<64x128xf16, #mmopd2>) -> tensor<64x128xf32, #frags1> {
    %cst = arith.constant dense<0.000000e+00> : tensor<64x64xf32, #frags>
    %cst_0 = arith.constant dense<0.000000e+00> : tensor<64x128xf32, #frags1>
    %0 = kapy.matmul %arg0, %arg1, %cst : tensor<64x128xf16, #mmopd>, tensor<128x64xf16, #mmopd1> -> tensor<64x64xf32, #frags>
    %1 = arith.truncf %0 : tensor<64x64xf32, #frags> to tensor<64x64xf16, #frags>
    %2 = kgpu.change %1 : tensor<64x64xf16, #frags> to tensor<64x64xf16, #mmopd3>
    // CHECK: kapy.matmul
    // CHECK-SAME: #[[MMOPD0]]
    // CHECK-SAME: #[[MMOPD1]]
    // CHECK-SAME: #[[NVMMA]]
    %3 = kapy.matmul %2, %arg2, %cst_0 : tensor<64x64xf16, #mmopd3>, tensor<64x128xf16, #mmopd2> -> tensor<64x128xf32, #frags1>
    kapy.return %3 : tensor<64x128xf32, #frags1>
  }
}

// -----

#glmem = #kapy.glmem<[?, 1]>
#nvmma = #kgpu.nvmma<[2, 2], [2, 1]>
#frags = #kgpu.frags<[4, 1], [1, 1], [2, 16], [4, 4], 1>
#mmopd = #kgpu.mmopd<#nvmma, 0>
#mmopd1 = #kgpu.mmopd<#nvmma, 1>
module attributes {kgpu.num_warps = 4 : i64, kgpu.nvidia_cc = 80 : i64} {
  // CHECK-LABEL: move_change_op_next_to_load_op
  kapy.func @move_change_op_next_to_load_op(%arg0: !kapy.memref<64x64xi8, #glmem>, %arg1: !kapy.memref<64x64xf16, #glmem>, %arg2: tensor<64x64xf32, #nvmma>) -> tensor<64x64xf32, #nvmma> {
    %0 = kapy.load %arg0 : !kapy.memref<64x64xi8, #glmem> -> tensor<64x64xi8, #frags>
    %1 = kapy.load %arg1 : !kapy.memref<64x64xf16, #glmem> -> tensor<64x64xf16, #frags>
    // CHECK: kgpu.change
    // CHECK-NEXT: arith.bitcast
    // CHECK-NEXT: kapy.fptofp
    %2 = arith.bitcast %0 : tensor<64x64xi8, #frags> to tensor<64x64xf8E5M2, #frags>
    %3 = kapy.fptofp %2 : tensor<64x64xf8E5M2, #frags> to tensor<64x64xf16, #frags>
    %4 = kgpu.change %3 : tensor<64x64xf16, #frags> to tensor<64x64xf16, #mmopd>
    %5 = kgpu.change %1 : tensor<64x64xf16, #frags> to tensor<64x64xf16, #mmopd1>
    %6 = kapy.matmul %4, %5, %arg2 : tensor<64x64xf16, #mmopd>, tensor<64x64xf16, #mmopd1> -> tensor<64x64xf32, #nvmma>
    kapy.return %6 : tensor<64x64xf32, #nvmma>
  }
}
