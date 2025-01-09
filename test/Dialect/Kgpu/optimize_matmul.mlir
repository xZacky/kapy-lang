// RUN: kapy-opt %s -split-input-file -kgpu-optimize-matmul | FileCheck %s

// CHECK: #[[NVMMA:.*]] = #kgpu.nvmma<[4, 1], [2, 1]>
// CHECK: #[[MMOPD0:.*]] = #kgpu.mmopd<#nvmma, 0, 16>
// CHECK: #[[MMOPD1:.*]] = #kgpu.mmopd<#nvmma, 1, 16>
#regis = #kgpu.regis<[4, 1], [1, 1], [2, 16], [4, 4], (0, 1)>
#regis1 = #kgpu.regis<[4, 1], [1, 1], [1, 32], [4, 4], (0, 1)>
#mmopd = #kgpu.mmopd<#regis, 0, 16>
#mmopd1 = #kgpu.mmopd<#regis, 1, 16>
#mmopd2 = #kgpu.mmopd<#regis1, 1, 16>
#mmopd3 = #kgpu.mmopd<#regis1, 0, 16>
module attributes {kgpu.num_warps = 4 : i64, kgpu.nvidia_cc = 80 : i64} {
  kapy.func @optimize_chained_matmul_ops(%arg0: tensor<64x128xf16, #mmopd>, %arg1: tensor<128x64xf16, #mmopd1>, %arg2: tensor<64x128xf16, #mmopd2>) -> tensor<64x128xf32, #regis1> {
    %cst = arith.constant dense<0.000000e+00> : tensor<64x64xf32, #regis>
    %cst_0 = arith.constant dense<0.000000e+00> : tensor<64x128xf32, #regis1>
    %0 = kapy.matmul %arg0, %arg1, %cst : tensor<64x128xf16, #mmopd>, tensor<128x64xf16, #mmopd1> -> tensor<64x64xf32, #regis>
    %1 = arith.truncf %0 : tensor<64x64xf32, #regis> to tensor<64x64xf16, #regis>
    %2 = kgpu.change %1 : tensor<64x64xf16, #regis> to tensor<64x64xf16, #mmopd3>
    // CHECK: kapy.matmul
    // CHECK-SAME: #[[MMOPD0]]
    // CHECK-SAME: #[[MMOPD1]]
    // CHECK-SAME: #[[NVMMA]]
    %3 = kapy.matmul %2, %arg2, %cst_0 : tensor<64x64xf16, #mmopd3>, tensor<64x128xf16, #mmopd2> -> tensor<64x128xf32, #regis1>
    kapy.return %3 : tensor<64x128xf32, #regis1>
  }
}

// -----

#glmem = #kapy.glmem<[?, 1]>
#nvmma = #kgpu.nvmma<[2, 2], [2, 1]>
#regis = #kgpu.regis<[4, 1], [1, 1], [2, 16], [4, 4], (0, 1)>
#mmopd = #kgpu.mmopd<#nvmma, 0, 8>
#mmopd1 = #kgpu.mmopd<#nvmma, 1, 8>
module attributes {kgpu.num_warps = 4 : i64, kgpu.nvidia_cc = 80 : i64} {
  // CHECK-LABEL: move_change_op_next_to_load_op
  kapy.func @move_change_op_next_to_load_op(%arg0: !kapy.memref<64x64xi8, #glmem>, %arg1: !kapy.memref<64x64xf16, #glmem>, %arg2: tensor<64x64xf32, #nvmma>) -> tensor<64x64xf32, #nvmma> {
    %0 = kapy.load %arg0 : !kapy.memref<64x64xi8, #glmem> -> tensor<64x64xi8, #regis>
    %1 = kapy.load %arg1 : !kapy.memref<64x64xf16, #glmem> -> tensor<64x64xf16, #regis>
    // CHECK: kgpu.change
    // CHECK-NEXT: arith.bitcast
    // CHECK-NEXT: kapy.fptofp
    %2 = arith.bitcast %0 : tensor<64x64xi8, #regis> to tensor<64x64xf8E5M2, #regis>
    %3 = kapy.fptofp %2 : tensor<64x64xf8E5M2, #regis> to tensor<64x64xf16, #regis>
    %4 = kgpu.change %3 : tensor<64x64xf16, #regis> to tensor<64x64xf16, #mmopd>
    %5 = kgpu.change %1 : tensor<64x64xf16, #regis> to tensor<64x64xf16, #mmopd1>
    %6 = kapy.matmul %4, %5, %arg2 : tensor<64x64xf16, #mmopd>, tensor<64x64xf16, #mmopd1> -> tensor<64x64xf32, #nvmma>
    kapy.return %6 : tensor<64x64xf32, #nvmma>
  }
}
