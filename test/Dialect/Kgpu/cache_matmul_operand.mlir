// RUN: kapy-opt %s  -kgpu-cache-matmul-operand | FileCheck %s

// CHECK: #[[SHMEM:.*]] = #kgpu.shmem<[256, 1], 16, 8>
#nvmma = #kgpu.nvmma<[1, 4], [2, 1]>
#regis = #kgpu.regis<[1, 4], [1, 1], [16, 2], [1, 1], (0, 1)>
#mmopd = #kgpu.mmopd<#nvmma, 0, 16>
module attributes {kgpu.num_warps = 4 : i64, kgpu.nvidia_cc = 80 : i64} {
  kapy.func @cache_matmul_operand(%arg0: tensor<16x256xf16, #regis>) {
    // CHECK: kgpu.local_alloc
    // CHECK-SAME: #[[SHMEM]]
    // CHECK: kgpu.local_load
    // CHECK-SAME: #[[SHMEM]]
    %0 = kgpu.change %arg0 : tensor<16x256xf16, #regis> to tensor<16x256xf16, #mmopd>
    kapy.return
  }
}
