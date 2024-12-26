// RUN: kapy-opt %s -kgpu-accelerate-matmul | FileCheck %s

// CHECK: #[[NVMMA:.*]] = #kgpu.nvmma<warp_per_cta = [1, 4]>
// CHECK: #[[DOTLD0:.*]] = #kgpu.dotld<parent = #nvmma, operand_index = 0, bit_width = 16>
// CHECK: #[[DOTLD1:.*]] = #kgpu.dotld<parent = #nvmma, operand_index = 1, bit_width = 16>
#regs = #kgpu.regs<map = (d0, d1, d2) -> ((d0 floordiv 4) mod 4 + (d1 floordiv 16) * 4 + d2 * 8 + ((d0 floordiv 4) floordiv 4) * 32, d0 mod 4 + (d1 mod 16) * 4 + ((d0 mod 4) floordiv 4) * 64)>
#regs1 = #kgpu.regs<map = (d0, d1, d2) -> ((d0 floordiv 4) mod 4 + (d1 floordiv 32) * 4 + d2 * 4 + ((d0 floordiv 4) floordiv 4) * 16, d0 mod 4 + (d1 mod 32) * 4 + ((d0 mod 4) floordiv 4) * 128)>
#dotld = #kgpu.dotld<parent = #regs, operand_index = 0, bit_width = 16>
#dotld1 = #kgpu.dotld<parent = #regs, operand_index = 1, bit_width = 16>
#dotld2 = #kgpu.dotld<parent = #regs1, operand_index = 1, bit_width = 16>
#dotld3 = #kgpu.dotld<parent = #regs1, operand_index = 0, bit_width = 16>
module attributes {kgpu.num_warps = 4 : i32, kgpu.nvidia_cc = 80 : i32} {
  kapy.func @accelerate_chained_matmul_ops(%arg0: tensor<64x128xf16, #dotld>, %arg1: tensor<128x64xf16, #dotld1>, %arg2: tensor<64x128xf16, #dotld2>) -> tensor<64x128xf32, #regs1> {
    %cst = arith.constant dense<0.000000e+00> : tensor<64x64xf32, #regs>
    %cst_0 = arith.constant dense<0.000000e+00> : tensor<64x128xf32, #regs1>
    %0 = kapy.dot %arg0, %arg1, %cst : tensor<64x128xf16, #dotld>, tensor<128x64xf16, #dotld1> -> tensor<64x64xf32, #regs>
    %1 = arith.truncf %0 : tensor<64x64xf32, #regs> to tensor<64x64xf16, #regs>
    %2 = kgpu.change %1 : tensor<64x64xf16, #regs> to tensor<64x64xf16, #dotld3>
    // CHECK: kapy.dot
    // CHECK-SAME: #[[DOTLD0]]
    // CHECK-SAME: #[[DOTLD1]]
    // CHECK-SAME: #[[NVMMA]]
    %3 = kapy.dot %2, %arg2, %cst_0 : tensor<64x64xf16, #dotld3>, tensor<64x128xf16, #dotld2> -> tensor<64x128xf32, #regs1>
    kapy.return %3 : tensor<64x128xf32, #regs1>
  }
}
