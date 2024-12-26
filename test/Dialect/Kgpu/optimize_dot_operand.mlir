// RUN: kapy-opt %s -kgpu-optimize-dot-operand | FileCheck %s

#gmem = #kapy.gmem<map = (d0, d1)[s0] -> ((d0 + s0 * 64) * 128 + d1)>
#nvmma = #kgpu.nvmma<warp_per_cta = [2, 2]>
#regs = #kgpu.regs<map = (d0, d1, d2) -> (d1 floordiv 32 + d2 floordiv 2 + d0 * 2, d1 mod 32 + (d2 mod 2) * 32)>
#dotld = #kgpu.dotld<parent = #nvmma, operand_index = 0, bit_width = 8>
#dotld1 = #kgpu.dotld<parent = #nvmma, operand_index = 1, bit_width = 8>
module attributes {kgpu.num_warps = 4 : i32, kgpu.nvidia_cc = 80 : i32} {
  // CHECK-LABEL: move_change_ops_next_to_load_ops
  kapy.func @move_change_ops_next_to_load_ops(%arg0: !kapy.memref<64x64xi8, #gmem>, %arg1: !kapy.memref<64x64xf16, #gmem>, %arg2: tensor<64x64xf32, #nvmma>) -> tensor<64x64xf32, #nvmma> {
    %0 = kapy.load %arg0 : !kapy.memref<64x64xi8, #gmem> -> tensor<64x64xi8, #regs>
    %1 = kapy.load %arg1 : !kapy.memref<64x64xf16, #gmem> -> tensor<64x64xf16, #regs>
    // CHECK: kgpu.change
    // CHECK-NEXT: arith.bitcast
    // CHECK-NEXT: kapy.fptofp
    %2 = arith.bitcast %0 : tensor<64x64xi8, #regs> to tensor<64x64xf8E5M2, #regs>
    %3 = kapy.fptofp %2 : tensor<64x64xf8E5M2, #regs> to tensor<64x64xf16, #regs>
    %4 = kgpu.change %3 : tensor<64x64xf16, #regs> to tensor<64x64xf16, #dotld>
    %5 = kgpu.change %1 : tensor<64x64xf16, #regs> to tensor<64x64xf16, #dotld1>
    %6 = kapy.dot %4, %5, %arg2 : tensor<64x64xf16, #dotld>, tensor<64x64xf16, #dotld1> -> tensor<64x64xf32, #nvmma>
    kapy.return %6 : tensor<64x64xf32, #nvmma>
  }
}
