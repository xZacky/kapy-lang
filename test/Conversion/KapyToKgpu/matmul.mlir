module attributes {kapy.num_warps = 4 : i64, kapy.nvidia_cc = 89 : i64} {
  kapy.func @matmul_kernel(%arg0: i64 {kapy.alignment = 128 : i64}, %arg1: i64 {kapy.alignment = 128 : i64}, %arg2: i64 {kapy.alignment = 128 : i64}) {
    %cst = arith.constant dense<0.0> : tensor<64x64xf16>

    %c0_i32 = arith.constant 0 : i32
    %c1_i32 = arith.constant 1 : i32
    %c2_i32 = arith.constant 2 : i32
    %c16_i32 = arith.constant 16 : i32
    %c64_i32 = arith.constant 64 : i32
    %c128_i32 = arith.constant 128 : i32
    %c1024_i32 = arith.constant 1024 : i32

    %g0 = kapy.mk_global %arg0, [%c1024_i32, %c1024_i32], [%c1024_i32, %c1_i32] : !kapy.global<?x?xf16>
    %g1 = kapy.mk_global %arg1, [%c1024_i32, %c1024_i32], [%c1024_i32, %c1_i32] : !kapy.global<?x?xf16>
    %g2 = kapy.mk_global %arg2, [%c1024_i32, %c1024_i32], [%c1024_i32, %c1_i32] : !kapy.global<?x?xf16>

    %s0 = kapy.mk_shared : !kapy.shared<128x64xf16>
    %s1 = kapy.mk_shared : !kapy.shared<64x128xf16>
    %s2 = kapy.mk_shared : !kapy.shared<128x64xf16>
    %s3 = kapy.mk_shared : !kapy.shared<64x128xf16>

    %px = kapy.program_id {axis = 0 : i32} : i32
    %py = kapy.program_id {axis = 1 : i32} : i32

    %w = kapy.warp_id : i32
    %wx = arith.divui %w, %c2_i32 : i32
    %wy = arith.remui %w, %c2_i32 : i32

    %0 = arith.muli %px, %c128_i32 : i32
    %1 = arith.muli %py, %c128_i32 : i32
    %2 = arith.muli %wx, %c64_i32 : i32
    %3 = arith.muli %wy, %c64_i32 : i32
    %4 = arith.addi %0, %2 : i32
    %5 = arith.addi %1, %3 : i32

    kapy.cp_async_global_to_shared %g0[%4, %c0_i32], %s0[%2, %c0_i32], %cst : !kapy.global<?x?xf16>, !kapy.shared<128x64xf16>, tensor<64x64xf16>
    kapy.cp_async_global_to_shared %g1[%c0_i32, %5], %s1[%c0_i32, %3], %cst : !kapy.global<?x?xf16>, !kapy.shared<64x128xf16>, tensor<64x64xf16>
    kapy.cp_async_global_to_shared %g0[%4, %c64_i32], %s2[%2, %c0_i32], %cst : !kapy.global<?x?xf16>, !kapy.shared<128x64xf16>, tensor<64x64xf16>
    kapy.cp_async_global_to_shared %g1[%c64_i32, %5], %s3[%c0_i32, %3], %cst : !kapy.global<?x?xf16>, !kapy.shared<64x128xf16>, tensor<64x64xf16>
    kapy.cp_async_commit_group

    %c1 = scf.for %arg3 = %c2_i32 to %c16_i32 step %c1_i32 iter_args(%arg4 = %cst) -> (tensor<64x64xf16>) : i32 {
      %6 = arith.remui %arg3, %c2_i32 : i32
      %7 = arith.cmpi eq, %6, %c0_i32 : i32

      %s4 = arith.select %7, %s0, %s2 : !kapy.shared<128x64xf16>
      %s5 = arith.select %7, %s1, %s3 : !kapy.shared<64x128xf16>

      kapy.cp_async_wait_group {num_pending = 2 : i32}
      %a0 = kapy.ld_shared %s4[%2, %c0_i32] : !kapy.shared<128x64xf16> -> tensor<64x64xf16>
      %b0 = kapy.ld_shared %s5[%c0_i32, %3] : !kapy.shared<64x128xf16> -> tensor<64x64xf16>
      %d0 = kapy.matmul mma_m16n8k8_f16 %a0, %b0, %arg4 : tensor<64x64xf16>, tensor<64x64xf16> -> tensor<64x64xf16>

      %8 = arith.muli %arg3, %c64_i32 : i32

      kapy.cp_async_global_to_shared %g0[%4, %8], %s4[%2, %c0_i32], %cst : !kapy.global<?x?xf16>, !kapy.shared<128x64xf16>, tensor<64x64xf16>
      kapy.cp_async_global_to_shared %g1[%8, %5], %s5[%c0_i32, %3], %cst : !kapy.global<?x?xf16>, !kapy.shared<64x128xf16>, tensor<64x64xf16>
      kapy.cp_async_commit_group

      scf.yield %d0 : tensor<64x64xf16>
    }

    kapy.cp_async_wait_group {num_pending = 2 : i32}
    %a1 = kapy.ld_shared %s0[%2, %c0_i32] : !kapy.shared<128x64xf16> -> tensor<64x64xf16>
    %b1 = kapy.ld_shared %s1[%c0_i32, %3] : !kapy.shared<64x128xf16> -> tensor<64x64xf16>
    %d1 = kapy.matmul mma_m16n8k8_f16 %a1, %b1, %c1 : tensor<64x64xf16>, tensor<64x64xf16> -> tensor<64x64xf16>

    kapy.cp_async_wait_group {num_pending = 0 : i32}
    %a2 = kapy.ld_shared %s2[%2, %c0_i32] : !kapy.shared<128x64xf16> -> tensor<64x64xf16>
    %b2 = kapy.ld_shared %s3[%c0_i32, %3] : !kapy.shared<64x128xf16> -> tensor<64x64xf16>
    %d2 = kapy.matmul mma_m16n8k16_f16 %a2, %b2, %d1 : tensor<64x64xf16>, tensor<64x64xf16> -> tensor<64x64xf16>

    kapy.st_global %g2[%4, %5], %d2 : !kapy.global<?x?xf16>, tensor<64x64xf16>

    kapy.return
  }
}
