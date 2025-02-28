#global = #kapy.encoding<global_memory>
#shared = #kapy.encoding<shared_memory>
#values = #kapy.encoding<register_file>
module attributes {kapy.num_warps = 4 : i64, kapy.nvidia_cc = 89 : i64} {
  kapy.func @matmul_kernel(%arg0: i64 {kapy.alignment = 128 : i64}, %arg1: i64 {kapy.alignment = 128 : i64}, %arg2: i64 {kapy.alignment = 128 : i64}) {
    %cst = arith.constant dense<0.000000e+00> : tensor<64x64xf16, #values>
    %c0_i32 = arith.constant 0 : i32
    %c1_i32 = arith.constant 1 : i32
    %c2_i32 = arith.constant 2 : i32
    %c16_i32 = arith.constant 16 : i32
    %c64_i32 = arith.constant 64 : i32
    %c128_i32 = arith.constant 128 : i32
    %c1024_i32 = arith.constant 1024 : i32
    %0 = kapy.mk_global %arg0, [%c1024_i32, %c1024_i32], [%c1024_i32, %c1_i32] : tensor<?x?xf16, #global>
    %1 = kapy.mk_global %arg1, [%c1024_i32, %c1024_i32], [%c1024_i32, %c1_i32] : tensor<?x?xf16, #global>
    %2 = kapy.mk_global %arg2, [%c1024_i32, %c1024_i32], [%c1024_i32, %c1_i32] : tensor<?x?xf16, #global>
    %3 = kapy.mk_shared : tensor<128x64xf16, #shared>
    %4 = kapy.mk_shared : tensor<64x128xf16, #shared>
    %5 = kapy.mk_shared : tensor<128x64xf16, #shared>
    %6 = kapy.mk_shared : tensor<64x128xf16, #shared>
    %7 = kapy.program_id {axis = 0 : i32} : i32
    %8 = kapy.program_id {axis = 1 : i32} : i32
    %9 = kapy.warp_id : i32
    %10 = arith.divui %9, %c2_i32 : i32
    %11 = arith.remui %9, %c2_i32 : i32
    %12 = arith.muli %7, %c128_i32 : i32
    %13 = arith.muli %8, %c128_i32 : i32
    %14 = arith.muli %10, %c64_i32 : i32
    %15 = arith.muli %11, %c64_i32 : i32
    %16 = arith.addi %12, %14 : i32
    %17 = arith.addi %13, %15 : i32
    %18 = kapy.sv_global %0[%16, %c0_i32] : tensor<?x?xf16, #global> -> tensor<64x64xf16, #global>
    %19 = kapy.sv_global %1[%c0_i32, %17] : tensor<?x?xf16, #global> -> tensor<64x64xf16, #global>
    %20 = kapy.sv_global %0[%16, %c64_i32] : tensor<?x?xf16, #global> -> tensor<64x64xf16, #global>
    %21 = kapy.sv_global %1[%c64_i32, %17] : tensor<?x?xf16, #global> -> tensor<64x64xf16, #global>
    %22 = kapy.sv_shared %3[%14, %c0_i32] : tensor<128x64xf16, #shared> -> tensor<64x64xf16, #shared>
    %23 = kapy.sv_shared %4[%c0_i32, %15] : tensor<64x128xf16, #shared> -> tensor<64x64xf16, #shared>
    %24 = kapy.sv_shared %5[%14, %c0_i32] : tensor<128x64xf16, #shared> -> tensor<64x64xf16, #shared>
    %25 = kapy.sv_shared %6[%c0_i32, %15] : tensor<64x128xf16, #shared> -> tensor<64x64xf16, #shared>
    kapy.cp_async_global_to_shared %18, %22, %cst : tensor<64x64xf16, #global>, tensor<64x64xf16, #shared>, tensor<64x64xf16, #values>
    kapy.cp_async_global_to_shared %19, %23, %cst : tensor<64x64xf16, #global>, tensor<64x64xf16, #shared>, tensor<64x64xf16, #values>
    kapy.cp_async_global_to_shared %20, %24, %cst : tensor<64x64xf16, #global>, tensor<64x64xf16, #shared>, tensor<64x64xf16, #values>
    kapy.cp_async_global_to_shared %21, %25, %cst : tensor<64x64xf16, #global>, tensor<64x64xf16, #shared>, tensor<64x64xf16, #values>
    kapy.cp_async_commit_group
    %26 = scf.for %arg3 = %c2_i32 to %c16_i32 step %c1_i32 iter_args(%arg4 = %cst) -> (tensor<64x64xf16, #values>)  : i32 {
      %34 = arith.remui %arg3, %c2_i32 : i32
      %35 = arith.cmpi eq, %34, %c0_i32 : i32
      %36 = arith.select %35, %22, %24 : tensor<64x64xf16, #shared>
      %37 = arith.select %35, %23, %25 : tensor<64x64xf16, #shared>
      kapy.cp_async_wait_group {num_pending = 2 : i32}
      %38 = kapy.ld_shared %36 : tensor<64x64xf16, #shared> -> tensor<64x64xf16, #values>
      %39 = kapy.ld_shared %37 : tensor<64x64xf16, #shared> -> tensor<64x64xf16, #values>
      %40 = kapy.matmul mma_m16n8k8_f16 %38, %39, %arg4 : tensor<64x64xf16, #values>, tensor<64x64xf16, #values> -> tensor<64x64xf16, #values>
      %41 = arith.muli %arg3, %c64_i32 : i32
      %42 = kapy.sv_global %0[%16, %41] : tensor<?x?xf16, #global> -> tensor<64x64xf16, #global>
      %43 = kapy.sv_global %1[%41, %17] : tensor<?x?xf16, #global> -> tensor<64x64xf16, #global>
      kapy.cp_async_global_to_shared %42, %36, %cst : tensor<64x64xf16, #global>, tensor<64x64xf16, #shared>, tensor<64x64xf16, #values>
      kapy.cp_async_global_to_shared %43, %37, %cst : tensor<64x64xf16, #global>, tensor<64x64xf16, #shared>, tensor<64x64xf16, #values>
      kapy.cp_async_commit_group
      scf.yield %40 : tensor<64x64xf16, #values>
    }
    kapy.cp_async_wait_group {num_pending = 2 : i32}
    %27 = kapy.ld_shared %22 : tensor<64x64xf16, #shared> -> tensor<64x64xf16, #values>
    %28 = kapy.ld_shared %23 : tensor<64x64xf16, #shared> -> tensor<64x64xf16, #values>
    %29 = kapy.matmul mma_m16n8k8_f16 %27, %28, %26 : tensor<64x64xf16, #values>, tensor<64x64xf16, #values> -> tensor<64x64xf16, #values>
    kapy.cp_async_wait_group {num_pending = 0 : i32}
    %30 = kapy.ld_shared %24 : tensor<64x64xf16, #shared> -> tensor<64x64xf16, #values>
    %31 = kapy.ld_shared %25 : tensor<64x64xf16, #shared> -> tensor<64x64xf16, #values>
    %32 = kapy.matmul mma_m16n8k16_f16 %30, %31, %29 : tensor<64x64xf16, #values>, tensor<64x64xf16, #values> -> tensor<64x64xf16, #values>
    %33 = kapy.sv_global %2[%16, %17] : tensor<?x?xf16, #global> -> tensor<64x64xf16, #global>
    kapy.st_global %32, %33 : tensor<64x64xf16, #values>, tensor<64x64xf16, #global>
    kapy.return
  }
}
