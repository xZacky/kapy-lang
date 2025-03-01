#global = #kapy.encoding<global_memory>
#shared = #kapy.encoding<shared_memory>
#values = #kapy.encoding<register_file>
module attributes {kapy.num_warps = 4 : i64, kapy.nvidia_cc = 89 : i64} {
  kapy.func @matmul_kernel(%arg0: i64 {kapy.alignment = 128 : i64}, %arg1: i64 {kapy.alignment = 128 : i64}, %arg2: i64 {kapy.alignment = 128 : i64}, %arg3: i32 {kapy.alignment = 1 : i64}, %arg4: i32 {kapy.alignment = 1 : i64}, %arg5: i32 {kapy.alignment = 1 : i64}, %arg6: i32 {kapy.alignment = 1 : i64}, %arg7: i32 {kapy.alignment = 1 : i64}, %arg8: i32 {kapy.alignment = 1 : i64}, %arg9: i32 {kapy.alignment = 1 : i64}, %arg10: i32 {kapy.alignment = 1 : i64}, %arg11: i32 {kapy.alignment = 1 : i64}) {
    %cst = arith.constant dense<0.000000e+00> : tensor<64x64xf16, #values>
    %c0_i32 = arith.constant 0 : i32
    %c1_i32 = arith.constant 1 : i32
    %c2_i32 = arith.constant 2 : i32
    %c16_i32 = arith.constant 16 : i32
    %c64_i32 = arith.constant 64 : i32
    %c128_i32 = arith.constant 128 : i32
    %0 = kapy.mk_global %arg0, [%arg3, %arg4], [%arg9, %c1_i32] : tensor<?x?xf16, #global>
    %1 = kapy.mk_global %arg1, [%arg5, %arg6], [%arg10, %c1_i32] : tensor<?x?xf16, #global>
    %2 = kapy.mk_global %arg2, [%arg7, %arg8], [%arg11, %c1_i32] : tensor<?x?xf16, #global>
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
    %18 = arith.addi %16, %c64_i32 : i32
    %19 = arith.addi %17, %c64_i32 : i32
    %20 = arith.addi %14, %c64_i32 : i32
    %21 = arith.addi %15, %c64_i32 : i32
    %22 = kapy.sv_global %0[%16 : %18, %c0_i32 : %c64_i32] : tensor<?x?xf16, #global> -> tensor<64x64xf16, #global>
    %23 = kapy.sv_global %1[%c0_i32 : %c64_i32, %17 : %19] : tensor<?x?xf16, #global> -> tensor<64x64xf16, #global>
    %24 = kapy.sv_global %0[%16 : %18, %c64_i32 : %c128_i32] : tensor<?x?xf16, #global> -> tensor<64x64xf16, #global>
    %25 = kapy.sv_global %1[%c64_i32 : %c128_i32, %17 : %19] : tensor<?x?xf16, #global> -> tensor<64x64xf16, #global>
    %26 = kapy.sv_shared %3[%14 : %20, %c0_i32 : %c64_i32] : tensor<128x64xf16, #shared> -> tensor<64x64xf16, #shared>
    %27 = kapy.sv_shared %4[%c0_i32 : %c64_i32, %15 : %21] : tensor<64x128xf16, #shared> -> tensor<64x64xf16, #shared>
    %28 = kapy.sv_shared %5[%14 : %20, %c0_i32 : %c64_i32] : tensor<128x64xf16, #shared> -> tensor<64x64xf16, #shared>
    %29 = kapy.sv_shared %6[%c0_i32 : %c64_i32, %15 : %21] : tensor<64x128xf16, #shared> -> tensor<64x64xf16, #shared>
    kapy.cp_async_global_to_shared %22, %26, %cst : tensor<64x64xf16, #global>, tensor<64x64xf16, #shared>, tensor<64x64xf16, #values>
    kapy.cp_async_global_to_shared %23, %27, %cst : tensor<64x64xf16, #global>, tensor<64x64xf16, #shared>, tensor<64x64xf16, #values>
    kapy.cp_async_global_to_shared %24, %28, %cst : tensor<64x64xf16, #global>, tensor<64x64xf16, #shared>, tensor<64x64xf16, #values>
    kapy.cp_async_global_to_shared %25, %29, %cst : tensor<64x64xf16, #global>, tensor<64x64xf16, #shared>, tensor<64x64xf16, #values>
    kapy.cp_async_commit_group
    %30 = scf.for %arg12 = %c2_i32 to %c16_i32 step %c1_i32 iter_args(%arg13 = %cst) -> (tensor<64x64xf16, #values>)  : i32 {
      %38 = arith.remui %arg12, %c2_i32 : i32
      %39 = arith.cmpi eq, %38, %c0_i32 : i32
      %40 = arith.select %39, %26, %28 : tensor<64x64xf16, #shared>
      %41 = arith.select %39, %27, %29 : tensor<64x64xf16, #shared>
      kapy.cp_async_wait_group {num_pending = 2 : i32}
      // %42 = kapy.ld_shared %40, %cst : tensor<64x64xf16, #shared> -> tensor<64x64xf16, #values>
      // %43 = kapy.ld_shared %41, %cst : tensor<64x64xf16, #shared> -> tensor<64x64xf16, #values>
      %42 = kapy.ld_matrix %40, %cst : tensor<64x64xf16, #shared>, tensor<64x64xf16, #values> -> tensor<64x64xf16, #values>
      %43 = kapy.ld_matrix %41, %cst : tensor<64x64xf16, #shared>, tensor<64x64xf16, #values> -> tensor<64x64xf16, #values>
      %44 = kapy.matmul mma_m16n8k8_f16 %42, %43, %arg13 : tensor<64x64xf16, #values>, tensor<64x64xf16, #values> -> tensor<64x64xf16, #values>
      %45 = arith.muli %arg12, %c64_i32 : i32
      %46 = arith.addi %45, %c64_i32 : i32
      %47 = kapy.sv_global %0[%16 : %18, %45 : %46] : tensor<?x?xf16, #global> -> tensor<64x64xf16, #global>
      %48 = kapy.sv_global %1[%45 : %46, %17 : %19] : tensor<?x?xf16, #global> -> tensor<64x64xf16, #global>
      kapy.cp_async_global_to_shared %47, %40, %cst : tensor<64x64xf16, #global>, tensor<64x64xf16, #shared>, tensor<64x64xf16, #values>
      kapy.cp_async_global_to_shared %48, %41, %cst : tensor<64x64xf16, #global>, tensor<64x64xf16, #shared>, tensor<64x64xf16, #values>
      kapy.cp_async_commit_group
      scf.yield %44 : tensor<64x64xf16, #values>
    }
    kapy.cp_async_wait_group {num_pending = 2 : i32}
    // %31 = kapy.ld_shared %26, %cst : tensor<64x64xf16, #shared> -> tensor<64x64xf16, #values>
    // %32 = kapy.ld_shared %27, %cst : tensor<64x64xf16, #shared> -> tensor<64x64xf16, #values>
    %31 = kapy.ld_matrix %26, %cst : tensor<64x64xf16, #shared>, tensor<64x64xf16, #values> -> tensor<64x64xf16, #values>
    %32 = kapy.ld_matrix %27, %cst : tensor<64x64xf16, #shared>, tensor<64x64xf16, #values> -> tensor<64x64xf16, #values>
    %33 = kapy.matmul mma_m16n8k8_f16 %31, %32, %30 : tensor<64x64xf16, #values>, tensor<64x64xf16, #values> -> tensor<64x64xf16, #values>
    kapy.cp_async_wait_group {num_pending = 0 : i32}
    // %34 = kapy.ld_shared %28, %cst : tensor<64x64xf16, #shared> -> tensor<64x64xf16, #values>
    // %35 = kapy.ld_shared %29, %cst : tensor<64x64xf16, #shared> -> tensor<64x64xf16, #values>
    %34 = kapy.ld_matrix %28, %cst : tensor<64x64xf16, #shared>, tensor<64x64xf16, #values> -> tensor<64x64xf16, #values>
    %35 = kapy.ld_matrix %29, %cst : tensor<64x64xf16, #shared>, tensor<64x64xf16, #values> -> tensor<64x64xf16, #values>
    %36 = kapy.matmul mma_m16n8k16_f16 %34, %35, %33 : tensor<64x64xf16, #values>, tensor<64x64xf16, #values> -> tensor<64x64xf16, #values>
    %37 = kapy.sv_global %2[%16 : %18, %17 : %19] : tensor<?x?xf16, #global> -> tensor<64x64xf16, #global>
    kapy.st_global %36, %37 : tensor<64x64xf16, #values>, tensor<64x64xf16, #global>
    kapy.return
  }
}
