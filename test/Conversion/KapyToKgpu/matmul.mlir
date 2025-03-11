#global = #kapy.encoding<global_memory, #kapy.strided2d<[?, ?]>>
#shared = #kapy.encoding<shared_memory, #kapy.swizzling<[64, 1], (?, ?)>>
#shared1 = #kapy.encoding<shared_memory, #kapy.swizzling<[128, 1], (?, ?)>>
#values = #kapy.encoding<register_file>
module attributes {kapy.num_warps = 4 : i64, kapy.nvidia_cc = 89 : i64} {
  kapy.func @matmul_kernel(%arg0: i64 {kapy.alignment = 128 : i64}, %arg1: i64 {kapy.alignment = 128 : i64}, %arg2: i64 {kapy.alignment = 128 : i64}, %arg3: i32 {kapy.alignment = 1 : i64}, %arg4: i32 {kapy.alignment = 1 : i64}, %arg5: i32 {kapy.alignment = 1 : i64}, %arg6: i32 {kapy.alignment = 1 : i64}, %arg7: i32 {kapy.alignment = 1 : i64}, %arg8: i32 {kapy.alignment = 1 : i64}, %arg9: i32 {kapy.alignment = 1 : i64}, %arg10: i32 {kapy.alignment = 1 : i64}, %arg11: i32 {kapy.alignment = 1 : i64}) {
    %cst = arith.constant dense<0.000000e+00> : tensor<64x64xf16, #values>
    %cst_0 = arith.constant dense<0.000000e+00> : tensor<64x16xf16, #values>
    %cst_1 = arith.constant dense<0.000000e+00> : tensor<16x64xf16, #values>
    %c0_i32 = arith.constant 0 : i32
    %c1_i32 = arith.constant 1 : i32
    %c2_i32 = arith.constant 2 : i32
    %c16_i32 = arith.constant 16 : i32
    %c32_i32 = arith.constant 32 : i32
    %c48_i32 = arith.constant 48 : i32
    %c64_i32 = arith.constant 64 : i32
    %c128_i32 = arith.constant 128 : i32
    %0 = kapy.mk_global %arg0, [%arg3, %arg4], [%arg9, %c1_i32] : tensor<?x?xf16, #global>
    %1 = kapy.mk_global %arg1, [%arg5, %arg6], [%arg10, %c1_i32] : tensor<?x?xf16, #global>
    %2 = kapy.mk_global %arg2, [%arg7, %arg8], [%arg11, %c1_i32] : tensor<?x?xf16, #global>
    %3 = kapy.mk_shared : tensor<128x64xf16, #shared>
    %4 = kapy.mk_shared : tensor<64x128xf16, #shared1>
    %5 = kapy.mk_shared : tensor<128x64xf16, #shared>
    %6 = kapy.mk_shared : tensor<64x128xf16, #shared1>
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
    %27 = kapy.sv_shared %4[%c0_i32 : %c64_i32, %15 : %21] : tensor<64x128xf16, #shared1> -> tensor<64x64xf16, #shared1>
    %28 = kapy.sv_shared %5[%14 : %20, %c0_i32 : %c64_i32] : tensor<128x64xf16, #shared> -> tensor<64x64xf16, #shared>
    %29 = kapy.sv_shared %6[%c0_i32 : %c64_i32, %15 : %21] : tensor<64x128xf16, #shared1> -> tensor<64x64xf16, #shared1>
    kapy.cp_async_global_to_shared %22, %26, %cst : tensor<64x64xf16, #global>, tensor<64x64xf16, #shared>, tensor<64x64xf16, #values>
    kapy.cp_async_global_to_shared %23, %27, %cst : tensor<64x64xf16, #global>, tensor<64x64xf16, #shared1>, tensor<64x64xf16, #values>
    kapy.cp_async_global_to_shared %24, %28, %cst : tensor<64x64xf16, #global>, tensor<64x64xf16, #shared>, tensor<64x64xf16, #values>
    kapy.cp_async_global_to_shared %25, %29, %cst : tensor<64x64xf16, #global>, tensor<64x64xf16, #shared1>, tensor<64x64xf16, #values>
    kapy.cp_async_wait_group {num_pending = 2 : i32}
    %30 = kapy.sv_shared %26[%c0_i32 : %c64_i32, %c0_i32 : %c16_i32] : tensor<64x64xf16, #shared> -> tensor<64x16xf16, #shared>
    %31 = kapy.sv_shared %27[%c0_i32 : %c16_i32, %c0_i32 : %c64_i32] : tensor<64x64xf16, #shared1> -> tensor<16x64xf16, #shared1>
    %32 = kapy.ld_matrix %30, %cst_0 : tensor<64x16xf16, #shared>, tensor<64x16xf16, #values> -> tensor<64x16xf16, #values>
    %33 = kapy.ld_matrix %31, %cst_1 : tensor<16x64xf16, #shared1>, tensor<16x64xf16, #values> -> tensor<16x64xf16, #values>
    %34:3 = scf.for %arg12 = %c2_i32 to %c16_i32 step %c1_i32 iter_args(%arg13 = %32, %arg14 = %33, %arg15 = %cst) -> (tensor<64x16xf16, #values>, tensor<16x64xf16, #values>, tensor<64x64xf16, #values>)  : i32 {
      %72 = arith.remui %arg12, %c2_i32 : i32
      %73 = arith.cmpi eq, %72, %c0_i32 : i32
      %74 = arith.select %73, %26, %28 : tensor<64x64xf16, #shared>
      %75 = arith.select %73, %27, %29 : tensor<64x64xf16, #shared1>
      %76 = kapy.sv_shared %74[%c0_i32 : %c64_i32, %c16_i32 : %c32_i32] : tensor<64x64xf16, #shared> -> tensor<64x16xf16, #shared>
      %77 = kapy.sv_shared %75[%c16_i32 : %c32_i32, %c0_i32 : %c64_i32] : tensor<64x64xf16, #shared1> -> tensor<16x64xf16, #shared1>
      %78 = kapy.ld_matrix %76, %cst_0 : tensor<64x16xf16, #shared>, tensor<64x16xf16, #values> -> tensor<64x16xf16, #values>
      %79 = kapy.ld_matrix %77, %cst_1 : tensor<16x64xf16, #shared1>, tensor<16x64xf16, #values> -> tensor<16x64xf16, #values>
      %80 = kapy.matmul mma_m16n8k8_f16 %arg13, %arg14, %arg15 : tensor<64x16xf16, #values>, tensor<16x64xf16, #values> -> tensor<64x64xf16, #values>
      %81 = kapy.sv_shared %74[%c0_i32 : %c64_i32, %c32_i32 : %c48_i32] : tensor<64x64xf16, #shared> -> tensor<64x16xf16, #shared>
      %82 = kapy.sv_shared %75[%c32_i32 : %c48_i32, %c0_i32 : %c64_i32] : tensor<64x64xf16, #shared1> -> tensor<16x64xf16, #shared1>
      %83 = kapy.ld_matrix %81, %cst_0 : tensor<64x16xf16, #shared>, tensor<64x16xf16, #values> -> tensor<64x16xf16, #values>
      %84 = kapy.ld_matrix %82, %cst_1 : tensor<16x64xf16, #shared1>, tensor<16x64xf16, #values> -> tensor<16x64xf16, #values>
      %85 = kapy.matmul mma_m16n8k8_f16 %78, %79, %80 : tensor<64x16xf16, #values>, tensor<16x64xf16, #values> -> tensor<64x64xf16, #values>
      %86 = kapy.sv_shared %74[%c0_i32 : %c64_i32, %c48_i32 : %c64_i32] : tensor<64x64xf16, #shared> -> tensor<64x16xf16, #shared>
      %87 = kapy.sv_shared %75[%c48_i32 : %c64_i32, %c0_i32 : %c64_i32] : tensor<64x64xf16, #shared1> -> tensor<16x64xf16, #shared1>
      %88 = kapy.ld_matrix %86, %cst_0 : tensor<64x16xf16, #shared>, tensor<64x16xf16, #values> -> tensor<64x16xf16, #values>
      %89 = kapy.ld_matrix %87, %cst_1 : tensor<16x64xf16, #shared1>, tensor<16x64xf16, #values> -> tensor<16x64xf16, #values>
      %90 = kapy.matmul mma_m16n8k8_f16 %83, %84, %85 : tensor<64x16xf16, #values>, tensor<16x64xf16, #values> -> tensor<64x64xf16, #values>
      %91 = arith.muli %arg12, %c64_i32 : i32
      %92 = arith.addi %91, %c64_i32 : i32
      %93 = kapy.sv_global %0[%16 : %18, %91 : %92] : tensor<?x?xf16, #global> -> tensor<64x64xf16, #global>
      %94 = kapy.sv_global %1[%91 : %92, %17 : %19] : tensor<?x?xf16, #global> -> tensor<64x64xf16, #global>
      kapy.cp_async_global_to_shared %93, %74, %cst : tensor<64x64xf16, #global>, tensor<64x64xf16, #shared>, tensor<64x64xf16, #values>
      kapy.cp_async_global_to_shared %94, %75, %cst : tensor<64x64xf16, #global>, tensor<64x64xf16, #shared1>, tensor<64x64xf16, #values>
      kapy.cp_async_wait_group {num_pending = 2 : i32}
      %95 = arith.select %73, %28, %26 : tensor<64x64xf16, #shared>
      %96 = arith.select %73, %29, %27 : tensor<64x64xf16, #shared1>
      %97 = kapy.sv_shared %95[%c0_i32 : %c64_i32, %c0_i32 : %c16_i32] : tensor<64x64xf16, #shared> -> tensor<64x16xf16, #shared>
      %98 = kapy.sv_shared %96[%c0_i32 : %c16_i32, %c0_i32 : %c64_i32] : tensor<64x64xf16, #shared1> -> tensor<16x64xf16, #shared1>
      %99 = kapy.ld_matrix %97, %cst_0 : tensor<64x16xf16, #shared>, tensor<64x16xf16, #values> -> tensor<64x16xf16, #values>
      %100 = kapy.ld_matrix %98, %cst_1 : tensor<16x64xf16, #shared1>, tensor<16x64xf16, #values> -> tensor<16x64xf16, #values>
      %101 = kapy.matmul mma_m16n8k8_f16 %88, %89, %90 : tensor<64x16xf16, #values>, tensor<16x64xf16, #values> -> tensor<64x64xf16, #values>
      scf.yield %99, %100, %101 : tensor<64x16xf16, #values>, tensor<16x64xf16, #values>, tensor<64x64xf16, #values>
    }
    %35 = kapy.sv_shared %26[%c0_i32 : %c64_i32, %c16_i32 : %c32_i32] : tensor<64x64xf16, #shared> -> tensor<64x16xf16, #shared>
    %36 = kapy.sv_shared %27[%c16_i32 : %c32_i32, %c0_i32 : %c64_i32] : tensor<64x64xf16, #shared1> -> tensor<16x64xf16, #shared1>
    %37 = kapy.ld_matrix %35, %cst_0 : tensor<64x16xf16, #shared>, tensor<64x16xf16, #values> -> tensor<64x16xf16, #values>
    %38 = kapy.ld_matrix %36, %cst_1 : tensor<16x64xf16, #shared1>, tensor<16x64xf16, #values> -> tensor<16x64xf16, #values>
    %39 = kapy.matmul mma_m16n8k8_f16 %34#0, %34#1, %34#2 : tensor<64x16xf16, #values>, tensor<16x64xf16, #values> -> tensor<64x64xf16, #values>
    %40 = kapy.sv_shared %26[%c0_i32 : %c64_i32, %c32_i32 : %c48_i32] : tensor<64x64xf16, #shared> -> tensor<64x16xf16, #shared>
    %41 = kapy.sv_shared %27[%c32_i32 : %c48_i32, %c0_i32 : %c64_i32] : tensor<64x64xf16, #shared1> -> tensor<16x64xf16, #shared1>
    %42 = kapy.ld_matrix %40, %cst_0 : tensor<64x16xf16, #shared>, tensor<64x16xf16, #values> -> tensor<64x16xf16, #values>
    %43 = kapy.ld_matrix %41, %cst_1 : tensor<16x64xf16, #shared1>, tensor<16x64xf16, #values> -> tensor<16x64xf16, #values>
    %44 = kapy.matmul mma_m16n8k8_f16 %37, %38, %39 : tensor<64x16xf16, #values>, tensor<16x64xf16, #values> -> tensor<64x64xf16, #values>
    %45 = kapy.sv_shared %26[%c0_i32 : %c64_i32, %c48_i32 : %c64_i32] : tensor<64x64xf16, #shared> -> tensor<64x16xf16, #shared>
    %46 = kapy.sv_shared %27[%c48_i32 : %c64_i32, %c0_i32 : %c64_i32] : tensor<64x64xf16, #shared1> -> tensor<16x64xf16, #shared1>
    %47 = kapy.ld_matrix %45, %cst_0 : tensor<64x16xf16, #shared>, tensor<64x16xf16, #values> -> tensor<64x16xf16, #values>
    %48 = kapy.ld_matrix %46, %cst_1 : tensor<16x64xf16, #shared1>, tensor<16x64xf16, #values> -> tensor<16x64xf16, #values>
    %49 = kapy.matmul mma_m16n8k8_f16 %42, %43, %44 : tensor<64x16xf16, #values>, tensor<16x64xf16, #values> -> tensor<64x64xf16, #values>
    kapy.cp_async_wait_group {num_pending = 0 : i32}
    %50 = kapy.sv_shared %28[%c0_i32 : %c64_i32, %c0_i32 : %c16_i32] : tensor<64x64xf16, #shared> -> tensor<64x16xf16, #shared>
    %51 = kapy.sv_shared %29[%c0_i32 : %c16_i32, %c0_i32 : %c64_i32] : tensor<64x64xf16, #shared1> -> tensor<16x64xf16, #shared1>
    %52 = kapy.ld_matrix %50, %cst_0 : tensor<64x16xf16, #shared>, tensor<64x16xf16, #values> -> tensor<64x16xf16, #values>
    %53 = kapy.ld_matrix %51, %cst_1 : tensor<16x64xf16, #shared1>, tensor<16x64xf16, #values> -> tensor<16x64xf16, #values>
    %54 = kapy.matmul mma_m16n8k8_f16 %47, %48, %49 : tensor<64x16xf16, #values>, tensor<16x64xf16, #values> -> tensor<64x64xf16, #values>
    %55 = kapy.sv_shared %28[%c0_i32 : %c64_i32, %c16_i32 : %c32_i32] : tensor<64x64xf16, #shared> -> tensor<64x16xf16, #shared>
    %56 = kapy.sv_shared %29[%c16_i32 : %c32_i32, %c0_i32 : %c64_i32] : tensor<64x64xf16, #shared1> -> tensor<16x64xf16, #shared1>
    %57 = kapy.ld_matrix %55, %cst_0 : tensor<64x16xf16, #shared>, tensor<64x16xf16, #values> -> tensor<64x16xf16, #values>
    %58 = kapy.ld_matrix %56, %cst_1 : tensor<16x64xf16, #shared1>, tensor<16x64xf16, #values> -> tensor<16x64xf16, #values>
    %59 = kapy.matmul mma_m16n8k8_f16 %52, %53, %54 : tensor<64x16xf16, #values>, tensor<16x64xf16, #values> -> tensor<64x64xf16, #values>
    %60 = kapy.sv_shared %28[%c0_i32 : %c64_i32, %c32_i32 : %c48_i32] : tensor<64x64xf16, #shared> -> tensor<64x16xf16, #shared>
    %61 = kapy.sv_shared %29[%c32_i32 : %c48_i32, %c0_i32 : %c64_i32] : tensor<64x64xf16, #shared1> -> tensor<16x64xf16, #shared1>
    %62 = kapy.ld_matrix %60, %cst_0 : tensor<64x16xf16, #shared>, tensor<64x16xf16, #values> -> tensor<64x16xf16, #values>
    %63 = kapy.ld_matrix %61, %cst_1 : tensor<16x64xf16, #shared1>, tensor<16x64xf16, #values> -> tensor<16x64xf16, #values>
    %64 = kapy.matmul mma_m16n8k8_f16 %57, %58, %59 : tensor<64x16xf16, #values>, tensor<16x64xf16, #values> -> tensor<64x64xf16, #values>
    %65 = kapy.sv_shared %28[%c0_i32 : %c64_i32, %c48_i32 : %c64_i32] : tensor<64x64xf16, #shared> -> tensor<64x16xf16, #shared>
    %66 = kapy.sv_shared %29[%c48_i32 : %c64_i32, %c0_i32 : %c64_i32] : tensor<64x64xf16, #shared1> -> tensor<16x64xf16, #shared1>
    %67 = kapy.ld_matrix %65, %cst_0 : tensor<64x16xf16, #shared>, tensor<64x16xf16, #values> -> tensor<64x16xf16, #values>
    %68 = kapy.ld_matrix %66, %cst_1 : tensor<16x64xf16, #shared1>, tensor<16x64xf16, #values> -> tensor<16x64xf16, #values>
    %69 = kapy.matmul mma_m16n8k8_f16 %62, %63, %64 : tensor<64x16xf16, #values>, tensor<16x64xf16, #values> -> tensor<64x64xf16, #values>
    %70 = kapy.matmul mma_m16n8k8_f16 %67, %68, %69 : tensor<64x16xf16, #values>, tensor<16x64xf16, #values> -> tensor<64x64xf16, #values>
    %71 = kapy.sv_global %2[%16 : %18, %17 : %19] : tensor<?x?xf16, #global> -> tensor<64x64xf16, #global>
    kapy.st_global %70, %71 : tensor<64x64xf16, #values>, tensor<64x64xf16, #global>
    kapy.return
  }
}
