#global = #kapy.encoding<global_memory>
#shared = #kapy.encoding<shared_memory>
#values = #kapy.encoding<register_file>
module attributes {kapy.num_warps = 4 : i64, kapy.nvidia_cc = 89 : i64} {
  kapy.func public @matmul_kernel(%arg0: !kapy.ptr<1> {kapy.alignment = 128 : i64}, %arg1: !kapy.ptr<1> {kapy.alignment = 128 : i64}, %arg2: !kapy.ptr<1> {kapy.alignment = 128 : i64}, %arg3: i32 {kapy.alignment = 128 : i64}, %arg4: i32 {kapy.alignment = 64 : i64}, %arg5: i32 {kapy.alignment = 128 : i64}) {
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
    %0 = kapy.mk_global %arg0[%c0_i32] [%arg3, %arg4] [%arg4, %c1_i32] : !kapy.ptr<1> -> tensor<?x?xf16, #global>
    %1 = kapy.mk_global %arg1[%c0_i32] [%arg4, %arg5] [%arg5, %c1_i32] : !kapy.ptr<1> -> tensor<?x?xf16, #global>
    %2 = kapy.mk_shared {kapy.row_major = true} : tensor<128x64xf16, #shared>
    %3 = kapy.mk_shared {kapy.row_major = true} : tensor<64x128xf16, #shared>
    %4 = kapy.mk_shared {kapy.row_major = true} : tensor<128x64xf16, #shared>
    %5 = kapy.mk_shared {kapy.row_major = true} : tensor<64x128xf16, #shared>
    %6 = kapy.program_id {axis = 0 : i32} : i32
    %7 = kapy.program_id {axis = 1 : i32} : i32
    %8 = kapy.warp_id : i32
    %9 = arith.divui %8, %c2_i32 : i32
    %10 = arith.remui %8, %c2_i32 : i32
    %11 = arith.muli %6, %c128_i32 : i32
    %12 = arith.muli %7, %c128_i32 : i32
    %13 = arith.muli %9, %c64_i32 : i32
    %14 = arith.muli %10, %c64_i32 : i32
    %15 = arith.addi %11, %13 : i32
    %16 = arith.addi %12, %14 : i32
    %17 = arith.addi %15, %c64_i32 : i32
    %18 = arith.addi %16, %c64_i32 : i32
    %19 = arith.addi %13, %c64_i32 : i32
    %20 = arith.addi %14, %c64_i32 : i32
    %21 = kapy.sv_global %0[%15 : %17, %c0_i32 : %c64_i32] : tensor<?x?xf16, #global> -> tensor<64x64xf16, #global>
    %22 = kapy.sv_global %1[%c0_i32 : %c64_i32, %16 : %18] : tensor<?x?xf16, #global> -> tensor<64x64xf16, #global>
    %23 = kapy.sv_global %0[%15 : %17, %c64_i32 : %c128_i32] : tensor<?x?xf16, #global> -> tensor<64x64xf16, #global>
    %24 = kapy.sv_global %1[%c64_i32 : %c128_i32, %16 : %18] : tensor<?x?xf16, #global> -> tensor<64x64xf16, #global>
    %25 = kapy.sv_shared %2[%13 : %19, %c0_i32 : %c64_i32] : tensor<128x64xf16, #shared> -> tensor<64x64xf16, #shared>
    %26 = kapy.sv_shared %3[%c0_i32 : %c64_i32, %14 : %20] : tensor<64x128xf16, #shared> -> tensor<64x64xf16, #shared>
    %27 = kapy.sv_shared %4[%13 : %19, %c0_i32 : %c64_i32] : tensor<128x64xf16, #shared> -> tensor<64x64xf16, #shared>
    %28 = kapy.sv_shared %5[%c0_i32 : %c64_i32, %14 : %20] : tensor<64x128xf16, #shared> -> tensor<64x64xf16, #shared>
    kapy.cp_async_global_to_shared %21, %25, %cst : tensor<64x64xf16, #global>, tensor<64x64xf16, #shared>, tensor<64x64xf16, #values>
    kapy.cp_async_global_to_shared %22, %26, %cst : tensor<64x64xf16, #global>, tensor<64x64xf16, #shared>, tensor<64x64xf16, #values>
    kapy.cp_async_global_to_shared %23, %27, %cst : tensor<64x64xf16, #global>, tensor<64x64xf16, #shared>, tensor<64x64xf16, #values>
    kapy.cp_async_global_to_shared %24, %28, %cst : tensor<64x64xf16, #global>, tensor<64x64xf16, #shared>, tensor<64x64xf16, #values>
    kapy.cp_async_wait_group {num_pending = 2 : i32}
    %29 = kapy.sv_shared %25[%c0_i32 : %c64_i32, %c0_i32 : %c16_i32] : tensor<64x64xf16, #shared> -> tensor<64x16xf16, #shared>
    %30 = kapy.sv_shared %26[%c0_i32 : %c16_i32, %c0_i32 : %c64_i32] : tensor<64x64xf16, #shared> -> tensor<16x64xf16, #shared>
    %31 = kapy.ld_matrix %29, %cst_0 : tensor<64x16xf16, #shared>, tensor<64x16xf16, #values> -> tensor<64x16xf16, #values>
    %32 = kapy.ld_matrix %30, %cst_1 : tensor<16x64xf16, #shared>, tensor<16x64xf16, #values> -> tensor<16x64xf16, #values>
    %33 = arith.divui %arg4, %c64_i32 : i32
    %34:3 = scf.for %arg6 = %c2_i32 to %33 step %c1_i32 iter_args(%arg7 = %31, %arg8 = %32, %arg9 = %cst) -> (tensor<64x16xf16, #values>, tensor<16x64xf16, #values>, tensor<64x64xf16, #values>)  : i32 {
      %73 = arith.remui %arg6, %c2_i32 : i32
      %74 = arith.cmpi eq, %73, %c0_i32 : i32
      %75 = arith.select %74, %25, %27 : tensor<64x64xf16, #shared>
      %76 = arith.select %74, %26, %28 : tensor<64x64xf16, #shared>
      %77 = kapy.sv_shared %75[%c0_i32 : %c64_i32, %c16_i32 : %c32_i32] : tensor<64x64xf16, #shared> -> tensor<64x16xf16, #shared>
      %78 = kapy.sv_shared %76[%c16_i32 : %c32_i32, %c0_i32 : %c64_i32] : tensor<64x64xf16, #shared> -> tensor<16x64xf16, #shared>
      %79 = kapy.ld_matrix %77, %cst_0 : tensor<64x16xf16, #shared>, tensor<64x16xf16, #values> -> tensor<64x16xf16, #values>
      %80 = kapy.ld_matrix %78, %cst_1 : tensor<16x64xf16, #shared>, tensor<16x64xf16, #values> -> tensor<16x64xf16, #values>
      %81 = kapy.matmul mma_m16n8k8_f16 %arg7, %arg8, %arg9 : tensor<64x16xf16, #values>, tensor<16x64xf16, #values> -> tensor<64x64xf16, #values>
      %82 = kapy.sv_shared %75[%c0_i32 : %c64_i32, %c32_i32 : %c48_i32] : tensor<64x64xf16, #shared> -> tensor<64x16xf16, #shared>
      %83 = kapy.sv_shared %76[%c32_i32 : %c48_i32, %c0_i32 : %c64_i32] : tensor<64x64xf16, #shared> -> tensor<16x64xf16, #shared>
      %84 = kapy.ld_matrix %82, %cst_0 : tensor<64x16xf16, #shared>, tensor<64x16xf16, #values> -> tensor<64x16xf16, #values>
      %85 = kapy.ld_matrix %83, %cst_1 : tensor<16x64xf16, #shared>, tensor<16x64xf16, #values> -> tensor<16x64xf16, #values>
      %86 = kapy.matmul mma_m16n8k8_f16 %79, %80, %81 : tensor<64x16xf16, #values>, tensor<16x64xf16, #values> -> tensor<64x64xf16, #values>
      %87 = kapy.sv_shared %75[%c0_i32 : %c64_i32, %c48_i32 : %c64_i32] : tensor<64x64xf16, #shared> -> tensor<64x16xf16, #shared>
      %88 = kapy.sv_shared %76[%c48_i32 : %c64_i32, %c0_i32 : %c64_i32] : tensor<64x64xf16, #shared> -> tensor<16x64xf16, #shared>
      %89 = kapy.ld_matrix %87, %cst_0 : tensor<64x16xf16, #shared>, tensor<64x16xf16, #values> -> tensor<64x16xf16, #values>
      %90 = kapy.ld_matrix %88, %cst_1 : tensor<16x64xf16, #shared>, tensor<16x64xf16, #values> -> tensor<16x64xf16, #values>
      %91 = kapy.matmul mma_m16n8k8_f16 %84, %85, %86 : tensor<64x16xf16, #values>, tensor<16x64xf16, #values> -> tensor<64x64xf16, #values>
      %92 = arith.muli %arg6, %c64_i32 : i32
      %93 = arith.addi %92, %c64_i32 : i32
      %94 = kapy.sv_global %0[%15 : %17, %92 : %93] : tensor<?x?xf16, #global> -> tensor<64x64xf16, #global>
      %95 = kapy.sv_global %1[%92 : %93, %16 : %18] : tensor<?x?xf16, #global> -> tensor<64x64xf16, #global>
      kapy.cp_async_global_to_shared %94, %75, %cst : tensor<64x64xf16, #global>, tensor<64x64xf16, #shared>, tensor<64x64xf16, #values>
      kapy.cp_async_global_to_shared %95, %76, %cst : tensor<64x64xf16, #global>, tensor<64x64xf16, #shared>, tensor<64x64xf16, #values>
      kapy.cp_async_wait_group {num_pending = 2 : i32}
      %96 = arith.select %74, %27, %25 : tensor<64x64xf16, #shared>
      %97 = arith.select %74, %28, %26 : tensor<64x64xf16, #shared>
      %98 = kapy.sv_shared %96[%c0_i32 : %c64_i32, %c0_i32 : %c16_i32] : tensor<64x64xf16, #shared> -> tensor<64x16xf16, #shared>
      %99 = kapy.sv_shared %97[%c0_i32 : %c16_i32, %c0_i32 : %c64_i32] : tensor<64x64xf16, #shared> -> tensor<16x64xf16, #shared>
      %100 = kapy.ld_matrix %98, %cst_0 : tensor<64x16xf16, #shared>, tensor<64x16xf16, #values> -> tensor<64x16xf16, #values>
      %101 = kapy.ld_matrix %99, %cst_1 : tensor<16x64xf16, #shared>, tensor<16x64xf16, #values> -> tensor<16x64xf16, #values>
      %102 = kapy.matmul mma_m16n8k8_f16 %89, %90, %91 : tensor<64x16xf16, #values>, tensor<16x64xf16, #values> -> tensor<64x64xf16, #values>
      scf.yield %100, %101, %102 : tensor<64x16xf16, #values>, tensor<16x64xf16, #values>, tensor<64x64xf16, #values>
    }
    %35 = kapy.sv_shared %25[%c0_i32 : %c64_i32, %c16_i32 : %c32_i32] : tensor<64x64xf16, #shared> -> tensor<64x16xf16, #shared>
    %36 = kapy.sv_shared %26[%c16_i32 : %c32_i32, %c0_i32 : %c64_i32] : tensor<64x64xf16, #shared> -> tensor<16x64xf16, #shared>
    %37 = kapy.ld_matrix %35, %cst_0 : tensor<64x16xf16, #shared>, tensor<64x16xf16, #values> -> tensor<64x16xf16, #values>
    %38 = kapy.ld_matrix %36, %cst_1 : tensor<16x64xf16, #shared>, tensor<16x64xf16, #values> -> tensor<16x64xf16, #values>
    %39 = kapy.matmul mma_m16n8k8_f16 %34#0, %34#1, %34#2 : tensor<64x16xf16, #values>, tensor<16x64xf16, #values> -> tensor<64x64xf16, #values>
    %40 = kapy.sv_shared %25[%c0_i32 : %c64_i32, %c32_i32 : %c48_i32] : tensor<64x64xf16, #shared> -> tensor<64x16xf16, #shared>
    %41 = kapy.sv_shared %26[%c32_i32 : %c48_i32, %c0_i32 : %c64_i32] : tensor<64x64xf16, #shared> -> tensor<16x64xf16, #shared>
    %42 = kapy.ld_matrix %40, %cst_0 : tensor<64x16xf16, #shared>, tensor<64x16xf16, #values> -> tensor<64x16xf16, #values>
    %43 = kapy.ld_matrix %41, %cst_1 : tensor<16x64xf16, #shared>, tensor<16x64xf16, #values> -> tensor<16x64xf16, #values>
    %44 = kapy.matmul mma_m16n8k8_f16 %37, %38, %39 : tensor<64x16xf16, #values>, tensor<16x64xf16, #values> -> tensor<64x64xf16, #values>
    %45 = kapy.sv_shared %25[%c0_i32 : %c64_i32, %c48_i32 : %c64_i32] : tensor<64x64xf16, #shared> -> tensor<64x16xf16, #shared>
    %46 = kapy.sv_shared %26[%c48_i32 : %c64_i32, %c0_i32 : %c64_i32] : tensor<64x64xf16, #shared> -> tensor<16x64xf16, #shared>
    %47 = kapy.ld_matrix %45, %cst_0 : tensor<64x16xf16, #shared>, tensor<64x16xf16, #values> -> tensor<64x16xf16, #values>
    %48 = kapy.ld_matrix %46, %cst_1 : tensor<16x64xf16, #shared>, tensor<16x64xf16, #values> -> tensor<16x64xf16, #values>
    %49 = kapy.matmul mma_m16n8k8_f16 %42, %43, %44 : tensor<64x16xf16, #values>, tensor<16x64xf16, #values> -> tensor<64x64xf16, #values>
    kapy.cp_async_wait_group {num_pending = 0 : i32}
    %50 = kapy.sv_shared %27[%c0_i32 : %c64_i32, %c0_i32 : %c16_i32] : tensor<64x64xf16, #shared> -> tensor<64x16xf16, #shared>
    %51 = kapy.sv_shared %28[%c0_i32 : %c16_i32, %c0_i32 : %c64_i32] : tensor<64x64xf16, #shared> -> tensor<16x64xf16, #shared>
    %52 = kapy.ld_matrix %50, %cst_0 : tensor<64x16xf16, #shared>, tensor<64x16xf16, #values> -> tensor<64x16xf16, #values>
    %53 = kapy.ld_matrix %51, %cst_1 : tensor<16x64xf16, #shared>, tensor<16x64xf16, #values> -> tensor<16x64xf16, #values>
    %54 = kapy.matmul mma_m16n8k8_f16 %47, %48, %49 : tensor<64x16xf16, #values>, tensor<16x64xf16, #values> -> tensor<64x64xf16, #values>
    %55 = kapy.sv_shared %27[%c0_i32 : %c64_i32, %c16_i32 : %c32_i32] : tensor<64x64xf16, #shared> -> tensor<64x16xf16, #shared>
    %56 = kapy.sv_shared %28[%c16_i32 : %c32_i32, %c0_i32 : %c64_i32] : tensor<64x64xf16, #shared> -> tensor<16x64xf16, #shared>
    %57 = kapy.ld_matrix %55, %cst_0 : tensor<64x16xf16, #shared>, tensor<64x16xf16, #values> -> tensor<64x16xf16, #values>
    %58 = kapy.ld_matrix %56, %cst_1 : tensor<16x64xf16, #shared>, tensor<16x64xf16, #values> -> tensor<16x64xf16, #values>
    %59 = kapy.matmul mma_m16n8k8_f16 %52, %53, %54 : tensor<64x16xf16, #values>, tensor<16x64xf16, #values> -> tensor<64x64xf16, #values>
    %60 = kapy.sv_shared %27[%c0_i32 : %c64_i32, %c32_i32 : %c48_i32] : tensor<64x64xf16, #shared> -> tensor<64x16xf16, #shared>
    %61 = kapy.sv_shared %28[%c32_i32 : %c48_i32, %c0_i32 : %c64_i32] : tensor<64x64xf16, #shared> -> tensor<16x64xf16, #shared>
    %62 = kapy.ld_matrix %60, %cst_0 : tensor<64x16xf16, #shared>, tensor<64x16xf16, #values> -> tensor<64x16xf16, #values>
    %63 = kapy.ld_matrix %61, %cst_1 : tensor<16x64xf16, #shared>, tensor<16x64xf16, #values> -> tensor<16x64xf16, #values>
    %64 = kapy.matmul mma_m16n8k8_f16 %57, %58, %59 : tensor<64x16xf16, #values>, tensor<16x64xf16, #values> -> tensor<64x64xf16, #values>
    %65 = kapy.sv_shared %27[%c0_i32 : %c64_i32, %c48_i32 : %c64_i32] : tensor<64x64xf16, #shared> -> tensor<64x16xf16, #shared>
    %66 = kapy.sv_shared %28[%c48_i32 : %c64_i32, %c0_i32 : %c64_i32] : tensor<64x64xf16, #shared> -> tensor<16x64xf16, #shared>
    %67 = kapy.ld_matrix %65, %cst_0 : tensor<64x16xf16, #shared>, tensor<64x16xf16, #values> -> tensor<64x16xf16, #values>
    %68 = kapy.ld_matrix %66, %cst_1 : tensor<16x64xf16, #shared>, tensor<16x64xf16, #values> -> tensor<16x64xf16, #values>
    %69 = kapy.matmul mma_m16n8k8_f16 %62, %63, %64 : tensor<64x16xf16, #values>, tensor<16x64xf16, #values> -> tensor<64x64xf16, #values>
    %70 = kapy.matmul mma_m16n8k8_f16 %67, %68, %69 : tensor<64x16xf16, #values>, tensor<16x64xf16, #values> -> tensor<64x64xf16, #values>
    %71 = kapy.mk_global %arg2[%c0_i32] [%arg3, %arg5] [%arg5, %c1_i32] : !kapy.ptr<1> -> tensor<?x?xf16, #global>
    %72 = kapy.sv_global %71[%15 : %17, %16 : %18] : tensor<?x?xf16, #global> -> tensor<64x64xf16, #global>
    kapy.st_global %70, %72 : tensor<64x64xf16, #values>, tensor<64x64xf16, #global>
    kapy.return
  }
}
