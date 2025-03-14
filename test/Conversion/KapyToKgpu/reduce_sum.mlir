#global = #kapy.encoding<global_memory>
#values = #kapy.encoding<register_file>
module attributes {kapy.num_warps = 2 : i64, kapy.nvidia_cc = 89 : i64} {
  kapy.func public @reduce_sum_kernel(%arg0: !kapy.ptr<1> {kapy.alignment = 128 : i64}, %arg1: !kapy.ptr<1> {kapy.alignment = 128 : i64}, %arg2: i32 {kapy.alignment = 4 : i64}) {
    %c0_i32 = arith.constant 0 : i32
    %c1_i32 = arith.constant 1 : i32
    %c2_i32 = arith.constant 2 : i32
    %c4_i32 = arith.constant 4 : i32
    %c256_i32 = arith.constant 256 : i32
    %0 = kapy.mk_global %arg0[%c0_i32] [%arg2, %c256_i32] [%c256_i32, %c1_i32] : !kapy.ptr<1> -> tensor<?x?xf16, #global>
    %1 = kapy.program_id {axis = 0 : i32} : i32
    %2 = kapy.warp_id : i32
    %3 = arith.muli %1, %c4_i32 : i32
    %4 = arith.muli %2, %c2_i32 : i32
    %5 = arith.addi %3, %4 : i32
    %6 = arith.addi %5, %c2_i32 : i32
    %7 = kapy.sv_global %0[%5 : %6, %c0_i32 : %c256_i32] : tensor<?x?xf16, #global> -> tensor<2x256xf16, #global>
    %8 = kapy.ld_global %7 : tensor<2x256xf16, #global> -> tensor<2x256xf16, #values>
    %9 = kapy.reduce %8 {axis = 1 : i32} lambda(%arg3: f16, %arg4: f16) {
      %12 = arith.addf %arg3, %arg4 : f16
      kapy.return %12 : f16
    } : tensor<2x256xf16, #values> -> tensor<2x1xf16, #values>
    %10 = kapy.mk_global %arg1[%c0_i32] [%arg2, %c1_i32] [%c1_i32, %arg2] : !kapy.ptr<1> -> tensor<?x?xf16, #global>
    %11 = kapy.sv_global %10[%5 : %6, %c0_i32 : %c1_i32] : tensor<?x?xf16, #global> -> tensor<2x1xf16, #global>
    kapy.st_global %9, %11 : tensor<2x1xf16, #values>, tensor<2x1xf16, #global>
    kapy.return
  }
}
