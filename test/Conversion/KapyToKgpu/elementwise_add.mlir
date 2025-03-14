#global = #kapy.encoding<global_memory>
#values = #kapy.encoding<register_file>
module attributes {kapy.num_warps = 2 : i64, kapy.nvidia_cc = 89 : i64} {
  kapy.func public @elementwise_add_kernel(%arg0: !kapy.ptr<1> {kapy.alignment = 128 : i64}, %arg1: !kapy.ptr<1> {kapy.alignment = 128 : i64}, %arg2: !kapy.ptr<1> {kapy.alignment = 128 : i64}, %arg3: i32 {kapy.alignment = 512 : i32}) {
    %c0_i32 = arith.constant 0 : i32
    %c1_i32 = arith.constant 1 : i32
    %c256_i32 = arith.constant 256 : i32
    %c512_i32 = arith.constant 512 : i32
    %0 = kapy.mk_global %arg0[%c0_i32] [%c1_i32, %arg3] [%arg3, %c1_i32] : !kapy.ptr<1> -> tensor<?x?xf16, #global>
    %1 = kapy.mk_global %arg1[%c0_i32] [%c1_i32, %arg3] [%arg3, %c1_i32] : !kapy.ptr<1> -> tensor<?x?xf16, #global>
    %2 = kapy.program_id {axis = 0 : i32} : i32
    %3 = kapy.warp_id : i32
    %4 = arith.muli %2, %c512_i32 : i32
    %5 = arith.muli %3, %c256_i32 : i32
    %6 = arith.addi %4, %5 : i32
    %7 = arith.addi %6, %c256_i32 : i32
    %8 = kapy.sv_global %0[%c0_i32 : %c1_i32, %6 : %7] : tensor<?x?xf16, #global> -> tensor<1x256xf16, #global>
    %9 = kapy.sv_global %1[%c0_i32 : %c1_i32, %6 : %7] : tensor<?x?xf16, #global> -> tensor<1x256xf16, #global>
    %10 = kapy.ld_global %8 : tensor<1x256xf16, #global> -> tensor<1x256xf16, #values>
    %11 = kapy.ld_global %9 : tensor<1x256xf16, #global> -> tensor<1x256xf16, #values>
    %12 = arith.addf %10, %11 : tensor<1x256xf16, #values>
    %13 = kapy.mk_global %arg2[%c0_i32] [%c1_i32, %arg3] [%arg3, %c1_i32] : !kapy.ptr<1> -> tensor<?x?xf16, #global>
    %14 = kapy.sv_global %13[%c0_i32 : %c1_i32, %6 : %7] : tensor<?x?xf16, #global> -> tensor<1x256xf16, #global>
    kapy.st_global %12, %14 : tensor<1x256xf16, #values>, tensor<1x256xf16, #global>
    kapy.return
  }
}
