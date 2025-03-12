module {
  llvm.func @loop_add() -> i64 {
    %0 = llvm.mlir.constant(0 : i64) : i64
    %1 = llvm.mlir.constant(0 : i64) : i64
    %2 = llvm.mlir.constant(10 : i64) : i64
    %3 = llvm.mlir.constant(1 : i64) : i64
    llvm.br ^bb1(%1, %0 : i64, i64)
  ^bb1(%4: i64, %5: i64):  // 2 preds: ^bb0, ^bb2
    %6 = llvm.icmp "slt" %4, %2 : i64
    llvm.cond_br %6, ^bb2, ^bb3
  ^bb2:  // pred: ^bb1
    %7 = llvm.add %5, %4 : i64
    %8 = llvm.add %4, %3 : i64
    llvm.br ^bb1(%8, %7 : i64, i64)
  ^bb3:  // pred: ^bb1
    llvm.return %5 : i64
  }
  llvm.func @main() -> i32 {
    %0 = llvm.call @loop_add() : () -> i64
    %1 = llvm.trunc %0 : i64 to i32
    llvm.return %1 : i32
  }
}

