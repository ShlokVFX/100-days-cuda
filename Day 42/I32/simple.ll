; ModuleID = 'LLVMDialectModule'
source_filename = "LLVMDialectModule"

define i64 @loop_add() {
  br label %1

1:                                                ; preds = %5, %0
  %2 = phi i64 [ %7, %5 ], [ 0, %0 ]
  %3 = phi i64 [ %6, %5 ], [ 0, %0 ]
  %4 = icmp slt i64 %2, 10
  br i1 %4, label %5, label %8

5:                                                ; preds = %1
  %6 = add i64 %3, %2
  %7 = add i64 %2, 1
  br label %1

8:                                                ; preds = %1
  ret i64 %3
}

define i32 @main() {
  %1 = call i64 @loop_add()
  %2 = trunc i64 %1 to i32
  ret i32 %2
}

!llvm.module.flags = !{!0}

!0 = !{i32 2, !"Debug Info Version", i32 3}
