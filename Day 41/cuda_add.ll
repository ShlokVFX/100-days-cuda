; ModuleID = 'LLVMDialectModule'
source_filename = "LLVMDialectModule"

define void @matrix_add(ptr %0, ptr %1, i64 %2, i64 %3, i64 %4, ptr %5, ptr %6, i64 %7, i64 %8, i64 %9, ptr %10, ptr %11, i64 %12, i64 %13, i64 %14) {
  %16 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } poison, ptr %10, 0
  %17 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %16, ptr %11, 1
  %18 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %17, i64 %12, 2
  %19 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %18, i64 %13, 3, 0
  %20 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %19, i64 %14, 4, 0
  %21 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } poison, ptr %5, 0
  %22 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %21, ptr %6, 1
  %23 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %22, i64 %7, 2
  %24 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %23, i64 %8, 3, 0
  %25 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %24, i64 %9, 4, 0
  %26 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } poison, ptr %0, 0
  %27 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %26, ptr %1, 1
  %28 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %27, i64 %2, 2
  %29 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %28, i64 %3, 3, 0
  %30 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %29, i64 %4, 4, 0
  br label %31

31:                                               ; preds = %34, %15
  %32 = phi i64 [ %44, %34 ], [ 0, %15 ]
  %33 = icmp slt i64 %32, 1024
  br i1 %33, label %34, label %45

34:                                               ; preds = %31
  %35 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %30, 1
  %36 = getelementptr float, ptr %35, i64 %32
  %37 = load float, ptr %36, align 4
  %38 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %25, 1
  %39 = getelementptr float, ptr %38, i64 %32
  %40 = load float, ptr %39, align 4
  %41 = fadd float %37, %40
  %42 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %20, 1
  %43 = getelementptr float, ptr %42, i64 %32
  store float %41, ptr %43, align 4
  %44 = add i64 %32, 1
  br label %31

45:                                               ; preds = %31
  ret void
}

!llvm.module.flags = !{!0}

!0 = !{i32 2, !"Debug Info Version", i32 3}
