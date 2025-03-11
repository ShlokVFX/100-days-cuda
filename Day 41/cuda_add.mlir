module {
  func.func @matrix_add(%arg0: memref<1024xf32>, %arg1: memref<1024xf32>, %arg2: memref<1024xf32>) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c1024 = arith.constant 1024 : index

    scf.for %i = %c0 to %c1024 step %c1 {
      %a = memref.load %arg0[%i] : memref<1024xf32>
      %b = memref.load %arg1[%i] : memref<1024xf32>
      %sum = arith.addf %a, %b : f32
      memref.store %sum, %arg2[%i] : memref<1024xf32>
    }
    return
  }
}
