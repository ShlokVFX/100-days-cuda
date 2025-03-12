// Add this line at the top to include the `arith` dialect
module {
  func.func @loop_add() -> f32 {
    %init = arith.constant 0.0 : f32
    %lb = arith.constant 0 : index
    %ub = arith.constant 10 : index
    %step = arith.constant 1 : index

    %sum = scf.for %iv = %lb to %ub step %step iter_args(%acc = %init) -> (f32) {
      %iv_f32 = arith.index_cast %iv : index to f32
      %sum_next = arith.addf %acc, %iv_f32 : f32
      scf.yield %sum_next : f32
    }

    return %sum : f32
  }

  func.func @main() -> f32 {
    %out = call @loop_add() : () -> f32
    return %out : f32
  }
}
