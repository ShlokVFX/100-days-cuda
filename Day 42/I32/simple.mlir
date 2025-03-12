func.func @loop_add() -> (index) {
    %init = index.constant 0
    %lb = index.constant 0
    %ub = index.constant 10
    %step = index.constant 1

    %sum = scf.for %iv = %lb to %ub step %step iter_args(%acc = %init) -> (index) {
        %sum_next = arith.addi %acc, %iv : index
        scf.yield %sum_next : index
    }

    return %sum : index
}

func.func @main() -> i32 {
    %out = call @loop_add() : () -> index
    %out_i32 = arith.index_cast %out : index to i32
    func.return %out_i32 : i32
}