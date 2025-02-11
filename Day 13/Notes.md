Input : {1.0, 2.0, 3.0, 4.0,  5.0, 6.0, 7.0, 8.0};
Speeds Comparison:

Optmized GPU SoftMax that uses Warp-Level Reduction implementation:

0.0320586 0.0871443 0.236883 0.643914 
0.0320586 0.0871443 0.236883 0.643914 

Execution time: 3.93114 ms
GFLOPS: 4.07007e-06 GFLOPS

Softmax Output (FP16 Accelerated):
0.0320435 0.0900269 0.268799 1 
0.0320435 0.0900269 0.268799 1 

Execution time: 2.75968 ms
GFLOPS: 5.79777e-06 GFLOPS

🚀 Result: 42.46% Speedup with FP16 Acceleration 🚀