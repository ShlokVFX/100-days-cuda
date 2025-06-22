import torch
import time
import argparse

def generate_matrices(matrix_size, data_format):
    #print("Generating Random Matrix", flush=True)
    matrix = torch.rand(matrix_size, matrix_size, dtype=torch.float64)

    # casting fp64 tensor as needed.
    if(data_format=='fp64'):
        matrix=matrix
    elif(data_format=='fp32'):
        matrix=matrix.float()
    elif(data_format=='fp16'):
        matrix=matrix.half()
    elif(data_format=='bfloat16'):
        matrix=matrix.bfloat16()
    elif(data_format=='int64'):
        matrix=matrix.long()
    elif(data_format=='int32'):
        matrix=matrix.int()
    elif(data_format=='int16'):
        matrix=matrix.short()
    elif(data_format=='int8'):
        matrix=matrix.char()
    else:
        print("Non-supported Data Type.")
        exit()

    #print("Matrix Generation is done", flush=True)
    return(matrix)

def run_benchmark(matrix, matrix_size, useCUDA):

    # Calculating Number of Ops
    ops = 2*matrix_size**3
    # Copying matrix to GPU memory (if GPU is used)
    if(useCUDA==True):
        #print("Copying Matrix to GPU Memory", flush=True)
        matrix_GPU = matrix.cuda()
    else:
        matrix_GPU = matrix
    
    # Take a note for start time
    start = time.time()

    # Begin Multiplication
    #print("Multiplying Matrices", flush=True)
    result = torch.mm(matrix_GPU, matrix_GPU).cuda()

    # Wait operation to finish
    if(useCUDA==True):
        torch.cuda.synchronize()

    # Take a note for end time
    end = time.time()

    # Calculate Elapsed Time
    duration = end - start

    # Calculate TFLOPS
    tflops = ops / duration / 10**12
    print("{}x{} MM {} ops in {} sec = TFLOPS {}".format(matrix_size, matrix_size, ops, duration, tflops), flush=True)

    # Clean-up
    #print("Cleaning-up", flush=True)
    if(useCUDA==True):
        del matrix_GPU
        torch.cuda.empty_cache()

def cmdline_args():
    # Make parser object
    parser = argparse.ArgumentParser(description='PyTorch CUDA Matrix Multiplication Benchmark.')
    parser.add_argument("--useCUDA", type=bool, default=False, help='Use CUDA-capable GPU for Benchmark.', choices=[True, False])
    parser.add_argument("--precision", type=str, default='fp32', help='Data Precision for Benchmark', choices=['int8', 'int16', 'int32', 'int64', 'fp16', 'bfloat16', 'fp32', 'fp64'])
    parser.add_argument("--loop", type=bool, default=False, help='Run benchmark infinite times until manually cancelled.', choices=[True, False])
    parser.add_argument("--size", type=str, default='100, 500, 1000, 2000, 4000, 8000, 16000, 32000', help='List of matrices size separated by comma, e.g., 100, 500, 1000')
    return(parser.parse_args())


if __name__ == '__main__':
    
    # Parse Arguments
    try:
        args = cmdline_args()
        #print(args)
    except:
        print("Launch argument error!")
        print("Example: $python <script_name> --useCUDA=True --precision='fp32' --size='100, 500, 1000'")
        exit()

    # Parse Matrices Size
    try:
        size_list = [int(i) for i in args.size.split(",")]
    except:
        print("Invalid list of matrix size. Use only integer separated by comma to define the list of matrix size.")
        exit()

    while True:
        for matrix_size in size_list:
            matrix = generate_matrices(matrix_size,args.precision)
            run_benchmark(matrix, matrix_size, args.useCUDA)
            del matrix
        if (args.loop==False):
            break
    
   
