#include <stdio.h>
#include <cuda.h>

#include "common.h"

#define ITERS 10
#define BLOCK_SIZE 32

__global__
void MatrixVectorKernel(float *A, float *B, float *C, int nrow, int ncol)
{
    int row = blockDim.x * blockIdx.x + threadIdx.x;
    
    float aval = 0.0f;
    for (int col = 0; col < ncol; ++ col)
    {
        aval += B[row * ncol + col] * C[col]; 
    }

    A[row] = aval;
}

int main(int argc, char **argv)
{
    if (argc != 3)
    {
        fprintf(stderr, "Usage: ./matvec <nrow> <ncol>\n");
        return;
    }

    int nrow = atoi(argv[1]);
    int ncol = atoi(argv[2]);

    // Prepare data
    float *A_h = (float *) malloc(nrow * sizeof(float));
    float *B_h = (float *) malloc(nrow * ncol * sizeof(float));
    float *C_h = (float *) malloc(ncol * sizeof(float));

    for (int i = 0; i < nrow; ++ i)
    {
        for (int j = 0; j < ncol; ++ j)
        {
            B_h[i * ncol + j] = (i + 1.0) / (j + 1.0);
        }
    }

    for (int j = 0; j < ncol; ++ j)
    {
        C_h[j] = j + 1.0;
    }

    // Copy to device
    float *A_d, *B_d, *C_d;
    cudaMalloc((void **) &A_d, nrow * sizeof(float));
    cudaMalloc((void **) &B_d, nrow * ncol * sizeof(float));
    cudaMalloc((void **) &C_d, ncol * sizeof(float));
    
    // Benchmark
    double start = cpuSecond();
    for (int i = 0; i < ITERS; ++ i)
    {
        MatrixVectorKernel<<<ceil((double) ncol / BLOCK_SIZE), BLOCK_SIZE>>>(A_d, B_d, C_d, nrow, ncol);   
    }
    cudaDeviceSynchronize();
    double tavg = (cpuSecond() - start) / ITERS;

    double FLOPS = 2.0 * nrow * ncol / tavg;
    printf("Kernel execution time:  %10.5lf s\n", tavg);
    printf("Performance:            %10.5lf GLOPS\n", FLOPS / 1e9);

    // Copy results to host
    cudaMemcpy(A_h, A_d, nrow * sizeof(float), cudaMemcpyDeviceToHost);
    
    // Free memory
    cudaFree(A_d);
    cudaFree(B_d);
    cudaFree(C_d);

    return 0;
}