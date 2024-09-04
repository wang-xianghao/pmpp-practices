#include <stdio.h>
#include <cuda.h>

#include "common.h"

#define ITERS 10
#define BLOCK_ROW 16
#define BLOCK_COL 16

// Each thread produces one cell of output
__global__
void MatrixMulKernel(float *M, float *N, float *P, int width)
{
    int col = blockDim.x * blockIdx.x + threadIdx.x;
    int row = blockDim.y * blockIdx.y + threadIdx.y;

    if (row >= width || col >= width)
        return;

    float pval = 0.0f;
    for (int k = 0; k < width; ++ k)
    {
        pval += M[row * width + k] * N[k * width + col];
    }

    P[row * width + col] = pval;
}

// Each thread produces one row of output
__global__
void MatrixMulRowKernel(float *M, float *N, float *P, int width)
{
    int row = blockDim.x * blockIdx.x + threadIdx.x;

    if (row >= width)
        return;

    for (int col = 0; col < width; ++ col)
    {
        float pval = 0.0f;
        for (int k = 0; k < width; ++ k)
        {
            pval += M[row * width + k] * N[k * width + col];
        }
        P[row * width + col] = pval;
    }
}

// Each thread produces one column of output
__global__
void MatrixMulColKernel(float *M, float *N, float *P, int width)
{
    int col = blockDim.x * blockIdx.x + threadIdx.x;

    if (col >= width)
        return;

    for (int row = 0; row < width; ++ row)
    {
        float pval = 0.0f;
        for (int k = 0; k < width; ++ k)
        {
            pval += M[row * width + k] * N[k * width + col];
        }
        P[row * width + col] = pval;
    }
}

int main(int argc, char **argv)
{
    if (argc != 2)
    {
        fprintf(stderr, "Usage: ./matmul <width>\n");
        return;
    }

    int width = atoi(argv[1]);
    int N = width * width;

    dim3 dimGrid(ceil(1.0 * width / BLOCK_COL), ceil(1.0 * width / BLOCK_ROW), 1);
    dim3 dimBlock(BLOCK_COL, BLOCK_ROW, 1);

    // Prepare arguments
    float *M_h = (float *) malloc(N * sizeof(float));
    float *N_h = (float *) malloc(N * sizeof(float));
    float *P_h = (float *) malloc(N * sizeof(float));
    for (int i = 0; i < width; ++ i)
    {
        for (int j = 0; j < width; ++ j)
        {
            M_h[i * width + j] = (i + j) / 2.0;
            N_h[i * width + j] = (j + i) / 2.0;
        }
    }

    // Copy to device
    float *M_d, *N_d, *P_d;
    cudaMalloc((void **) &M_d, N * sizeof(float));
    cudaMalloc((void **) &N_d, N * sizeof(float));
    cudaMalloc((void **) &P_d, N * sizeof(float));

    // Kernel execution
    double callAvg = 0.0;
    double kernelAvg = 0.0;

    for (int i = 0; i < ITERS; ++ i)
    {
        double start = cpuSecond();
        MatrixMulKernel<<<dimGrid, dimBlock>>>(M_d, N_d, P_d, width);
        double callEnd = cpuSecond();
        cudaDeviceSynchronize();
        double kernelEnd = cpuSecond();

        callAvg += callEnd - start;
        kernelAvg += kernelEnd - start;
    }

    callAvg /= ITERS;
    kernelAvg /= ITERS;

    // Print time
    double FLOPS = 2.0 * width * width * width / kernelAvg;

    printf("Kernel launch time:     %10.6lf s\n", callAvg);
    printf("Kernel running time:    %10.6lf s\n", kernelAvg);
    printf("Performance:            %10.6lf GFLOPS\n", FLOPS / 1e9);

    // Compare matmul organized by row or column
    const int BLOCK_SIZE = 32;
    
    double tRow = 0.0, tCol = 0.0;
    for (int i = 0; i < ITERS; ++ i)
    {
        double start = cpuSecond();
        MatrixMulRowKernel<<<ceil(1.0 * width / BLOCK_SIZE), BLOCK_SIZE>>>(M_d, N_d, P_d, width);
        cudaDeviceSynchronize();
        tRow += (cpuSecond() - start) / ITERS;

        start = cpuSecond();
        MatrixMulColKernel<<<ceil(1.0 * width / BLOCK_SIZE), BLOCK_SIZE>>>(M_d, N_d, P_d, width);
        cudaDeviceSynchronize();
        tCol += (cpuSecond() - start) / ITERS;
    }
    
    printf("\n\n");
    printf("Row Matmul time:    %10.6lf s\n", tRow);
    printf("Col Matmul time:    %10.6lf s\n", tCol);

    
    // Copy results to host
    cudaMemcpy(P_h, P_d, N * sizeof(float), cudaMemcpyDeviceToHost);
    
    // Free memory
    cudaFree(M_d);
    cudaFree(N_d);
    cudaFree(P_d);

    return 0;
}