#include <stdio.h>
#include <stdlib.h>
#include <math.h>

__global__ void kernel(double * a, double * b, double * c, int N)
{
    // Getting the line and columns using block and thread indexes and dimensions
    int i = (blockIdx.x * blockDim.x + threadIdx.x) / N;
    int j = (blockIdx.x * blockDim.x + threadIdx.x) % N;

    // Index for cross all cases
    int index = i * N + j;
    // Performing Matrix Multiplication
    for (index; index < N * N; index += (blockDim.x * gridDim.x))
        for(int k = 0; k < N; ++k)
            c[index] += a[i * N + k] * b[k * N + j];
}

int main(int argc, char ** argv)
{
    int N = 10;
    // Contiguous 2D-Array[N][N]
    int sz_in_bytes = N * N * sizeof(double *);

    double * h_a, * h_b, * h_c;
    double * d_a, * d_b, * d_c;

    h_a = (double *)malloc(sz_in_bytes);
    h_b = (double *)malloc(sz_in_bytes);
    h_c = (double *)malloc(sz_in_bytes);


    // Initiate values on h_a and h_b
    for(int i = 0 ; i < N ; ++i)
        for (int j = 0; j < N; ++j)
        {
            h_a[i * N + j] = 200 + i;
            h_b[i * N + j] = 100 + i;
        }

    // Set the result matrix to 0
    for (int i = 0; i < N * N; ++i)
        h_c[i] = 0;

    // Allocating space in GPU for the two matrices and the result matrix
    cudaMalloc((void **)&d_a, sz_in_bytes);
    cudaMalloc((void **)&d_b, sz_in_bytes);
    cudaMalloc((void **)&d_c, sz_in_bytes);

    // Copying memory from heap to CUDA GPU
    cudaMemcpy(d_a, h_a, sz_in_bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, sz_in_bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_c, h_c, sz_in_bytes, cudaMemcpyHostToDevice);

    // Creating a block of dimension 64 and a grid of dimension 10
    dim3  dimBlock(64, 1, 1);
    dim3  dimGrid(10, 1, 1);
    // Calling kernel function by passing the grid and block dimension for the GPU parallelization
    // Passing matrices A, B and the result matrix C of dimension N
    kernel<<<dimGrid , dimBlock>>>(d_a, d_b, d_c, N);

    // Copying the result from CUDA GPU allocated C matrix to Heap allocated C matrix
    cudaMemcpy(h_c, d_c, sz_in_bytes, cudaMemcpyDeviceToHost);

    // Freeing on CUDA GPU
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);

    // Printing the result
    for (int i = 0; i < N; ++i)
    {
        for (int j = 0; j < N; ++j)
            printf("%2.2f ", h_c[i * N + j]);
        puts("");
    }

    // Freeing heap
    free(h_a);
    free(h_b);
    free(h_c);

    return EXIT_SUCCESS;
}