#include <stdio.h>
#include <stdlib.h>
#include <math.h>

__global__ void kernel(double * a, double * b, double * c, int N)
{
    int i = (blockIdx.x * blockDim.x + threadIdx.x) / N;
    int j = (blockIdx.x * blockDim.x + threadIdx.x) % N;

    int index = i * N + j;
    for (index; index < N * N; index += (blockDim.x * gridDim.x))
        for(int k = 0; k < N; ++k)
            c[index] += a[i * N + k] * b[k * N + j];
}

int main(int argc, char ** argv)
{
    int N = 10;
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

    for (int i = 0; i < N * N; ++i)
        h_c[i] = 0;


    cudaMalloc((void **)&d_a, sz_in_bytes);
    cudaMalloc((void **)&d_b, sz_in_bytes);
    cudaMalloc((void **)&d_c, sz_in_bytes);

    cudaMemcpy(d_a, h_a, sz_in_bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, sz_in_bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_c, h_c, sz_in_bytes, cudaMemcpyHostToDevice);


    dim3  dimBlock(64, 1, 1);
    dim3  dimGrid(10, 1, 1);
    kernel<<<dimGrid , dimBlock>>>(d_a, d_b, d_c, N);

    cudaMemcpy(h_c, d_c, sz_in_bytes, cudaMemcpyDeviceToHost);

    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);

    for (int i = 0; i < N; ++i)
    {
        for (int j = 0; j < N; ++j)
            printf("%2.2f ", h_c[i * N + j]);
        puts("");
    }

    free(h_a);
    free(h_b);
    free(h_c);

    return EXIT_SUCCESS;
}