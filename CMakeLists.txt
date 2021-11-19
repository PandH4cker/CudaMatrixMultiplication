cmake_minimum_required(VERSION 3.20)
project(CudaMatrixMultiplication CUDA)

set(CMAKE_CUDA_STANDARD 14)

add_executable(CudaMatrixMultiplication main.cu)

set_target_properties(CudaMatrixMultiplication PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON)
