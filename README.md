# CudaMatrixMultiplication
## Author: [Dray Raphael](https://www.linkedin.com/in/raphaeldray/)

Matrix Multiplication using **Nvidia Cuda** cores.

**Nvidia GPU/Accelerator** is required to run this program.

## Compile:
You will have to install the 
**Nvidia Cuda Development Toolkit** and to use **nvcc**:
```shell
nvcc main.cu -o CudaMatrixMultiplication
```

Or using **CMake**:
```shell
mkdir build
cd build
cmake --build ..
```

## Usage:
```shell
./CudaMatrixMultiplication
```
