#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
//#include "device_info.h"

const int N = 1024;
const int ARRAY_BYTES_INT = N * sizeof(int);//allocation size

void* cpu_p;
void* gpu_p;

void cpu_alloc()
{
    cpu_p = malloc(ARRAY_BYTES_INT);
    assert(cpu_p != nullptr);
}

void cpu_set_numbers()
{
    int* int_p = (int*)cpu_p;
    for (int i = 0; i < N; i++)
    {
        int_p[i] = i;
    }
}

void cpu_free()
{
    free(cpu_p);
    cpu_p = nullptr;
}

void gpu_alloc()
{
    cudaError_t err = cudaMalloc(&gpu_p, ARRAY_BYTES_INT);
    assert(err == cudaSuccess);
}

void gpu_free()
{
    cudaError_t err = cudaFree(gpu_p);
    assert(err == cudaSuccess);
    gpu_p = nullptr;
}

void cpu_memory_to_gpu_memory()
{
    cudaError_t err = cudaMemcpy(gpu_p, cpu_p, ARRAY_BYTES_INT, cudaMemcpyHostToDevice);
    assert(err == cudaSuccess);
}

void gpu_memory_to_cpu_memory()
{
    cudaError_t err = cudaMemcpy(cpu_p, gpu_p, ARRAY_BYTES_INT, cudaMemcpyDeviceToHost);
    assert(err == cudaSuccess);
}

__global__ void basic_gpu_increment_kernel(int* g_data)
{
    int idx = threadIdx.x;
    if (idx < N)
    {
        g_data[idx] *= 2;
    }
}

void print_cpu_numbers()
{
    int* int_p = (int*)cpu_p;
    for (int i = 0; i < N; i++)
    {
        std::cout << "Number " << i << " : " << int_p[i] << std::endl;
    }
}

int main()
{
    //PrintCudaDeviceInfo();

	cpu_alloc();
	cpu_set_numbers();

	gpu_alloc();
	cpu_memory_to_gpu_memory();
	basic_gpu_increment_kernel <<<1, N >>> ((int*)gpu_p);

	cudaError_t result = cudaDeviceSynchronize();
	assert(result == cudaSuccess);

	gpu_memory_to_cpu_memory();
    print_cpu_numbers();

	gpu_free();
	cpu_free();

	int rc = getchar();

    return 0;
}

