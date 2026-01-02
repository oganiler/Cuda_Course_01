#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <cstdio>
#include "device_info.h"

const int N = 64;
const int ARRAY_BYTES_INT = N * sizeof(int);//allocation size

void* cpu_p;
void* gpu_p;

__device__ unsigned int thrednum = 0;
__device__ unsigned int launchnum = 0;

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
        int_p[i] = i + 1;
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

__global__ void basic_gpu_increment_kernel_1B(int* g_data)
{
    int idx = threadIdx.x;
    unsigned int current_thrednum = atomicAdd(&thrednum, 1);

    if (threadIdx.x == 0) {
        printf("blockIdx=%d  gridDim=%d  blockDim=%d\n",
            blockIdx.x, gridDim.x, blockDim.x);
    }

    if (idx < N)
    {
        unsigned int current_launchnum = atomicAdd(&launchnum, 1);
        printf("launchnum=%u block=%d thread=%d idx=%d gridDim=%d blockDim=%d thrednum=%u\n",
            current_launchnum, blockIdx.x, threadIdx.x, idx, gridDim.x, blockDim.x, current_thrednum);

        //launch
        g_data[idx] *= 2;
    }

    //printf("thrednum=%u block=%d thread=%d idx=%d gridDim=%d blockDim=%d\n",
    //    current_thrednum, blockIdx.x, threadIdx.x, idx, gridDim.x, blockDim.x);
}

__global__ void basic_gpu_increment_kernel_MB(int* g_data, int N)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    unsigned int current_thrednum = atomicAdd(&thrednum, 1);

    if (threadIdx.x == 0) {
        printf("blockIdx=%d  gridDim=%d  blockDim=%d\n",
            blockIdx.x, gridDim.x, blockDim.x);
    }

    if (idx < N) {
        unsigned int current_launchnum = atomicAdd(&launchnum, 1);
        printf("launchnum=%u block=%d thread=%d idx=%d gridDim=%d blockDim=%d thrednum=%u\n",
            current_launchnum, blockIdx.x, threadIdx.x, idx, gridDim.x, blockDim.x, current_thrednum);

        //launch
        g_data[idx] *= 2;
    }

    //printf("thrednum=%u block=%d thread=%d idx=%d gridDim=%d blockDim=%d\n",
    //    current_thrednum, blockIdx.x, threadIdx.x, idx, gridDim.x, blockDim.x);
}


//Use a grid - stride loop, so you can launch “enough blocks to keep the GPU busy” regardless of N
__global__ void basic_gpu_increment_kernel_stride(int* g_data, int N)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    if (threadIdx.x == 0) {
        printf("blockIdx=%d  gridDim=%d  blockDim=%d\n",
            blockIdx.x, gridDim.x, blockDim.x);
    }

    unsigned int current_thrednum = atomicAdd(&thrednum, 1);

    for (; idx < N; idx += stride) {
        unsigned int current_launchnum = atomicAdd(&launchnum, 1);
        printf("launchnum=%u block=%d thread=%d idx=%d gridDim=%d blockDim=%d stride=%d thrednum=%u\n",
            current_launchnum, blockIdx.x, threadIdx.x, idx, gridDim.x, blockDim.x, stride, current_thrednum);

        //launch
        g_data[idx] *= 2;
    }

    //printf("thrednum=%u block=%d thread=%d idx=%d gridDim=%d blockDim=%d\n",
    //    current_thrednum, blockIdx.x, threadIdx.x, idx, gridDim.x, blockDim.x);
}

void print_cpu_numbers()
{
    int* int_p = (int*)cpu_p;
    for (int i = 0; i < N; i++)
    {
        std::cout << "Number " << i + 1 << " : " << int_p[i] << std::endl;
    }
}

int main()
{
    PrintCudaDeviceInfo();

    // Optional: increase printf buffer if you print a lot
    cudaDeviceSetLimit(cudaLimitPrintfFifoSize, 8 * 1024 * 1024);  // 8MB :contentReference[oaicite:2]{index=2}

	cpu_alloc();
	cpu_set_numbers();

	gpu_alloc();
	cpu_memory_to_gpu_memory();

    //On current GPUs, a thread block may contain up to 1024 threads.
    // 		dim3 threadsPerBlock(16, 16); // A thread block size of 16x16 (256 threads), although arbitrary in this case, is a common choice
    //      dim3 numBlocks(N / threadsPerBlock.x, N / threadsPerBlock.y);
    //cluster_kernel<<<numBlocks, threadsPerBlock>>>(input, output);
	//basic_gpu_increment_kernel <<<2, N >>> ((int*)gpu_p);

	//--------------------------- calculate occupancy ---------------------------
    int activeBlocksPerSM;        // Occupancy in terms of active blocks
    int blockSize = N/2;
    size_t dynamicSmemBytes = 0;

    // These variables are used to convert occupancy to warps
    int device;
    cudaDeviceProp prop;
    int activeWarps;
    int maxWarpsPerSM;

    cudaGetDevice(&device);
    cudaGetDeviceProperties(&prop, device);

    cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        &activeBlocksPerSM,
        basic_gpu_increment_kernel_MB,
        blockSize,
        dynamicSmemBytes);

    activeBlocksPerSM = 1;
    int warpsPerBlock = (blockSize + prop.warpSize - 1) / prop.warpSize;
    int activeWarpsPerSM = activeBlocksPerSM * warpsPerBlock;
    activeWarps = activeBlocksPerSM * blockSize / prop.warpSize;
    maxWarpsPerSM = prop.maxThreadsPerMultiProcessor / prop.warpSize;
    float occ = (float)activeWarpsPerSM / (float)maxWarpsPerSM;

    std::cout << "N: " << N << std::endl;
    std::cout << "blockSize (threadsPerBlock given): " << blockSize << std::endl;
    std::cout << "activeBlocksPerSM  (calculated by NVIDIA): " << activeBlocksPerSM << std::endl;
    std::cout << "warpSize: " << prop.warpSize << std::endl;
    std::cout << "maxThreadsPerMultiProcessor: " << prop.maxThreadsPerMultiProcessor << std::endl;
    std::cout << "maxWarpsPerSM: " << maxWarpsPerSM << " (prop.maxThreadsPerMultiProcessor / prop.warpSize)" << std::endl  << std::endl;

    std::cout << "activeBlocksPerSM  * blockSize: " << activeBlocksPerSM * blockSize << std::endl;
    std::cout << "activeWarps: " << activeWarps << " (activeBlocksPerSM  * blockSize / prop.warpSize)" << std::endl;
    std::cout << "Occupancy: " << (double)activeWarps / maxWarpsPerSM * 100 << "%" << " (activeWarps / maxWarpsPerSM)" << std::endl << std::endl;

    std::cout << "warpsPerBlock: " << warpsPerBlock << " ((blockSize + prop.warpSize - 1) / prop.warpSize)" << std::endl;
    std::cout << "activeWarpsPerSM: " << activeWarpsPerSM << " (activeBlocksPerSM * warpsPerBlock)" << std::endl;
    std::cout << "Occupancy: " << occ * 100 << "%" << " (activeWarpsPerSM / maxWarpsPerSM)" << std::endl;

    // (Optional) total resident warps on the whole GPU:
    std::cout << "total resident warps (device-wide, theoretical): " << activeWarpsPerSM * prop.multiProcessorCount << std::endl << std::endl;
    

    //basic_gpu_increment_kernel << <activeBlocksPerSM , blockSize >> > ((int*)gpu_p);
    basic_gpu_increment_kernel_MB << <activeBlocksPerSM, blockSize >> > ((int*)gpu_p, N);

    //--------------------------- calculate occupancy ---------------------------

	cudaError_t result = cudaDeviceSynchronize();
	assert(result == cudaSuccess);

	gpu_memory_to_cpu_memory();
    print_cpu_numbers();

	gpu_free();
	cpu_free();

	int rc = getchar();

    return 0;
}

