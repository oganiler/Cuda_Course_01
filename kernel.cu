#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <cstdio>
#include "device_info.h"

const int N = 128 * 1024;
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

    //if (threadIdx.x == 0) {
    //    printf("blockIdx=%d  gridDim=%d  blockDim=%d\n",
    //        blockIdx.x, gridDim.x, blockDim.x);
    //}

    if (idx < N)
    {
        unsigned int current_launchnum = atomicAdd(&launchnum, 1);

        //launch
        int current_data = g_data[idx];
        g_data[idx] *= 2;
        int current_updated_data = g_data[idx];

        //printf("launchnum=%u data=%d --> %d |block=%d thread=%d idx=%d gridDim=%d blockDim=%d thrednum=%u|\n",
        //    current_launchnum, current_data, current_updated_data, blockIdx.x, threadIdx.x, idx, gridDim.x, blockDim.x, current_thrednum);
    }

    //printf("thrednum=%u block=%d thread=%d idx=%d gridDim=%d blockDim=%d\n",
    //    current_thrednum, blockIdx.x, threadIdx.x, idx, gridDim.x, blockDim.x);
}

// So it can only update all elements if:gridDim.x * blockDim.x >= N when cudaOccupancyMaxActiveBlocksPerMultiprocessor is not used
// optionally "int numBlocksNeeded = (N + blockSize - 1) / blockSize" can be used to calculate the number of blocks needed without stride (simply N / blockSize if N is divisible by blockSize)
__global__ void basic_gpu_increment_kernel_MB(int* g_data, int N)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    unsigned int current_thrednum = atomicAdd(&thrednum, 1);

    if (threadIdx.x == 0) {
        printf("blockIdx=%d  gridDim=%d  blockDim=%d N=%d gridDim.x * blockDim.x = %d\n",
            blockIdx.x, gridDim.x, blockDim.x, N, gridDim.x * blockDim.x);
    }

    //if (blockIdx.x == 0 && threadIdx.x == 16) {
    //    printf("");
    //}

    if (idx < N) {
        unsigned int current_launchnum = atomicAdd(&launchnum, 1);

        //launch
        int current_data = g_data[idx];
        g_data[idx] *= 2;
        int current_updated_data = g_data[idx];

        //printf("launchnum=%u data=%d --> %d |block=%d thread=%d idx=%d gridDim=%d blockDim=%d thrednum=%u|\n",
        //    current_launchnum, current_data, current_updated_data, blockIdx.x, threadIdx.x, idx, gridDim.x, blockDim.x, current_thrednum);
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

    unsigned int current_thrednum = atomicAdd(&thrednum, 1) + 1;

    for (; idx < N; idx += stride) {
        unsigned int current_launchnum = atomicAdd(&launchnum, 1) + 1;

        //launch
        int current_data = g_data[idx];
        g_data[idx] *= 2;
		int current_updated_data = g_data[idx];

        //printf("launchnum=%u data=%d --> %d |block=%d thread=%d idx=%d gridDim=%d blockDim=%d stride=%d thrednum=%u|\n",
        //    current_launchnum, current_data, current_updated_data, blockIdx.x, threadIdx.x, idx, gridDim.x, blockDim.x, stride, current_thrednum);
    }

    //printf("thrednum=%u block=%d thread=%d idx=%d gridDim=%d blockDim=%d\n",
    //    current_thrednum, blockIdx.x, threadIdx.x, idx, gridDim.x, blockDim.x);
}

void print_cpu_numbers(bool all_elements, bool between, int firstN_element, int lastN_element)
{
    int* int_p = (int*)cpu_p;

    if(all_elements)
    {
        for (int i = 0; i < N; i++)
        {
            std::cout << "Number " << i + 1 << " : " << int_p[i] << std::endl;
        }
	}
    else
    {
        if (between)
        {
            for (int i = firstN_element; i <= lastN_element; i++)
            {
                std::cout << "Number " << i + 1 << " : " << int_p[i] << std::endl;
            }
        }
        else
        {
            for (int i = 0; i < firstN_element; i++)
            {
                std::cout << "Number " << i + 1 << " : " << int_p[i] << std::endl;
            }

            for (int i = N - lastN_element - 1; i < N; i++)
            {
                std::cout << "Number " << i + 1 << " : " << int_p[i] << std::endl;
            }
        }

    }

}

void printOutOccupancy(cudaDeviceProp &prop, int number_of_elements, int blockSize, int gridSize, std::string title)
{
	std::cout << "----- " << title << " -----" << std::endl;
    int warpsPerBlock = (blockSize + prop.warpSize - 1) / prop.warpSize;
    int activeWarpsPerSM = gridSize * warpsPerBlock;
    int activeWarps = gridSize * blockSize / prop.warpSize;
    int maxWarpsPerSM = prop.maxThreadsPerMultiProcessor / prop.warpSize;
    float occ = (float)activeWarpsPerSM / (float)maxWarpsPerSM;
    std::cout << "Number of elements: " << number_of_elements << std::endl;
    std::cout << "blockSize (threadsPerBlock): " << blockSize << std::endl;
    std::cout << "gridSize  (Number of Blocks): " << gridSize << std::endl;
    std::cout << "Total Cover (gridSize  * blockSize): " << gridSize * blockSize << std::endl;
    std::cout << "Total Cover : % " << static_cast<float>((gridSize * blockSize))/static_cast<float>(N)*100 << std::endl;
    std::cout << "warpSize: " << prop.warpSize << std::endl;
    std::cout << "maxThreadsPerMultiProcessor: " << prop.maxThreadsPerMultiProcessor << std::endl;
    std::cout << "maxWarpsPerSM: " << maxWarpsPerSM << " (prop.maxThreadsPerMultiProcessor / prop.warpSize)" << std::endl << std::endl;

    std::cout << "activeWarps: " << activeWarps << " (gridSize  * blockSize / prop.warpSize)" << std::endl;
    std::cout << "Occupancy: " << (double)activeWarps / maxWarpsPerSM * 100 << "%" << " (activeWarps / maxWarpsPerSM)" << std::endl << std::endl;

    std::cout << "warpsPerBlock: " << warpsPerBlock << " ((blockSize + prop.warpSize - 1) / prop.warpSize)" << std::endl;
    std::cout << "activeWarpsPerSM: " << activeWarpsPerSM << " (gridSize * warpsPerBlock)" << std::endl;
    std::cout << "Occupancy: " << occ * 100 << "%" << " (activeWarpsPerSM / maxWarpsPerSM)" << std::endl << std::endl;
    // (Optional) total resident warps on the whole GPU:
    std::cout << "total resident warps (device-wide, theoretical): " << activeWarpsPerSM * prop.multiProcessorCount << std::endl << std::endl;
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
    
    size_t dynamicSmemBytes = 0;//dynamicSMemSize: per - block dynamic shared memory you intend to use(bytes)
	size_t blockSizeLimit = 0;//blockSizeLimit : max block size your kernel is designed to work with; 0 means “no limit.”
    //So 0, 0 means: “assume no dynamic shared memory” and “don’t cap the block size.” 

    // These variables are used to convert occupancy to warps
    int device;
    cudaDeviceProp prop;

    cudaGetDevice(&device);
    cudaGetDeviceProperties(&prop, device);

    int blockSizeGiven = 128;//N/2;
    int activeBlocksPerSMByCUDA;        // Occupancy in terms of active blocks
    cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        &activeBlocksPerSMByCUDA,
        basic_gpu_increment_kernel_MB,
        blockSizeGiven,
        dynamicSmemBytes);

    printOutOccupancy(prop, N, blockSizeGiven, activeBlocksPerSMByCUDA, "GridSize (activeBlocksPerSM) By cudaOccupancyMaxActiveBlocksPerMultiprocessor");

    //Option B (grid-stride): you may use minGridSize (or a few multiples of SM count) as your launch grid size,
    int minGridSizeByCUDA = 0;
    int blockSizeByCUDA = 0;
    cudaOccupancyMaxPotentialBlockSize(
        &minGridSizeByCUDA, 
        &blockSizeByCUDA, 
        basic_gpu_increment_kernel_MB, 
        dynamicSmemBytes, blockSizeLimit);

    printOutOccupancy(prop, N, blockSizeByCUDA, minGridSizeByCUDA, "Option B GridSize & BlockSize By cudaOccupancyMaxPotentialBlockSize (Need Stride)");

    //Option A(no stride) : use occupancy helper only to pick a decent blockSize, but compute grid from N.
    int gridSizeNeeded = (N + blockSizeByCUDA - 1) / blockSizeByCUDA;   // coverage
    printOutOccupancy(prop, N, blockSizeByCUDA, gridSizeNeeded, "Option A BlockSize By cudaOccupancyMaxPotentialBlockSize (gridSizeNeeded NO Need Stride)");
  
    //basic_gpu_increment_kernel << <activeBlocksPerSM , blockSize >> > ((int*)gpu_p);
    int numBlocksNeeded = (N + blockSizeGiven - 1) / blockSizeGiven;//activeBlocksPerSM = N / blockSize; - without stride
    printOutOccupancy(prop, N, blockSizeGiven, numBlocksNeeded, "GridSize By (N/blockSize)");
	std::cout << "numBlocksNeeded (without stride): " << numBlocksNeeded << "<> activeBlocksPerSM: " << activeBlocksPerSMByCUDA << std::endl << std::endl;

	int blockSize = blockSizeByCUDA;
	int gridSize = gridSizeNeeded;
    basic_gpu_increment_kernel_MB << <gridSize, blockSize >> > ((int*)gpu_p, N);

    //--------------------------- calculate occupancy ---------------------------

	cudaError_t result = cudaDeviceSynchronize();
	assert(result == cudaSuccess);

	gpu_memory_to_cpu_memory();
    print_cpu_numbers(false, true, 55200, 55300);

	gpu_free();
	cpu_free();

	//
    // int rc = getchar();

    return 0;
}

