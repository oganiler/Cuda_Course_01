#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <cstdio>
#include "device_info.h"
#include "driver_types.h" //to work on streams //cuda_runtime_api.h

const int N = 128 * 1024;
const size_t ARRAY_BYTES_INT = N * sizeof(int);//allocation size
const int pin_limit = 0; // 4 * 1024 * 1024; //16KB

void* cpu_p;
void* gpu_p;

__device__ unsigned int thrednum = 0;
__device__ unsigned int launchnum = 0;

void print_cpu_numbers(int* cpu_memory, int number_of_elements, bool all_elements, bool between, int firstN_element, int lastN_element)
{
    int* int_p = cpu_memory;

    if (all_elements)
    {
        for (int i = 0; i < number_of_elements; i++)
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

            for (int i = number_of_elements - lastN_element - 1; i < number_of_elements; i++)
            {
                std::cout << "Number " << i + 1 << " : " << int_p[i] << std::endl;
            }
        }

    }

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

    //unsigned int current_thrednum = atomicAdd(&thrednum, 1);

    //if (threadIdx.x == 0) {
    //    printf("blockIdx=%d  gridDim=%d  blockDim=%d N=%d gridDim.x * blockDim.x = %d\n",
    //        blockIdx.x, gridDim.x, blockDim.x, N, gridDim.x * blockDim.x);
    //}

    //if (blockIdx.x == 0 && threadIdx.x == 16) {
    //    printf("");
    //}

    if (idx < N) {
        //unsigned int current_launchnum = atomicAdd(&launchnum, 1);

        //launch
        //int current_data = g_data[idx];
        g_data[idx] *= 2;
        //int current_updated_data = g_data[idx];

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

    //if (threadIdx.x == 0) {
    //    printf("blockIdx=%d  gridDim=%d  blockDim=%d\n",
    //        blockIdx.x, gridDim.x, blockDim.x);
    //}

    //unsigned int current_thrednum = atomicAdd(&thrednum, 1) + 1;

    for (; idx < N; idx += stride) {
        //unsigned int current_launchnum = atomicAdd(&launchnum, 1) + 1;

        //launch
        //int current_data = g_data[idx];
        g_data[idx] *= 2;
        //int current_updated_data = g_data[idx];

        //printf("launchnum=%u data=%d --> %d |block=%d thread=%d idx=%d gridDim=%d blockDim=%d stride=%d thrednum=%u|\n",
        //    current_launchnum, current_data, current_updated_data, blockIdx.x, threadIdx.x, idx, gridDim.x, blockDim.x, stride, current_thrednum);
    }

    //printf("thrednum=%u block=%d thread=%d idx=%d gridDim=%d blockDim=%d\n",
    //    current_thrednum, blockIdx.x, threadIdx.x, idx, gridDim.x, blockDim.x);
}

//------------------ - Streamline multiple instance management------------------------
#define INSTANCE_COUNT 2
struct cpu_gpu_mem
{
    void* cpu_p;
    void* gpu_p;
    size_t nc;
	cudaStream_t stream;
};

void executeGPUMultipleInstances()
{
    const int instance_count = INSTANCE_COUNT;
	struct cpu_gpu_mem cgs[INSTANCE_COUNT];

	//allocate memory for multiple instances
    for(int i = 0; i < instance_count; i++)
    {
		struct cpu_gpu_mem* cg = &cgs[i];
        cg->nc = static_cast<size_t>(32 * 1024) * 1024;//32M elements

		//CPU allocation
        cg->cpu_p = malloc(cg->nc * sizeof(int));
		assert(cg->cpu_p != nullptr);

		//GPU allocation
        cudaError_t err = cudaMalloc(&cg->gpu_p, cg->nc * sizeof(int));
        assert(err == cudaSuccess);

		//set numbers on CPU
        int* int_p = (int*)cg->cpu_p;
        for (size_t j = 0; j < cg->nc; j++)
        {
            int_p[j] = static_cast<int>(j + 1);
		}

		//register CPU memory for GPU access
        cudaError_t err_register = cudaHostRegister(cg->cpu_p, cg->nc * sizeof(int), 0);//cudaHostRegisterMapped is suggested
		assert(err_register == cudaSuccess);
	}

    //execute GPU kernel for multiple instances
    for (int i = 0; i < instance_count; i++)
    {
        struct cpu_gpu_mem* cg = &cgs[i];

        //copy CPU memory to GPU memory
        cudaError_t err_copy = cudaMemcpy(cg->gpu_p, cg->cpu_p, cg->nc * sizeof(int), cudaMemcpyHostToDevice);
        assert(err_copy == cudaSuccess);

        //launch kernel
        int blockSize = 256;
        int gridSize = (static_cast<int>(cg->nc) + blockSize - 1) / blockSize;
        basic_gpu_increment_kernel_MB << <gridSize, blockSize >> > ((int*)cg->gpu_p, static_cast<int>(cg->nc));

        //copy GPU memory back to CPU memory
        cudaError_t err_copy_back = cudaMemcpy(cg->cpu_p, cg->gpu_p, cg->nc * sizeof(int), cudaMemcpyDeviceToHost);
        assert(err_copy_back == cudaSuccess);
	}

	//unregister and free memory for multiple instances
    for (int i = 0; i < instance_count; i++)
    {
        struct cpu_gpu_mem* cg = &cgs[i];

        //unregister CPU memory
        cudaError_t err_unregister = cudaHostUnregister(cg->cpu_p);
        assert(err_unregister == cudaSuccess);

        //free GPU memory
        cudaError_t err_free_gpu = cudaFree(cg->gpu_p);
        assert(err_free_gpu == cudaSuccess);
        cg->gpu_p = nullptr;

		//print CPU memory
        //std::cout << "Print For Instance " << i << " :" << std::endl;
        //int* int_p = (int*)cg->cpu_p;
        //print_cpu_numbers(int_p, cg->nc, false, false, 18, 12);
        //std::cout << std::endl << std::endl;

        //free CPU memory
        free(cg->cpu_p);
        cg->cpu_p = nullptr;
	}

	std::cout << "Multiple Instance Execution Completed." << std::endl;
}

void executeGPUMultipleInstancesStream()
{
    const int instance_count = INSTANCE_COUNT;
	const int launches_per_instance = 5;
    struct cpu_gpu_mem cgs[INSTANCE_COUNT];

    //allocate memory for multiple instances
    for (int i = 0; i < instance_count; i++)
    {
        struct cpu_gpu_mem* cg = &cgs[i];
        cg->nc = static_cast<size_t>(64 * 1024) * 1024;//32M elements

		//create stream
        cudaError_t stream_err = cudaStreamCreate(&cg->stream);
        assert(stream_err == cudaSuccess);

        //CPU allocation
        cg->cpu_p = malloc(cg->nc * sizeof(int));
        assert(cg->cpu_p != nullptr);

        //GPU allocation
        cudaError_t err = cudaMalloc(&cg->gpu_p, cg->nc * sizeof(int));
        assert(err == cudaSuccess);

        //set numbers on CPU
        int* int_p = (int*)cg->cpu_p;
        for (size_t j = 0; j < cg->nc; j++)
        {
            int_p[j] = static_cast<int>(j + 1);
        }

        //register CPU memory for GPU access
        cudaError_t err_register = cudaHostRegister(cg->cpu_p, cg->nc * sizeof(int), 0);//cudaHostRegisterMapped is suggested
        assert(err_register == cudaSuccess);
    }

    //execute GPU kernel for multiple instances
    for (int i = 0; i < instance_count; i++)
    {
        struct cpu_gpu_mem* cg = &cgs[i];

        //copy CPU memory to GPU memory
        cudaError_t err_copy = cudaMemcpyAsync(cg->gpu_p, cg->cpu_p, cg->nc * sizeof(int), cudaMemcpyHostToDevice);
        assert(err_copy == cudaSuccess);

        //launch kernel
        int blockSize = 256;
        int gridSize = (static_cast<int>(cg->nc) + blockSize - 1) / blockSize;

		//launch multiple times per instance
        for (int launch_idx = 1; launch_idx <= launches_per_instance; launch_idx++)
        {
            basic_gpu_increment_kernel_MB << <gridSize, blockSize, 0, cg->stream >> > ((int*)cg->gpu_p, static_cast<int>(cg->nc));
        }

        //copy GPU memory back to CPU memory
        cudaError_t err_copy_back = cudaMemcpyAsync(cg->cpu_p, cg->gpu_p, cg->nc * sizeof(int), cudaMemcpyDeviceToHost);
        assert(err_copy_back == cudaSuccess);
    }

    //unregister and free memory for multiple instances
    for (int i = 0; i < instance_count; i++)
    {
        struct cpu_gpu_mem* cg = &cgs[i];

        //unregister CPU memory
        cudaError_t err_unregister = cudaHostUnregister(cg->cpu_p);
        assert(err_unregister == cudaSuccess);

        //free GPU memory
        cudaError_t err_free_gpu = cudaFree(cg->gpu_p);
        assert(err_free_gpu == cudaSuccess);
        cg->gpu_p = nullptr;

		//free CPU memory -- will free after synchronization to ensure all streams are completed and printed out
        //free(cg->cpu_p);
        //cg->cpu_p = nullptr;

		//destroy the streams created
        cudaError_t stream_err = cudaStreamDestroy(cg->stream);
		assert(stream_err == cudaSuccess);
    }

    cudaDeviceSynchronize(); //ensure all streams are completed

    //print CPU memory
    for (int i = 0; i < instance_count; i++)
    {
        std::cout << "Print For Instance " << i << " :" << std::endl;
        struct cpu_gpu_mem* cg = &cgs[i];
        int* int_p = (int*)cg->cpu_p;
        print_cpu_numbers(int_p, cg->nc, false, false, 5, 5);
        std::cout << std::endl << std::endl;
    }

    //free CPU memory
    for (int i = 0; i < instance_count; i++)
    {
        struct cpu_gpu_mem* cg = &cgs[i];
        free(cg->cpu_p);
        cg->cpu_p = nullptr;
    }

    std::cout << "Multiple Instance Execution Completed." << std::endl;
}

//------------------- Streamline multiple instance management ------------------------

void cpu_alloc()
{
    cpu_p = malloc(ARRAY_BYTES_INT);
    assert(cpu_p != nullptr);
}

void cpu_for_gpu_alloc()
{
// this is just sample how cudaHostAlloc works
// we will use cudoHostRegister to avoid pinned areas to be blocked for CPU allocation

//#define cudaHostAllocDefault                0x00  /**< Default page-locked allocation flag */
//#define cudaHostAllocPortable               0x01  /**< Pinned memory accessible by all CUDA contexts */
//#define cudaHostAllocMapped                 0x02  /**< Map allocation into device space */
//#define cudaHostAllocWriteCombined          0x04  /**< Write-combined memory */
//
//#define cudaHostRegisterDefault             0x00  /**< Default host memory registration flag */
//#define cudaHostRegisterPortable            0x01  /**< Pinned memory accessible by all CUDA contexts */
//#define cudaHostRegisterMapped              0x02  /**< Map registered memory into device space */
//#define cudaHostRegisterIoMemory            0x04  /**< Memory-mapped I/O space */
//#define cudaHostRegisterReadOnly            0x08  /**< Memory-mapped read-only */
//
//#define cudaPeerAccessDefault               0x00  /**< Default peer addressing enable flag */
//
//#define cudaStreamDefault                   0x00  /**< Default stream flag */
//#define cudaStreamNonBlocking               0x01  /**< Stream does not synchronize with stream 0 (the NULL stream) */
    cudaError_t result = cudaHostAlloc(&cpu_p, ARRAY_BYTES_INT, 0);
    assert(result == cudaSuccess);
    //assert(cpu_p != nullptr);
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

void cpu_for_gpu_free()
{
    cudaError_t result = cudaFreeHost(cpu_p);
	assert(result == cudaSuccess);
    //cpu_p = nullptr;
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

void cpu_memory_to_gpu_memory_by_pinned()
{
//#define cudaHostAllocDefault                0x00  /**< Default page-locked allocation flag */
//#define cudaHostAllocPortable               0x01  /**< Pinned memory accessible by all CUDA contexts */
//#define cudaHostAllocMapped                 0x02  /**< Map allocation into device space */
//#define cudaHostAllocWriteCombined          0x04  /**< Write-combined memory */
//
//#define cudaHostRegisterDefault             0x00  /**< Default host memory registration flag */
//#define cudaHostRegisterPortable            0x01  /**< Pinned memory accessible by all CUDA contexts */
//#define cudaHostRegisterMapped              0x02  /**< Map registered memory into device space */
//#define cudaHostRegisterIoMemory            0x04  /**< Memory-mapped I/O space */
//#define cudaHostRegisterReadOnly            0x08  /**< Memory-mapped read-only */
//
//#define cudaPeerAccessDefault               0x00  /**< Default peer addressing enable flag */
//
//#define cudaStreamDefault                   0x00  /**< Default stream flag */
//#define cudaStreamNonBlocking               0x01  /**< Stream does not synchronize with stream 0 (the NULL stream) */

	//pin the CPU RAM areas so that GPU can access it faster
    cudaError_t err_register = cudaHostRegister(cpu_p, ARRAY_BYTES_INT, 0);//cudaHostRegisterMapped is suggested
    assert(err_register == cudaSuccess);

	//apply the memory copy to the pinned area
    cudaError_t err_copy = cudaMemcpy(gpu_p, cpu_p, ARRAY_BYTES_INT, cudaMemcpyHostToDevice);
    assert(err_copy == cudaSuccess);

	//unregister the pinned area after the copy so that CPU can use it normally
    cudaError_t err_unregister = cudaHostUnregister(cpu_p);
	assert(err_unregister == cudaSuccess);
}

void gpu_memory_to_cpu_memory()
{
    cudaError_t err = cudaMemcpy(cpu_p, gpu_p, ARRAY_BYTES_INT, cudaMemcpyDeviceToHost);
    assert(err == cudaSuccess);
}

//attention: we pin only the cpu memory, so send here only cpu pointer
void cpu_gpu_pin()
{
    bool pin = (ARRAY_BYTES_INT > pin_limit);
    cudaError_t result;

    if (pin)
    {
        //pin the CPU RAM areas so that GPU can access it faster
        result = cudaHostRegister(cpu_p, ARRAY_BYTES_INT, 0);//cudaHostRegisterMapped is suggested
        assert(result == cudaSuccess);
        std::cout << "cudaHostRegister is ON" << std::endl;
    }
    else
        std::cout << "cudaHostRegister is OFF" << std::endl;
}

//attention: we unpin only the cpu memory, so send here only cpu pointer
void cpu_gpu_unpin()
{
    bool pin = (ARRAY_BYTES_INT > pin_limit);
    cudaError_t result;
    if (pin)
    {
        //unregister the pinned area after the copy so that CPU can use it normally
        result = cudaHostUnregister(cpu_p);
        assert(result == cudaSuccess);
    }
}

//this function is called twice by gpu and cpu pin but we do not need to register and unregister twice, will be handled in the main to register/unregister once.
void cpu_gpu_mem_copy(enum cudaMemcpyKind copyKind, int* p)
{
    cudaError_t result;

    switch (copyKind)
    {
    case cudaMemcpyHostToDevice:
        //apply the memory copy to the pinned area
        result = cudaMemcpy(gpu_p, cpu_p, ARRAY_BYTES_INT, cudaMemcpyHostToDevice);
        assert(result == cudaSuccess);
        break;
    case cudaMemcpyDeviceToHost:
        result = cudaMemcpy(cpu_p, gpu_p, ARRAY_BYTES_INT, cudaMemcpyDeviceToHost);
        assert(result == cudaSuccess);
        break;
    default:
        assert(false); //unsupported copy kind
    }
}

//this function is called twice by gpu and cpu pin but we do not need to register and unregister twice, will be handled in the main to register/unregister once.
void cpu_gpu_register_mem_copy(enum cudaMemcpyKind copyKind, int *p)
{
    const int pin_limit = 0; // 4 * 1024 * 1024; //16KB
    bool pin = (ARRAY_BYTES_INT > pin_limit);
    cudaError_t result;

    if(pin)
    {
        //pin the CPU RAM areas so that GPU can access it faster
        result = cudaHostRegister(p, ARRAY_BYTES_INT, 0);//cudaHostRegisterMapped is suggested
        assert(result == cudaSuccess);
        std::cout << "cudaHostRegister is ON" << std::endl;
	}
    else
        std::cout << "cudaHostRegister is OFF" << std::endl;

    switch (copyKind)
    {
        case cudaMemcpyHostToDevice:
            //apply the memory copy to the pinned area
            result = cudaMemcpy(gpu_p, cpu_p, ARRAY_BYTES_INT, cudaMemcpyHostToDevice);
            assert(result == cudaSuccess);
            break;
        case cudaMemcpyDeviceToHost:
            result = cudaMemcpy(cpu_p, gpu_p, ARRAY_BYTES_INT, cudaMemcpyDeviceToHost);
            assert(result == cudaSuccess);
            break;
        default:
            assert(false); //unsupported copy kind
    }

    if(pin)
    {
        //unregister the pinned area after the copy so that CPU can use it normally
        result = cudaHostUnregister(p);
        assert(result == cudaSuccess);
	}
}

void cpu_gpu_host_to_device()
{
    cpu_gpu_mem_copy(cudaMemcpyHostToDevice, (int*)cpu_p);
}

void cpu_gpu_device_to_host()
{
    cpu_gpu_mem_copy(cudaMemcpyDeviceToHost, (int*)gpu_p);
}

void printOutOccupancy(cudaDeviceProp& prop, int number_of_elements, int blockSize, int gridSize, std::string title)
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
    std::cout << "Total Cover : % " << static_cast<float>((gridSize * blockSize)) / static_cast<float>(N) * 100 << std::endl;
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

void executeGPU()
{
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

    //kernel_MB launch - blockSize by cudaOccupancyMaxPotentialBlockSize - without stride
    //int blockSize = blockSizeByCUDA;
    //int gridSize = gridSizeNeeded;
    // 
    //kernel_stride - gridSize by cudaOccupancyMaxActiveBlocksPerMultiprocessor
    //int blockSize = blockSizeGiven;
    //int gridSize = activeBlocksPerSMByCUDA;

    //kernel_stride launch - blockSize & gridSize by cudaOccupancyMaxPotentialBlockSize - need stride
    //int blockSize = blockSizeByCUDA;
    //int gridSize = minGridSizeByCUDA;

    //kernel_MB launch - manually calculated gridSize from Block Size - no need stride
    int blockSize = blockSizeGiven;
    int gridSize = numBlocksNeeded;
    basic_gpu_increment_kernel_MB << <gridSize, blockSize >> > ((int*)gpu_p, N);

    //--------------------------- calculate occupancy ---------------------------

    // CudaMemcpy is asynchronous with respect to the host unless cudaDeviceSynchronize is called to block the host until the device has completed all preceding requested tasks.
    // so that when you call cudaMemcpy to copy data from device to host, the copy will not start until all preceding kernels have completed.
    // we can avoid calling cudaDeviceSynchronize here.
    //cudaError_t result = cudaDeviceSynchronize();
    //assert(result == cudaSuccess);
}

int runProfilingMain()
{
    PrintCudaDeviceInfo();

    // Optional: increase printf buffer if you print a lot
    cudaDeviceSetLimit(cudaLimitPrintfFifoSize, 8 * 1024 * 1024);  // 8MB :contentReference[oaicite:2]{index=2}

    //cpu_for_gpu_alloc();// avoid pinned areas to be blocked for CPU allocation, use cudaHostRegister instead
    cpu_alloc();
    gpu_alloc();

    cpu_set_numbers();

	cpu_gpu_pin(); // pin cpu memory before copy

    cpu_gpu_host_to_device();
    //cpu_memory_to_gpu_memory();

    executeGPU();

    //gpu_memory_to_cpu_memory();
    cpu_gpu_device_to_host();
	cpu_gpu_unpin(); // unpin cpu memory after copy

    print_cpu_numbers((int*)cpu_p, N, false, false, 18, 12);

    std::cout << "***" << std::endl;

    gpu_free();
	//cpu_for_gpu_free(); // avoid pinned areas to be blocked for CPU allocation, use cudaHostRegister instead. this free is for "cudaHostAlloc" function only.
    cpu_free();

    //
    // int rc = getchar();
    //std::cin.get();

    return 0;
}

