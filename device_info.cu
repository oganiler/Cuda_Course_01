#include "device_info.h"
#include <cuda_runtime.h>
#include <iostream>
#include <iomanip>
#include <string>

#define CUDA_CHECK(call)                                                     \
  do {                                                                       \
    cudaError_t err = (call);                                                \
    if (err != cudaSuccess) {                                                \
      std::cerr << "CUDA error: " << cudaGetErrorString(err)                 \
                << " at " << __FILE__ << ":" << __LINE__ << "\n";           \
      std::exit(1);                                                          \
    }                                                                        \
  } while (0)

// Very small helper: map (major, minor) to an "FP32 cores per SM" estimate.
// NOTE: This is an architectural lookup (not reported by CUDA runtime).
static int fp32_cores_per_sm(int major, int minor) {
    // Turing (7.5): 64 FP32 cores/SM. :contentReference[oaicite:14]{index=14}
    if (major == 7 && minor == 5) return 64;

    // Ampere A100 (8.0): 64 FP32 cores/SM. :contentReference[oaicite:15]{index=15}
    if (major == 8 && minor == 0) return 64;

    // Ampere GA10x (8.6 / 8.7 consumer/workstation): 128 FP32 ops/clock per SM. :contentReference[oaicite:16]{index=16}
    if (major == 8 && (minor == 6 || minor == 7)) return 128;

    // Ada (8.9): 128 FP32 CUDA cores/SM. :contentReference[oaicite:17]{index=17}
    if (major == 8 && minor == 9) return 128;

    // Hopper (9.0): 128 FP32 CUDA cores/SM. :contentReference[oaicite:18]{index=18}
    if (major == 9) return 128;

    // Blackwell (12.0 etc.): 128 FP32 CUDA cores/SM (RTX Blackwell PDF). :contentReference[oaicite:19]{index=19}
    if (major >= 12) return 128;

    return -1; // unknown / not covered
}

void PrintCudaDeviceInfo()
{
    int n = 0;
    CUDA_CHECK(cudaGetDeviceCount(&n));

    std::cout << "CUDA devices: " << n << "\n\n";
    for (int dev = 0; dev < n; ++dev) {
        cudaDeviceProp p{};
        CUDA_CHECK(cudaGetDeviceProperties(&p, dev));

        std::cout << "=== Device " << dev << " ===\n";
        std::cout << "Name: " << p.name << "\n";
        std::cout << "Compute Capability: " << p.major << "." << p.minor << "\n";
        std::cout << "SM count (multiProcessorCount): " << p.multiProcessorCount << "\n";
        std::cout << "Warp size: " << p.warpSize << "\n";
        std::cout << "Max threads per block: " << p.maxThreadsPerBlock << "\n";
        std::cout << "Max threads dim: [" << p.maxThreadsDim[0] << ", "
            << p.maxThreadsDim[1] << ", " << p.maxThreadsDim[2] << "]\n";
        std::cout << "Max grid size: [" << p.maxGridSize[0] << ", "
            << p.maxGridSize[1] << ", " << p.maxGridSize[2] << "]\n";

        std::cout << "Global memory: "
            << std::fixed << std::setprecision(2)
            << (double)p.totalGlobalMem / (1024.0 * 1024.0 * 1024.0) << " GiB\n";

        std::cout << "Shared mem per block: " << p.sharedMemPerBlock / 1024 << " KiB\n";
        std::cout << "Regs per block: " << p.regsPerBlock << "\n";
        std::cout << "L2 cache size: " << p.l2CacheSize / 1024 << " KiB\n";

        // Clocks are reported in kHz by cudaDeviceProp.
        //std::cout << "SM clock: " << (p.clockRate / 1000) << " MHz\n";
        //std::cout << "Memory clock: " << (p.memoryClockRate / 1000) << " MHz\n";
        std::cout << "Memory bus width: " << p.memoryBusWidth << " bits\n";

        int cores_sm = fp32_cores_per_sm(p.major, p.minor);
        if (cores_sm > 0) {
            long long est_cuda_cores = 1LL * p.multiProcessorCount * cores_sm;
            std::cout << "Estimated 'CUDA cores' (SM * FP32/SM): " << est_cuda_cores
                << "  (heuristic)\n";
        }
        else {
            std::cout << "Estimated 'CUDA cores': unknown for this compute capability\n";
        }

        std::cout << "\n";
    }
}



