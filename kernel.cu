#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <cstdio>
#include "profiling.h"

int main()
{
    //PrintCudaDeviceInfo();
    //runProfilingMain();
    //executeGPUMultipleInstances();

    executeGPUMultipleInstancesStream();

    return 0;
}

