#include <stdio.h>
#include <cuda.h>

int main(int argc, char **argv)
{
    // Get number of devices
    int devCount;
    cudaGetDeviceCount(&devCount);
    printf("Device count: %d\n", devCount);

    // Query all device's properties
    printf("\n\n");
    cudaDeviceProp devProp;
    for (int i = 0; i < devCount; ++ i)
    {
        printf("Device %d:\n", i);
        cudaGetDeviceProperties(&devProp, i);

        printf("\tMaximum threads per block: %d\n", devProp.maxThreadsPerBlock);
        printf("\tMulti-processor count: %d\n", devProp.multiProcessorCount);
        printf("\tMaximum block dimension: (%d, %d, %d)\n"
            , devProp.maxThreadsDim[0]
            , devProp.maxThreadsDim[1]
            , devProp.maxThreadsDim[2]);
        printf("\tMaximum grid dimension: (%d, %d, %d)\n"
            , devProp.maxGridSize[0]
            , devProp.maxGridSize[1]
            , devProp.maxGridSize[2]);
        printf("\tMaximum registers per SM: %d\n", devProp.regsPerBlock);
        printf("\tMaximum threads per SM: %d\n", devProp.maxThreadsPerMultiProcessor);
        printf("\tMaximum blocks per SM: %d\n", devProp.maxBlocksPerMultiProcessor);
        printf("\tWarp size: %d\n", devProp.warpSize);
        printf("\n");
    }

    return 0;
}