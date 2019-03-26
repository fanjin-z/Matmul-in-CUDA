// Some useful utilities
// system includes
#include <stdio.h>
#include <assert.h>
#include <sys/time.h>
#include <cuda.h>


// External function definitions

void checkCUDAError(const char *msg)
{
    cudaError_t err = cudaGetLastError();
    if( cudaSuccess != err) 
    {
        fprintf(stderr, "Cuda error: %s: %s.\n", msg, 
                                  cudaGetErrorString( err) );
        exit(EXIT_FAILURE);
    }                         
}

// Report device characteristics
int ReportDevice()
{
        int number_of_devices;
        cudaError_t  errCode = cudaGetDeviceCount(&number_of_devices);
	if ((errCode ==  cudaErrorNoDevice) || (errCode == cudaErrorInsufficientDriver)){
	   printf("\n *** There are no available devices.\n");
	   printf("     Either you are not attached to a compute node or\n");
	   printf("     are not running in an appropraite batch queue.\n");
	   printf("\n Exiting...\n\n");
	   exit(EXIT_FAILURE);
	}
	printf("# devices: %d\n",number_of_devices);
        if (number_of_devices > 1) {
	    printf("\n%d Devices\n",number_of_devices);
            int device_number;
            for (device_number = 0; device_number < number_of_devices;
        device_number++) {
                cudaDeviceProp deviceProp;
                assert(cudaSuccess == cudaGetDeviceProperties(&deviceProp, device_number));
                printf("Device # %d: capability %d.%d, %d cores\n",device_number,deviceProp.name, deviceProp.major, deviceProp.minor, deviceProp.multiProcessorCount);
            }
	    printf("\n");
        }
// get number of SMs on this GPU
        int devID;
        cudaGetDevice(&devID);
        cudaDeviceProp deviceProp;
        assert(cudaSuccess == cudaGetDeviceProperties(&deviceProp, devID));
        if (deviceProp.major == 9999 && deviceProp.minor == 9999) {
            printf("There is no device supporting CUDA.\n");
            cudaThreadExit();
        }
	// Output the characteristics
	// To report on others, see the following URL:
	// https://www.clear.rice.edu/comp422/resources/cuda/html/cuda-runtime-api/index.html#structcudaDeviceProp_1dee14230e417cb3059d697d6804da414

	printf("\nDevice is a %s, capability: %d.%d\n",  deviceProp.name, deviceProp.major, deviceProp.minor);

	printf("Clock speed: %f MHz\n",((double)deviceProp.clockRate)/1000);
        printf("# cores: %d\n",  deviceProp.multiProcessorCount);
	double gb = 1024*1024*1024;
        printf("\nGlobal memory: %fGB\n", ((double)deviceProp.totalGlobalMem)/gb);
	printf("Memory Clock Rate (MHz): %f\n", (double)deviceProp.memoryClockRate/1000);
	printf("Memory Bus Width (bits): %d\n", deviceProp.memoryBusWidth);
        printf("Peak Memory Bandwidth (GB/s): %f\n", 2.0*deviceProp.memoryClockRate*(deviceProp.memoryBusWidth/8)/1.0e6);
        printf("L2 Cache size: (KB): %f\n", (double)deviceProp.l2CacheSize/1024);
        if (deviceProp.ECCEnabled)
	    printf("ECC Enabled\n");
	else
	    printf("ECC NOT Enabled\n");

        if (deviceProp.asyncEngineCount == 1)
	    printf("Device can concurrently copy memory between host and device while executing a kernel\n");
        else if (deviceProp.asyncEngineCount == 2)
	    printf("Device can concurrently copy memory between host and device in both directions\n     and execute a kernel at the same time\n");
        else if (deviceProp.asyncEngineCount == 0){
	   printf("Device CANNOT copy memory between host and device while executing a kernel.\n");
	   printf("Device CANNOT copy memory between host and device in both directions at the same time.\n");
	}
        if (deviceProp.unifiedAddressing == 1)
	  printf("Device shares a unified address space with the host\n");
	else
	  printf("Device DOES NOT share a unified address space with the host\n");

	cudaSharedMemConfig sMemConfig;
	assert(cudaSuccess == cudaDeviceGetSharedMemConfig(&sMemConfig));
	printf("Device Shared Memory Config (override with -D or -S) = %s\n",
	       (sMemConfig == cudaSharedMemBankSizeDefault) ? "Shared Mem Bank Size Default" :
	       (sMemConfig == cudaSharedMemBankSizeFourByte) ? "Shared Mem Bank Size 4B" :
	       (sMemConfig == cudaSharedMemBankSizeEightByte) ? "Shared Mem Bank Size 4B" :
	       "Unknown value returned from cudaDeviceGetSharedMemConfig");

	printf("\n --------- \n");

        int driverVersion, runtimeVersion;
	assert(cudaSuccess == cudaDriverGetVersion(&driverVersion));
	assert(cudaSuccess == cudaRuntimeGetVersion(&runtimeVersion));
        printf("CUDA Driver version: %d, runtime version: %d\n\n", driverVersion, runtimeVersion);


	return(100*deviceProp.major + deviceProp.minor);
}
