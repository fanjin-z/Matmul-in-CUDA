/*
 * Simplest matrix multiplication in CUDA
 *
 * Scott B. Baden, University of California, San Diego
 * April 2010
 *
 * We compute C = A * B
 *
 * This code assumes that the  matrices are square though there
 * are hooks to facilitate  extending the code to non-square matrices
 *
 */

// system includes
#include <stdio.h>
#include <assert.h>

#include <iostream>

//  include the kernel
#include "mmpy_kernel.cu"

#include "types.h"
#include "utils.h"

// External function definitions
void genMatrix( _DOUBLE_ *a, unsigned int m, unsigned int n);
void genMatrix_bt( _DOUBLE_ *a, _DOUBLE_ *b, unsigned int n);
void genMatrix_rand( _DOUBLE_ *a, _DOUBLE_ *b, unsigned int n);
void verify( _DOUBLE_ *c, unsigned int m, unsigned int n, _DOUBLE_ eps, const char *mesg);
void verify_bt( _DOUBLE_ *c, unsigned int n, const char *mesg);
void verify( _DOUBLE_ *c_d, _DOUBLE_ *c_h,  unsigned int m, unsigned int n, _DOUBLE_ eps, const char *mesg);
void verify_bt( _DOUBLE_ *c_d, _DOUBLE_ *c_h,  unsigned int n, const char *mesg);
void verify_bt( _DOUBLE_ *c_d, _DOUBLE_ *c_h,  unsigned int m, unsigned int n,  const char *mesg);
void verify_rand( _DOUBLE_ *a, _DOUBLE_ *b, _DOUBLE_ *c, unsigned int n);

void printMatrix( _DOUBLE_ *a, unsigned int m, unsigned int n);
void cmdLine(int argc, char *argv[], int& n, int& reps, int& ntx, int& nty, _DOUBLE_ & eps, int& do_host, int& prefer_l1, int& use_rand, int& use_bt, int& use_shm_double);
void perfString(int n, int ntx, int nty, int reps, double t_h, double gflops_h, double t_d, double gflops_d, int do_host, int prefer_l1, int use_rand, int use_bt, int use_shm_double);
// extern "C"{
    double getTime();
    double gflops(int n, int niter, double time);
//}
void matMulHost(_DOUBLE_ *, const _DOUBLE_ *, const _DOUBLE_ *, unsigned int, unsigned int);
void setGrid(int n, dim3 &blockDim, dim3 &gridDim);

int
main(int argc, char** argv) {
    // To improve repeatabilty of measurements taken on the device,
    // we multiply the number of reps by this scale factor
    // Adjust as needed
    const int SCALE = 10;

// Read in the command line elements
    int n, reps, ntx, nty, do_host, prefer_l1, use_rand, use_bt, use_shm_double;
    _DOUBLE_ eps;

    cmdLine(argc, argv, n, reps, ntx, nty, eps, do_host, prefer_l1, use_rand, use_bt, use_shm_double);

   // The thread geometry must evenly divide N
   /*if ((n % ntx != 0) || (n % nty != 0) )
   {
        printf("Thread geometry: %d x %d\n",ntx, nty);
        printf("The length of the thread geometry axis ");
        printf("[ %d x %d]\n",ntx, nty);
        printf("  nust divide N [%d] evenly\n",n);
        exit(-1);
   }
   */

    // Total amount of storage for entries
    unsigned int n2 = n*n*sizeof(_DOUBLE_);

    // Report on Device Characteristics
    int capability = ReportDevice();
#ifdef _DOUBLE
    int major = capability/100;
    int minor = capability%100;
    if ((major == 1) && (minor < 3)){
        printf("   *** You are running on a capability %d.%d device\n",major, minor);
	printf("       which does not support double precision arithmetic.\n");
	printf("       Recompile with single precision.\n\n");
	exit(-1);
    }
#endif

    // setup execution configurations
    int _ntx, _nty;
#if (!defined(BLOCKDIM_X) && !defined(BLOCKDIM_Y))
    _ntx = ntx;
    _nty = nty;
#else
    _ntx = BLOCKDIM_X;
    _nty = BLOCKDIM_Y;
#endif

    dim3 threads(_ntx, _nty,1);
    int numblocksX = n/_ntx;
    int numblocksY = n/_nty;

    if( n % _ntx != 0  )
        numblocksX++;

    if( n % _nty != 0  )
        numblocksY++;
 
    dim3 grid(numblocksX, numblocksY, 1);

    setGrid(n, threads, grid);

    // print configurations
    printf("n: %d, tx: %d, ty: %d, gridX: %d, gridY: %d, reps: %d, epsilon: %g\n\n", n, threads.x, threads.y, grid.x, grid.y, reps, eps);

  
#ifndef _DOUBLE
    printf("Using Single precision arithmetic\n\n");
#else
    printf("Using Double precision arithmetic\n\n");
#endif

    if (use_bt)
        printf("Using bidiagonal inputs\n");

    if (use_rand)
        printf("Using random inputs\n");

    if (do_host)
        printf("Doing host computation for comparison\n\n");

     printf("\n");

    // allocate an initialize host memory for A and B matrices
    _DOUBLE_ *h_A = (_DOUBLE_ *) malloc(n2);
    assert(h_A);
    _DOUBLE_ *h_B = (_DOUBLE_ *) malloc(n2);
    assert(h_B);
    if (use_bt){
        genMatrix_bt(h_A, h_B, n);
    }
    else if (use_rand){
        genMatrix_rand(h_A, h_B, n);
    }
    else{
        genMatrix(h_A, n, n);
        genMatrix(h_B, n, n);
    }

    if (n <= 8){
        cout << "\nA:\n";
        printMatrix( h_A, n,n);
        cout << "\nB:\n";
        printMatrix( h_B, n,n);
    }

    _DOUBLE_  *hostC;
    double t_host=0.0, gflops_h=0.0;
    if (do_host){
        // compute matrix product on the host
        hostC = (_DOUBLE_ *) malloc(n2);
        t_host = -getTime();
        for (int r=0; r< reps; r++)
            matMulHost(hostC, h_A, h_B, n, n);
        t_host += getTime();
        gflops_h = gflops(n, reps, t_host );
        printf("Host computation time: %f sec. [%f gflops]\n",t_host,gflops_h);

        // Verify host result
        if (use_bt)
            verify_bt( hostC,n, "Host result");
        else if (use_rand)
            cout << "Verfication of host result not supported for random matrices\n";
        else
            verify( hostC,n,n, eps, "Host result");

        if (n <= 8){
            printf("\nC:\n");
            printMatrix( hostC, n,n);
        }
    }

    // allocate device memory
    _DOUBLE_ *d_A, *d_B, *d_C;
    cudaMalloc((void**) &d_A, n2);
    checkCUDAError("Error allocating device memory for matrix A");
    cudaMalloc((void**) &d_B, n2);
    checkCUDAError("Error allocating device memory for matrix B");
    cudaMalloc((void**) &d_C, n2);
    checkCUDAError("Error allocating device memory for matrix C");
    cudaMemset((void **) d_A,-99,n2);
    checkCUDAError("Error initializing device memory matrix A");
    cudaMemset((void **) d_B,-99,n2);
    checkCUDAError("Error initializing device memory matrix B");
    cudaMemset((void **) d_C,0,n2);
    checkCUDAError("Error clearing device memory matrix C");

    // copy host memory to device
    cudaMemcpy(d_A, h_A, n2, cudaMemcpyHostToDevice);
    checkCUDAError("Error copying matrix A to device");
    cudaMemcpy(d_B, h_B, n2, cudaMemcpyHostToDevice);
    checkCUDAError("Error copying matrix B to device");


    // allocate host memory for the result
    _DOUBLE_  *h_C = (_DOUBLE_ *) malloc(n2);
    assert(h_C);


// If we set the preference for L1 cache, rather than
// shared memory, we may run slightly faster on devices that have the capability
    cudaFuncCache Preference;
    if (prefer_l1){
        Preference = cudaFuncCachePreferL1;
    }
    else{
        Preference = cudaFuncCachePreferShared;
    } 
    cudaFuncSetCacheConfig(matMul,Preference);

    cudaSharedMemConfig  shmPreference;
    if (use_shm_double){
      shmPreference = cudaSharedMemBankSizeEightByte;
    }else{
      shmPreference = cudaSharedMemBankSizeFourByte;
    }
    cudaFuncSetSharedMemConfig( matMul, shmPreference);

    // Start the timer
#ifdef CUDA_TIMER
    cudaEvent_t start_event, stop_event;
    cudaEventCreate(&start_event) ;
    cudaEventCreate(&stop_event);
#endif

#ifdef CUDA_TIMER
    cudaEventRecord(start_event, 0);
    float t_device;
#else
    cudaThreadSynchronize();
    double t_device = -getTime();
#endif

    // execute the kernel
    for (int r=0; r< SCALE*reps; r++)
        matMul<<< grid, threads >>>(n, d_C, d_A, d_B);

#ifdef CUDA_TIMER
    cudaEventRecord(stop_event, 0);
    cudaEventSynchronize(stop_event);
    cudaEventElapsedTime(&t_device, start_event, stop_event);
    t_device /= 1000.0;

#else
    // block until the device has finished
    cudaThreadSynchronize();
    // Stop the timer
    t_device +=getTime();
#endif

    checkCUDAError("Error in matrixMul kernel");

    // copy result from device to host
    cudaMemcpy(h_C, d_C, n2, cudaMemcpyDeviceToHost);
    checkCUDAError("Unable to retrieve result from device");



    double gflops_d = gflops(n, SCALE*reps, t_device );
    printf("Device computation time: %f sec. [%f gflops]\n",t_device,gflops_d);
    perfString(n, ntx, nty, reps, t_host, gflops_h, t_device, gflops_d, do_host, prefer_l1, use_rand, use_bt, use_shm_double);

    if (n <= 8){
        printf("\nC (device):\n");
        printMatrix( h_C, n,n);
    }
    // Verify the device result
    if (use_bt)
        verify_bt( h_C,n,"Device result");
    else if (use_rand)
        verify_rand( h_A, h_B, h_C, n);
    else
        verify( h_C,n,n, eps,"Device result");

    // But not for random matrices
    if (do_host)
        // Compare host and device results
        if (use_bt)
            verify_bt( h_C, hostC, n,"Device vs. host");
        else if (!use_rand)
            verify( h_C, hostC, n, n, eps,"Device vs. host");

    // clean up memory
    free(h_A);
    free(h_B);
    free(h_C);
    if (do_host)
        free(hostC);

    assert(cudaSuccess ==cudaFree(d_A));
    assert(cudaSuccess ==cudaFree(d_B));
    assert(cudaSuccess ==cudaFree(d_C));

    cudaThreadExit();
}
