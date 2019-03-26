#include <assert.h>
#include <getopt.h>
#include <stdlib.h>
#include <stdio.h>
#include <iostream>
#include <iomanip>
#include <string.h>
#include "types.h"
using namespace std;

void cmdLine(int argc, char *argv[], int& n, int& reps, int& ntx, int& nty, _DOUBLE_ & eps, int& do_host, int& prefer_l1, int& use_rand, int& use_bt, int& use_shm_double){

// Command line arguments, default settings

    n=8;
    reps = 10;

    // Threshold for comparison
    eps = 1.0e-6;

    // We don't do the computation on the host.
    do_host = 0;


    // We prefer Shared memory by default
    prefer_l1 = 0;

    // Use biadiagonal A & B for initial inputs
    use_rand = 0;

    // Use random initial A & B 
    use_bt = 0;

    // default to 4B interleaved shared memory
    use_shm_double = 0;

    // Ntx and Nty will be overriden by statically specified values
    // from the Make command line but are only useful when optimizing
    // for shared memory
#ifdef BLOCKDIM_X
    ntx = BLOCKDIM_X;
#else
    ntx = 8;
#endif

#ifdef BLOCKDIM_Y
    nty = BLOCKDIM_Y;
#else
    nty = 8;
#endif

 // Default value of the domain sizes
 static struct option long_options[] = {
        {"n", required_argument, 0, 'n'},
        {"r", required_argument, 0, 'r'},
        {"ntx", required_argument, 0, 'x'},
        {"nty", required_argument, 0, 'y'},
        {"do_host", no_argument, 0, 'h'},
        {"eps", required_argument, 0, 'e'},
        {"l1", no_argument, 0, 'l'},
        {"bt", no_argument, 0, 't'},
        {"rand", no_argument, 0, 'q'},
        {"shared_mem_double", no_argument, 0, 'D'},
        {"shared_mem_single", no_argument, 0, 'S'},
 };
    // Process command line arguments
 int ac;
 for(ac=1;ac<argc;ac++) {
    int c;
    while ((c=getopt_long(argc,argv,"n:r:x:y:he:lbRDS",long_options,NULL)) != -1){
        switch (c) {

	    // Size of the computational box
            case 'n':
                n = atoi(optarg);
                break;

            // Number of repititions
            case 'r':
                reps = atoi(optarg);
                break;

	    // X thread block geometry
            case 'x':
#ifdef BLOCKDIM_X
                cout << " *** The thread block size is statically compiled.\n     Ignoring the X thread geometry command-line setting\n";
#else
                ntx = atoi(optarg);
#endif
                break;

	    // Y thread block geometry
            case 'y':
#ifdef BLOCKDIM_Y
                cout << " *** The thread block size is statically compiled.\n      Ignoring the Y thread geometry command-line setting\n";
#else
                nty = atoi(optarg);
#endif
                break;

            // Run on the host (default: don't run on the host)
            case 'h':
                do_host = 1;
                break;

	    // comparison tolerance 
            case 'e':
#ifdef _DOUBLE
                sscanf(optarg,"%lf",&eps);
#else
                sscanf(optarg,"%f",&eps);
#endif
                break;


	    // Favor L1 cache (48 KB), else favor Shared memory
            case 'l':
                prefer_l1 = 1;
                break;

	    // Use bidiagonal matrices as inputs
            case 'b':
                use_bt = 1;
                break;

            // set shared memory config for this kernel to 8B
    	    case 'D':
  	       use_shm_double = 1;
	       break;
	    // Use random matrices as inputs
            case 'R':
                use_rand = 1;
                break;
    	    case 'S':
  	       use_shm_double = 0;
	       break;

	    // Error
            default:
                printf("Usage: mm [-n <domain size>] [-r <reps>] [-x <x thread geometry> [-y <y thread geometry] [-e <epsilon>] [-h {do_host}] [-l  <prefer l1>] [-b <use bt>] [-R <use rand>]\n");
                exit(-1);
            }
    }
 }
 if (use_rand && use_bt){
     cout << "You asked to use a random, bidiagonal matrix. This option is not supported.\n";
     exit(0);
 }
}
