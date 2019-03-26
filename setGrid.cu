#define TS 128 // Tile size of N
#define WPT 8
void setGrid(int n, dim3 &blockDim, dim3 &gridDim){ // Reg Block
    // cudaDeviceSetSharedMemConfig(cudaSharedMemBankSizeEightByte);
    // cudaDeviceSetCacheConfig(cudaFuncCachePreferEqual);

    blockDim.x = TS/WPT;
    blockDim.y = TS/WPT;

    gridDim.x = n / TS;
    gridDim.y = n / TS;

   if(n % TS != 0){
       gridDim.x++;
       gridDim.y++;
   }

}


// #define TS 32
// #define WPT 8
// void setGrid(int n, dim3 &blockDim, dim3 &gridDim){ // Tile
//     blockDim.x = TS;
//     blockDim.y = TS/WPT;
//
//     gridDim.x = n / TS;
//     gridDim.y = n / TS;
//     if(n % TS != 0){
//        gridDim.x++;
//        gridDim.y++;
//        }
// }


// void setGrid(int n, dim3 &blockDim, dim3 &gridDim)  // Naive
// {
//    // set your block dimensions and grid dimensions here
//    gridDim.x = n / blockDim.x;
//    gridDim.y = n / blockDim.y;
//    if(n % blockDim.x != 0)
//    	gridDim.x++;
//    if(n % blockDim.y != 0)
//     	gridDim.y++;
// }
