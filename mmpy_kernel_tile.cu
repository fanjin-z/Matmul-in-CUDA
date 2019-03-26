// Matrix multiply device code
#include <assert.h>
#include <math.h>
#include "utils.h"
#include "types.h"
using namespace std;


#define A(i, j) A[(i)*N + (j)]
#define B(i, j) B[(i)*N + (j)]
#define C(i, j) C[(i)*N + (j)]

#define TS 32
#define WPT 8 // work per thread
#define RTS (TS/WPT)


// srun -u -v --gres=gpu:1 ./mmpy -n 512 -x 1 -y 512 -r 3
// ./mmpy -n 512 -r 3
// make


__global__ void matMul(const int N, _DOUBLE_ *C, const _DOUBLE_ *A, const _DOUBLE_ *B) {
    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    const int bx = blockIdx.x;
    const int by = blockIdx.y;
    const int numTiles = N / TS;

    __shared__ _DOUBLE_ As[TS][TS], Bs[TS][TS+1];
    _DOUBLE_ Cs[WPT];

    #pragma unroll
    for (int w=0; w<WPT; w++)
        Cs[w] = 0.0f;

    for (int t=0; t<numTiles; t++){
        #pragma unroll
        const int AtileRow = bx * TS;
        const int AtileCol = t * TS;
        const int BtileRow = t * TS;
        const int BtileCol = by * TS;

        for (int w=0; w<WPT; w++){
            const int AworkRow = tx;
            const int AworkCol = ty + w * RTS;
            const int BworkRow = tx;
            const int BworkCol = ty + w * RTS;

            As[AworkCol][AworkRow] = __ldg(&A(AtileCol+AworkCol, AtileRow+AworkRow));
            Bs[BworkCol][BworkRow] = __ldg(&B(BtileCol+BworkCol, BtileRow+BworkRow));
        }
        __syncthreads();

        #pragma unroll
        for (int k=0; k<TS; k++){
            #pragma unroll
            for (int w=0; w<WPT; w++){
                Cs[w] += As[k][tx] * Bs[ty+w*RTS][k];
            }
        }
        __syncthreads();

        #pragma unroll
        for (int w=0; w<WPT; w++){
            const int CtileRow = bx * TS;
            const int CtileCol = by * TS;
            const int CworkRow = tx;
            const int CworkCol = ty + w * RTS;
            C(CtileCol+CworkCol, CtileRow+CworkRow) = Cs[w];
        }
    }

}
