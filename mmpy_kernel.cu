// Matrix multiply device code
#include <assert.h>
#include <math.h>
#include "utils.h"
#include "types.h"
using namespace std;

// srun -u -v --gres=gpu:1 ./mmpy -n 512 -x 1 -y 512 -r 3
// ./mmpy -n 512 -r 3
// make


#define A(i, j) A[(i)*N + (j)]
#define B(i, j) B[(i)*N + (j)]
#define C(i, j) C[(i)*N + (j)]

// Make sure to change TS and WPT in setGrid.cu accordingly
#define TS 128
#define TSK 16
#define WPT 8 // work per thread (work size)
#define RTS (TS/WPT)
#define LPTX (TS/RTS)
#define LPTY (TSK/RTS)


__global__ void matMul(const int N, _DOUBLE_ *C, const _DOUBLE_ *A, const _DOUBLE_ *B) {
    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    const int bx = blockIdx.x;
    const int by = blockIdx.y;
    const int numTiles = N / TSK;

    __shared__ _DOUBLE_ As[TSK][TS], Bs[TS][TSK+1];
    _DOUBLE_ Areg, Breg[WPT], Creg[WPT][WPT];

    #pragma unroll
    for (int w1=0; w1<WPT; w1++){
        #pragma unroll
        for (int w2=0; w2<WPT; w2++){
            Creg[w1][w2] = 0.0f;
        }
    }


    for (int t=0; t<numTiles; t++){
        const int AtileRow = bx * TS;
        const int AtileCol = t * TSK;
        const int BtileRow = t * TSK;
        const int BtileCol = by * TS;

        #pragma unroll
        for (int w1=0; w1<LPTX; w1++){
            #pragma unroll
            for (int w2=0; w2<LPTY; w2++){
                const int AworkRow = tx + w1 * RTS;
                const int AworkCol = ty + w2 * RTS;
                const int BworkRow = tx + w2 * RTS;
                const int BworkCol = ty + w1 * RTS;

                As[AworkCol][AworkRow] = __ldg(&A(AtileCol+AworkCol, AtileRow+AworkRow));
                Bs[BworkCol][BworkRow] = __ldg(&B(BtileCol+BworkCol, BtileRow+BworkRow));
                // As[AworkCol][AworkRow] = A(0, 0);
                // Bs[BworkCol][BworkRow] = B(0, 0);
            }
        }
        __syncthreads();

        for (int k=0; k<TSK; k++){

            #pragma unroll
            for (int w=0; w<WPT; w++){
                Breg[w] = Bs[ty+w*RTS][k];
            }

            #pragma unroll
            for (int w1=0; w1<WPT; w1++){
                Areg = As[k][tx+w1*RTS];
                #pragma unroll
                for (int w2=0; w2<WPT; w2++){
                    Creg[w1][w2] += Areg * Breg[w2];
                }
            }
        }
        __syncthreads();
    }

        #pragma unroll
        for (int w1=0; w1<WPT; w1++){
            int CtileRow = bx * TS;
            int CtileCol = by * TS;
            int CworkRow = tx + w1 * RTS;
            #pragma unroll
            for (int w2=0; w2<WPT; w2++){
                int CworkCol = ty + w2 * RTS;
                C(CtileCol+CworkCol, CtileRow+CworkRow) = Creg[w1][w2];
            }
        }
}
