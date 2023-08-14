#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/timeb.h>
#include "sha1.cuh"
#include "string.cuh"

const int HOST_NONCE_MAX_LEN = 6;
__device__ const int NONCE_MAX_LEN = 6;
__device__ const BYTE INIT_STRING[] = "adam";
__device__ const BYTE REQUIRED_SUFFIX[] = "00";

//Computes nonce based on thread id
__device__ BYTE* computeNonceBasedOnThreadId(int threadId)
{
    int wholePart = threadId, rest;
    BYTE c[NONCE_MAX_LEN] = {'\0'};
    int round = 0;

    while (wholePart > 0)
    {
        if(round == NONCE_MAX_LEN -1){
            return {'\0'};
        }

        if(threadId > 255 && round != 0){
            rest = ((wholePart-1) % 256);
            if(wholePart % 256 == 0){
                wholePart--;
            }
        }else{
            rest = (wholePart % 256);
        }

        wholePart = wholePart / 256;

        BYTE cu = rest;
        custrcat(c, &cu);
        round ++;
    }
    cudaRevstr(c);
    return c;
}

__global__ void getNonce(int *found, BYTE* foundNonce){
    if(*found == -1){
        int threadId = blockIdx.x *blockDim.x + threadIdx.x;

        BYTE* nonce = computeNonceBasedOnThreadId(threadId);

        const size_t newStringLength = custrlen(nonce) + custrlen(INIT_STRING);
        BYTE* combinedString = (BYTE *)malloc(sizeof(BYTE) * newStringLength);

        custrcpy(combinedString, INIT_STRING);
        custrcat(combinedString, nonce);

        BYTE buf[SHA1_BLOCK_SIZE];
        SHA1_CTX ctx;
        
        sha1_init(&ctx);
        sha1_update(&ctx, combinedString, custrlen(combinedString));
        sha1_final(&ctx, buf);

        //for some reason the length of buf becomes 22 after sha1_final, 
        //the last 2 digits being the nonce that was applied to the inital string
        //buf[custrlen(buf)-2] = '\0';

        if(buf[SHA1_BLOCK_SIZE-1] == REQUIRED_SUFFIX[1] && buf[SHA1_BLOCK_SIZE-2] == REQUIRED_SUFFIX[0]){
            printf("In thread: %d: %s -> %s\n", threadId, buf, combinedString);
            *found = 1;

            custrcpy(foundNonce, nonce);
        }

        free(combinedString);
    }
}

int main(int argc, char **argv)
{
    struct timeb start, end;

    ftime(&start);

    int found = -1;
    int* kernelFound;

    int gridSize = 128;
    int blockSize = 128;

    cudaOccupancyMaxPotentialBlockSize(&gridSize, &blockSize, getNonce);

    gridSize = 256;

    //printf("%d %d ", gridSize, blockSize);

    BYTE* nonce;
    BYTE* kernelNonce;

    nonce = (BYTE *)malloc(sizeof(BYTE) * HOST_NONCE_MAX_LEN);
    nonce[0] = '\0';

    cudaMalloc((void**)&kernelNonce, sizeof(BYTE) * HOST_NONCE_MAX_LEN);
    cudaMemcpy(kernelNonce, &nonce, sizeof(BYTE) * HOST_NONCE_MAX_LEN, cudaMemcpyHostToDevice);

    cudaMalloc((void**)&kernelFound, sizeof(int));
    cudaMemset(kernelFound, -1, sizeof(int));

    getNonce <<<gridSize,blockSize>>> (kernelFound, kernelNonce);
    cudaDeviceSynchronize();

    cudaMemcpy(nonce, kernelNonce, sizeof(BYTE) * HOST_NONCE_MAX_LEN, cudaMemcpyDeviceToHost);
    cudaMemcpy(&found, kernelFound, sizeof(int), cudaMemcpyDeviceToHost);

    if(found == 1){
        printf("\nFound nonce:\n");
        int i;
        for(i=0; i<custrlen(nonce);i++){
            printf("%d -> %c\n", nonce[i], nonce[i]);
        }
    }else{
        printf("\nNonce not found, increase thread number:\n");
    }

    ftime(&end);

    printf("Elapsed time: %.2lf\n", end.time - start.time + ((double)end.millitm - (double)start.millitm) / 1000.0);

    free(nonce);
    cudaFree(kernelFound);
    cudaFree(kernelNonce);
    return 0;
}

