#include <cuda.h>
#include <cuda_runtime.h>
#include <driver_functions.h>
#include <stdio.h>
#include <stdlib.h>
#include <thrust/device_free.h>
#include <thrust/device_malloc.h>
#include <thrust/device_ptr.h>
#include <thrust/extrema.h>
#include <thrust/scan.h>
#include <time.h>

#define SIZE 8192
#define THREADSIZE 64
#define BLOCKSIZE ((SIZE - 1) / THREADSIZE + 1)
#define RADIX 10

__global__ void copyKernel(int *inArray, int *semiSortArray, int arrayLength) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;

    if (index < arrayLength) {
        inArray[index] = semiSortArray[index];
    }
}
__global__ void reduceMax(int *g_idata, int *g_odata) {
    __shared__ int sdata[SIZE / BLOCKSIZE];  // each thread loads one element from global to shared mem unsigned
    int tid = threadIdx.x;
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    sdata[tid] = g_idata[i];
    __syncthreads();  // do reduction in shared mem
    for (unsigned int s = 1; s < blockDim.x; s *= 2) {
        if (tid % (2 * s) == 0) {
            if (sdata[tid + s] > sdata[tid]) {
                sdata[tid] = sdata[tid + s];
            }
        }
        __syncthreads();
    }  // write result for this block to global mem
    if (tid == 0) {
        g_odata[blockIdx.x] = sdata[0];
    }
}

__global__ void reduceMax_Service(int *g_idata, int *max) {
    __shared__ int sdata[THREADSIZE];  // each thread loads one element from global to shared mem unsigned
    int tid = threadIdx.x;
    if (g_idata[tid] > g_idata[THREADSIZE + tid])
        sdata[tid] = g_idata[tid];
    else
        sdata[tid] = g_idata[THREADSIZE + tid];
    __syncthreads();  // do reduction in shared mem
    for (unsigned int s = 1; s < blockDim.x; s *= 2) {
        if (tid % (2 * s) == 0) {
            if (sdata[tid + s] > sdata[tid]) {
                sdata[tid] = sdata[tid + s];
            }
        }
        __syncthreads();
    }  // write result for this block to global mem
    if (tid == 0) {
        *max = sdata[0];
        printf("my max is %d", *max);
    }
}

__global__ void histogramKernel(int *inArray, int *outArray, int *radixArray, int arrayLength, int significantDigit, int minElement) {
    __shared__ int inArrayShared[THREADSIZE];
    __shared__ int outArrayShared[RADIX];
    __shared__ int radixArrayShared[THREADSIZE];

    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int thread = threadIdx.x;
    int blockIndex = blockIdx.x * RADIX;

    int radix;
    int arrayElement;
    int i;

    if (thread == 0) {
        for (i = 0; i < RADIX; i++) {
            outArrayShared[i] = 0;
        }
    }

    if (index < arrayLength) {
        inArrayShared[thread] = inArray[index];
    }

    __syncthreads();

    if (index < arrayLength) {
        arrayElement = inArrayShared[thread] - minElement;
        radix = ((arrayElement / significantDigit) % 10);
        radixArrayShared[thread] = radix;
        atomicAdd(&outArrayShared[radix], 1);
    }

    if (index < arrayLength) {
        radixArray[index] = radixArrayShared[thread];
    }
    __syncthreads();
    // forse possimao fare il casino che diventa supermegaultravelocissimo !!!!!!
    if (thread == 0) {
        for (i = 0; i < RADIX; i++) {
            outArray[blockIndex + i] = outArrayShared[i];
        }
    }
}

__global__ void combineBucket(int *blockBucketArray, int *bucketArray) {
    __shared__ int bucketArrayShared[RADIX];

    int index = blockIdx.x * blockDim.x + threadIdx.x;

    int i;

    bucketArrayShared[index] = 0;

    for (i = index; i < RADIX * BLOCKSIZE; i = i + RADIX) {
        atomicAdd(&bucketArrayShared[index], blockBucketArray[i]);
    }
    __syncthreads();
    if (threadIdx.x == 0) {
        for (i = 1; i < RADIX; i++)
            bucketArrayShared[i] += bucketArrayShared[i - 1];
    }
    __syncthreads();
    bucketArray[index] = bucketArrayShared[index];
}

__global__ void indexArrayKernel(int *radixArray, int *bucketArray, int *indexArray, int arrayLength, int significantDigit) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;

    int i;
    int radix;
    int pocket;

    if (index < RADIX) {
        for (i = 0; i < arrayLength; i++) {
            radix = radixArray[arrayLength - i - 1];
            if (radix == index) {
                pocket = --bucketArray[radix];
                indexArray[arrayLength - i - 1] = pocket;
            }
        }
    }
}

__global__ void semiSortKernel(int *inArray, int *outArray, int *indexArray, int arrayLength, int significantDigit) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;

    int arrayElement;
    int arrayIndex;

    if (index < arrayLength) {
        arrayElement = inArray[index];
        arrayIndex = indexArray[index];
        outArray[arrayIndex] = arrayElement;
    }
}

void printArray(int *array, int size) {
    int i;
    printf("[ ");
    for (i = 0; i < size; i++)
        printf("%d ", array[i]);
    printf("]\n");
}

int findLargestNum(int *array, int size) {
    int i;
    int largestNum = -1;
    for (i = 0; i < size; i++) {
        if (array[i] > largestNum)
            largestNum = array[i];
    }
    return largestNum;
}

void radixSort(int *array, int size) {
    int significantDigit = 1;
    cudaEvent_t start, stop;
    int threadCount;
    int blockCount;

    threadCount = THREADSIZE;
    blockCount = BLOCKSIZE;
    ;

    int *outputArray;
    int *inputArray;
    int *radixArray;
    int *bucketArray;
    int *indexArray;
    int *semiSortArray;
    int *blockBucketArray;
    int *g_odata;

    cudaMalloc((void **)&inputArray, sizeof(int) * size);
    cudaMalloc((void **)&indexArray, sizeof(int) * size);

    cudaMalloc((void **)&g_odata, sizeof(int) * BLOCKSIZE);

    cudaMalloc((void **)&radixArray, sizeof(int) * size);

    cudaMalloc((void **)&outputArray, sizeof(int) * size);

    cudaMalloc((void **)&semiSortArray, sizeof(int) * size);
    cudaMalloc((void **)&bucketArray, sizeof(int) * RADIX);
    cudaMalloc((void **)&blockBucketArray, sizeof(int) * RADIX * BLOCKSIZE);

    cudaMemcpy(inputArray, array, sizeof(int) * size, cudaMemcpyHostToDevice);

    int largestNum, smallestNum, max_digit;
    cudaThreadSynchronize();
    reduceMax<<<blockCount, threadCount>>>(inputArray, g_odata);
    reduceMax_Service<<<1, THREADSIZE>>>(g_odata, &largestNum);

    /* thrust::device_ptr<int> d_in = thrust::device_pointer_cast(inputArray);
     thrust::device_ptr<int> d_out;
     d_out = thrust::max_element(d_in, d_in + size);
     largestNum = *d_out;
     d_out = thrust::min_element(d_in, d_in + size);
     smallestNum = *d_out;*/
    printf("\tLargestNumThrust : %d\n", largestNum);
    printf("\tsmallestNumThrust : %d\n", smallestNum);
    max_digit = largestNum - smallestNum;
    cudaError_t mycudaerror;
    mycudaerror = cudaGetLastError();
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start, 0);

    while (max_digit / significantDigit > 0) {
        int bucket[RADIX] = {0};
        cudaMemcpy(bucketArray, bucket, sizeof(int) * RADIX, cudaMemcpyHostToDevice);
        // calcolo frequenza per ogni cifra, questo nel mio blocco.
        histogramKernel<<<blockCount, threadCount>>>(inputArray, blockBucketArray, radixArray, size, significantDigit, smallestNum);
        cudaThreadSynchronize();
        // calcolo la frequenza per ogni cifra, sommando quelle di tutti i block.
        // fondamentalmente sommo all'array delle frequenze il precedente, come facevamo nel vecchio algortimo. A[i-1] = A[i]
        combineBucket<<<1, RADIX>>>(blockBucketArray, bucketArray);
        cudaThreadSynchronize();
        // salva gli indici in cui memorizzare gli elementi ordinati --> fa la magia :D
        indexArrayKernel<<<blockCount, threadCount>>>(radixArray, bucketArray, indexArray, size, significantDigit);
        cudaThreadSynchronize();
        // salva gli elementi nella corretta posizione ordinati.
        semiSortKernel<<<blockCount, threadCount>>>(inputArray, semiSortArray, indexArray, size, significantDigit);
        cudaThreadSynchronize();
        // aggiorno inputArray con il semisortedarray
        copyKernel<<<blockCount, threadCount>>>(inputArray, semiSortArray, size);
        cudaThreadSynchronize();

        significantDigit *= RADIX;
    }
    mycudaerror = cudaGetLastError();
    if (mycudaerror != cudaSuccess) {
        fprintf(stderr, "%s\n", cudaGetErrorString(mycudaerror));
        exit(1);
    }
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    float transferTime;
    cudaEventElapsedTime(&transferTime, start, stop);
    printf("CUDA Time = %.5f ms dim=%d\n", transferTime, size);

    cudaMemcpy(array, inputArray, sizeof(int) * size, cudaMemcpyDeviceToHost);

    cudaFree(inputArray);
    cudaFree(indexArray);
    cudaFree(radixArray);
    cudaFree(bucketArray);
    cudaFree(blockBucketArray);
    cudaFree(outputArray);
    cudaFree(semiSortArray);
}

int main() {
    printf("\n\nRunning Radix Sort Example in C!\n");
    printf("----------------------------------\n");

    int size = SIZE;
    int array[size];
    int i;
    int max_digit = 9999;

    srand(time(NULL));

    for (i = 0; i < size; i++) {
        if (i % 2)
            array[i] = -(rand() % max_digit);
        else
            array[i] = (rand() % max_digit);
    }

    printf("\nUnsorted List: ");
    printArray(array, size);

    radixSort(array, size);
    for (int i = 1; i < size; i++)
        if (array[i - 1] > array[i])
            printf("SE SCASSATT O PUNTATOR");

    printf("\nSorted List:");
    printArray(array, size);

    printf("\n");

    return 0;
}
