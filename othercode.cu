#include <cuda.h>
#include <cuda_runtime.h>
#include <driver_functions.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <unistd.h>

#define SIZE 8192
#define THREADSIZE 64
#define BLOCKSIZE ((SIZE - 1) / THREADSIZE + 1)
#define RADIX 10
#define FILE_TO_OPEN "OURLASTCODE_shared_measures.csv"

__global__ void copyKernel(int *inArray, int *semiSortArray, int arrayLength) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;

    if (index < arrayLength) {
        inArray[index] = semiSortArray[index];
    }
}
__global__ void reduceMaxMin(int *g_idata, int *g_maxdata, int *g_mindata) {
    __shared__ int smaxdata[(SIZE / BLOCKSIZE)];  // each thread loads one element from global to shared mem unsigned
    __shared__ int smindata[(SIZE / BLOCKSIZE)];  // each thread loads one element from global to shared mem unsigned
    int tid = threadIdx.x;
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    smaxdata[tid] = g_idata[i];
    smindata[tid] = g_idata[i];
    __syncthreads();  // do reduction in shared mem
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            if (smaxdata[tid + s] > smaxdata[tid]) {
                smaxdata[tid] = smaxdata[tid + s];
            }
            if (smindata[tid + s] < smindata[tid]) {
                smindata[tid] = smindata[tid + s];
            }
        }
        __syncthreads();
    }  // write result for this block to global mem

    if (tid == 0) {
        g_maxdata[blockIdx.x] = smaxdata[0];
        g_mindata[blockIdx.x] = smindata[0];
    }
}

__global__ void reduceMaxMin_Service(int *g_maxdata, int *g_mindata, int *max, int *min) {
    __shared__ int smaxdata[(THREADSIZE)];  // each thread loads one element from global to shared mem unsigned
    __shared__ int smindata[(THREADSIZE)];
    int tid = threadIdx.x;
    if (g_maxdata[tid] > g_maxdata[THREADSIZE + tid])
        smaxdata[tid] = g_maxdata[tid];
    else
        smaxdata[tid] = g_maxdata[THREADSIZE + tid];
    if (g_mindata[tid] < g_mindata[THREADSIZE + tid])
        smindata[tid] = g_mindata[tid];
    else
        smindata[tid] = g_mindata[THREADSIZE + tid];
    __syncthreads();  // do reduction in shared mem
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            if (smaxdata[tid + s] > smaxdata[tid]) {
                smaxdata[tid] = smaxdata[tid + s];
            }
            if (smindata[tid + s] < smindata[tid]) {
                smindata[tid] = smindata[tid + s];
            }
        }
        __syncthreads();
    }  // write result for this block to global mem
    if (tid == 0) {
        *max = smaxdata[0];
        *min = smindata[0];
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
void make_csv(float gflops, float time, float N) {
    FILE *fp;
    if (access(FILE_TO_OPEN, F_OK) == 0) {
        fp = fopen(FILE_TO_OPEN, "a");

    } else {
        fp = fopen(FILE_TO_OPEN, "w");
        fprintf(fp, "N, gflops, time_sec\n");
    }
    fprintf(fp, "%f, %f, %.5f\n", N, gflops, time);
    fclose(fp);
}

void radixSort(int *array, int size) {
    int significantDigit = 1;
    cudaEvent_t start, stop;
    int threadCount;
    int blockCount;

    int min, max;

    threadCount = THREADSIZE;
    blockCount = BLOCKSIZE;

    int *outputArray;
    int *inputArray;
    int *radixArray;
    int *bucketArray;
    int *indexArray;
    int *semiSortArray;
    int *blockBucketArray;
    int *g_maxdata;
    int *g_mindata;
    int *largestNum;
    int *smallestNum;

    cudaMalloc((void **)&inputArray, sizeof(int) * size);
    cudaMalloc((void **)&indexArray, sizeof(int) * size);

    cudaMalloc((void **)&g_maxdata, sizeof(int) * BLOCKSIZE);
    cudaMalloc((void **)&g_mindata, sizeof(int) * BLOCKSIZE);

    cudaMalloc((void **)&radixArray, sizeof(int) * size);

    cudaMalloc((void **)&outputArray, sizeof(int) * size);

    cudaMalloc((void **)&semiSortArray, sizeof(int) * size);
    cudaMalloc((void **)&bucketArray, sizeof(int) * RADIX);
    cudaMalloc((void **)&blockBucketArray, sizeof(int) * RADIX * BLOCKSIZE);

    cudaMemcpy(inputArray, array, sizeof(int) * size, cudaMemcpyHostToDevice);

    int max_digit;
    cudaMalloc((void **)&largestNum, sizeof(int));
    cudaMalloc((void **)&smallestNum, sizeof(int));

    cudaError_t mycudaerror;
    mycudaerror = cudaGetLastError();
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start, 0);

    reduceMaxMin<<<blockCount, threadCount>>>(inputArray, g_maxdata, g_mindata);
    reduceMaxMin_Service<<<1, THREADSIZE>>>(g_maxdata, g_mindata, largestNum, smallestNum);

    cudaMemcpy(&max, largestNum, sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(&min, smallestNum, sizeof(int), cudaMemcpyDeviceToHost);

    max_digit = max - min;

    while (max_digit / significantDigit > 0) {
        int bucket[RADIX] = {0};
        cudaMemcpy(bucketArray, bucket, sizeof(int) * RADIX, cudaMemcpyHostToDevice);
        // calcolo frequenza per ogni cifra, questo nel mio blocco.
        histogramKernel<<<blockCount, threadCount>>>(inputArray, blockBucketArray, radixArray, size, significantDigit, min);
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
    make_csv(0, transferTime, size);
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

    // printf("\nUnsorted List: ");
    // printArray(array, size);

    radixSort(array, size);
    for (int i = 1; i < size; i++)
        if (array[i - 1] > array[i])
            printf("SE SCASSATT O PUNTATOR");

    // printf("\nSorted List:");
    // printArray(array, size);

    printf("\n");

    return 0;
}