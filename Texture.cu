#include <cuda.h>
#include <cuda_runtime.h>
#include <driver_functions.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <unistd.h>

#define CUDA_CHECK(X)                                                     \
    {                                                                     \
        cudaError_t _m_cudaStat = X;                                      \
        if (cudaSuccess != _m_cudaStat) {                                 \
            fprintf(stderr, "\nCUDA_ERROR: %s in file %s line %d\n",      \
                    cudaGetErrorString(_m_cudaStat), __FILE__, __LINE__); \
            exit(1);                                                      \
        }                                                                 \
    }

#ifndef SIZE
#define SIZE 14155776
#endif

#ifndef BLOCKSIZE
#define BLOCKSIZE 1024
#endif

#ifndef MAX_DIGIT
#define MAX_DIGIT 9999
#endif

#ifndef GIPS
#define GIPS 0
#endif

#ifndef TEST
#define TEST 0
#endif

#define GRIDSIZE ((SIZE - 1) / BLOCKSIZE + 1)
#define RADIX 10
#define FILE_TO_OPEN "Texture_measures.csv"

texture<int, 1> texture_radixArray;
/**
 * This device function is used to retrive values from the array
 * saved in the texture memory
 * */
__device__ int fetch_radixArrayElement(int value) {
    return tex1Dfetch(texture_radixArray, value);
}
/**
 *  This kernel will be launched on GRIDSIZE * BLOCKSIZE threads, in order to copy the values of the semiSortArray in inArray
 *
 * **/
__global__ void copyKernel(int *inArray, int *semiSortArray, int arrayLength) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;

    if (index < arrayLength) {
        inArray[index] = semiSortArray[index];
    }
}
/**
 * This kernel will be launched on GRIDSIZE * BLOCKSIZE threads, so that each thread will calculate its local maximum and minimum value
 *
 * */
__global__ void reduceMaxMin(int *g_idata, int *g_maxdata, int *g_mindata) {
    __shared__ int smaxdata[(SIZE / GRIDSIZE)];
    __shared__ int smindata[(SIZE / GRIDSIZE)];
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
/**
 * This kernel will be launched on 1 * BLOCKSIZE threads, so that a global
 * maximum and minimum value shared between all blocks will be calcualted
 * */
__global__ void reduceMaxMin_Service(int *g_maxdata, int *g_mindata, int *max, int *min) {
    __shared__ int smaxdata[(BLOCKSIZE)];
    __shared__ int smindata[(BLOCKSIZE)];
    int tid = threadIdx.x;
    smaxdata[tid] = g_maxdata[tid];
    smindata[tid] = g_mindata[tid];
    for (unsigned int s = 1; s < GRIDSIZE / BLOCKSIZE; s++) {
        int index = BLOCKSIZE * s + tid;
        if (smaxdata[tid] < g_maxdata[index])
            smaxdata[tid] = g_maxdata[index];
        if (smindata[tid] > g_mindata[index])
            smindata[tid] = g_mindata[index];
    }
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
/**
 * This kernel will be launched on GRIDSIZE * BLOCKSIZE threads, so that each thread for
 * the specific significant digit of the assigned value will increase the frequencies of the digit
 * */
__global__ void histogramKernel(int *inArray, int *outArray, int *radixArray, int arrayLength, int significantDigit, int minElement) {
    __shared__ int inArrayShared[BLOCKSIZE];
    __shared__ int outArrayShared[RADIX];
    __shared__ int radixArrayShared[BLOCKSIZE];

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
    if (thread == 0) {
        for (i = 0; i < RADIX; i++) {
            outArray[blockIndex + i] = outArrayShared[i];
        }
    }
}
/**
 * This kernel will be launched on 1 * RADIX threads, so that the array containing the frequencies
 * for each block is addictioned to that of the other blocks.
 * Then the value in each position of the resulting array is addictioned with the value in the previus one position.
 * */
__global__ void combineBucket(int *blockBucketArray, int *bucketArray) {
    __shared__ int bucketArrayShared[RADIX];

    int index = blockIdx.x * blockDim.x + threadIdx.x;

    int i;

    bucketArrayShared[index] = 0;

    for (i = index; i < RADIX * GRIDSIZE; i = i + RADIX) {
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
/**
 * This kernel will be launched on 1 * RADIX threads, so that each thread
 * takes care of one digit between 0-9 and estabilishes the specific position
 * at which collocate the number
 * */
__global__ void indexArrayKernel(int *bucketArray, int *indexArray, int arrayLength, int significantDigit) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;

    int i;
    int radix;
    int pocket;

    if (index < RADIX) {
        for (i = 0; i < arrayLength; i++) {
            radix = fetch_radixArrayElement(arrayLength - i - 1);
            if (radix == index) {
                pocket = --bucketArray[radix];
                indexArray[arrayLength - i - 1] = pocket;
            }
        }
    }
}
/**
 * This kernel will be launched on GRIDSIZE * BLOCKSIZE, so that a sorting
 * for the specific significantDigit between the numbers is made
 * */
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

/**
 * Print all the array
 * */
void printArray(int *array, int size) {
    int i;
    printf("[ ");
    for (i = 0; i < size; i++)
        printf("%d ", array[i]);
    printf("]\n");
}
/**
 * This functions makes the csv file with all the information needed
 * */
void make_csv(float time, float N) {
    FILE *fp;
    if (access(FILE_TO_OPEN, F_OK) == 0) {
        fp = fopen(FILE_TO_OPEN, "a");

    } else {
        fp = fopen(FILE_TO_OPEN, "w");
        fprintf(fp, "N, BLOCKSIZE, GRIDSIZE, MAX_DIGIT, GIPS, TIME_SEC\n");
    }
    fprintf(fp, "%f, %d, %d, %d, %f, %.5f\n", N, BLOCKSIZE, GRIDSIZE, MAX_DIGIT, GIPS / (time / 1000), time / 1000);
    fclose(fp);
}

/**
 * This functions test if the array is correctly sorted.
 * */
void TESTArray(int *array, int size) {
    for (int i = 1; i < size; i++)
        if (array[i - 1] > array[i]) {
            printf("\nERRORE NELL'ORDINAMENTO!\n");
            break;
        }
    printf("Ordinamento Corretto");
}
/**
 * This functions allocates all the resurces and launches all the kernel necessary
 * to sort the array
 * */
void radixSort(int *array, int size) {
    int significantDigit = 1;
    cudaEvent_t start, stop;
    int threadCount;
    int blockCount;

    int min, max;

    threadCount = BLOCKSIZE;
    blockCount = GRIDSIZE;

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
    CUDA_CHECK(cudaMalloc((void **)&inputArray, sizeof(int) * size));
    CUDA_CHECK(cudaMalloc((void **)&indexArray, sizeof(int) * size));

    CUDA_CHECK(cudaMalloc((void **)&g_maxdata, sizeof(int) * GRIDSIZE));
    CUDA_CHECK(cudaMalloc((void **)&g_mindata, sizeof(int) * GRIDSIZE));

    CUDA_CHECK(cudaMalloc((void **)&radixArray, sizeof(int) * size));

    CUDA_CHECK(cudaMalloc((void **)&outputArray, sizeof(int) * size));

    CUDA_CHECK(cudaMalloc((void **)&semiSortArray, sizeof(int) * size));
    CUDA_CHECK(cudaMalloc((void **)&bucketArray, sizeof(int) * RADIX));
    CUDA_CHECK(cudaMalloc((void **)&blockBucketArray, sizeof(int) * RADIX * GRIDSIZE));

    cudaMemcpy(inputArray, array, sizeof(int) * size, cudaMemcpyHostToDevice);

    // Create the channel between global and texture memory and bind them.
    cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<int>();
    cudaError_t errt = cudaBindTexture(0, texture_radixArray, radixArray, channelDesc);
    if (errt != cudaSuccess) printf("can not bind radixArray to texture \n");

    int max_digit_value;
    cudaMalloc((void **)&largestNum, sizeof(int));
    cudaMalloc((void **)&smallestNum, sizeof(int));

    cudaError_t mycudaerror;
    mycudaerror = cudaGetLastError();
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start, 0);

    reduceMaxMin<<<blockCount, threadCount>>>(inputArray, g_maxdata, g_mindata);
    mycudaerror = cudaGetLastError();
    if (mycudaerror != cudaSuccess) {
        fprintf(stderr, "%s\n", cudaGetErrorString(mycudaerror));
        exit(1);
    }
    reduceMaxMin_Service<<<1, BLOCKSIZE>>>(g_maxdata, g_mindata, largestNum, smallestNum);
    mycudaerror = cudaGetLastError();
    if (mycudaerror != cudaSuccess) {
        fprintf(stderr, "%s\n", cudaGetErrorString(mycudaerror));
        exit(1);
    }

    cudaMemcpy(&max, largestNum, sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(&min, smallestNum, sizeof(int), cudaMemcpyDeviceToHost);

    // We add minimum number of the array with the maximum one in order to support also the sorting of the negative numbers
    max_digit_value = max - min;
    // We iterate on the number of digit contained in the max_digit_value
    while (max_digit_value / significantDigit > 0) {
        int bucket[RADIX] = {0};
        cudaMemcpy(bucketArray, bucket, sizeof(int) * RADIX, cudaMemcpyHostToDevice);

        histogramKernel<<<blockCount, threadCount>>>(inputArray, blockBucketArray, radixArray, size, significantDigit, min);
        cudaThreadSynchronize();
        mycudaerror = cudaGetLastError();
        if (mycudaerror != cudaSuccess) {
            fprintf(stderr, "%s\n", cudaGetErrorString(mycudaerror));
            exit(1);
        }

        combineBucket<<<1, RADIX>>>(blockBucketArray, bucketArray);
        cudaThreadSynchronize();
        mycudaerror = cudaGetLastError();
        if (mycudaerror != cudaSuccess) {
            fprintf(stderr, "%s\n", cudaGetErrorString(mycudaerror));
            exit(1);
        }

        indexArrayKernel<<<1, RADIX>>>(bucketArray, indexArray, size, significantDigit);
        cudaThreadSynchronize();
        mycudaerror = cudaGetLastError();
        if (mycudaerror != cudaSuccess) {
            fprintf(stderr, "%s\n", cudaGetErrorString(mycudaerror));
            exit(1);
        }

        semiSortKernel<<<blockCount, threadCount>>>(inputArray, semiSortArray, indexArray, size, significantDigit);
        cudaThreadSynchronize();
        mycudaerror = cudaGetLastError();
        if (mycudaerror != cudaSuccess) {
            fprintf(stderr, "%s\n", cudaGetErrorString(mycudaerror));
            exit(1);
        }

        copyKernel<<<blockCount, threadCount>>>(inputArray, semiSortArray, size);
        cudaThreadSynchronize();
        mycudaerror = cudaGetLastError();
        if (mycudaerror != cudaSuccess) {
            fprintf(stderr, "%s\n", cudaGetErrorString(mycudaerror));
            exit(1);
        }

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
    printf("CUDA Time = %.5f ms GIPS = %.5f MAX_DIGIT = %d BLOCKSIZE = %d dim=%d\n", transferTime, GIPS, MAX_DIGIT, BLOCKSIZE, size);
    make_csv(transferTime, size);
    cudaMemcpy(array, inputArray, sizeof(int) * size, cudaMemcpyDeviceToHost);

    cudaFree(inputArray);
    cudaFree(indexArray);
    cudaFree(g_maxdata);
    cudaFree(g_mindata);
    cudaFree(radixArray);
    cudaFree(bucketArray);
    cudaFree(blockBucketArray);
    cudaFree(outputArray);
    cudaFree(semiSortArray);

    cudaUnbindTexture(texture_radixArray);
}

int main() {
    printf("\n\nRunning Radix Sort Example in C!\n");
    printf("----------------------------------\n");

    int size = SIZE;
    int *array = (int *)malloc(size * sizeof(int));
    int i;
    srand(time(NULL));

    for (i = 0; i < size; i++) {
        if (i % 2)
            array[i] = -(rand() % MAX_DIGIT);
        else
            array[i] = (rand() % MAX_DIGIT);
    }

    // printf("\nUnsorted List: ");
    // printArray(array, size);

    radixSort(array, size);

    if (TEST) {
        TESTArray(array, size);
    }

    // printf("\nSorted List:");
    // printArray(array, size);

    printf("\n");

    return 0;
}