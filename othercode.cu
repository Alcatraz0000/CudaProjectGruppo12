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

__global__ void histogramKernel(int *inArray, int *outArray, int *radixArray, int arrayLength, int significantDigit) {
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
        arrayElement = inArrayShared[thread];
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

__global__ void combineBucket(int *blockBucketArray, int *bucketArray) {
    __shared__ int bucketArrayShared[RADIX];

    int index = blockIdx.x * blockDim.x + threadIdx.x;

    int i;

    bucketArrayShared[index] = 0;

    for (i = index; i < RADIX * BLOCKSIZE; i = i + RADIX) {
        atomicAdd(&bucketArrayShared[index], blockBucketArray[i]);
    }

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

void cudaScanThrust(int *inarray, int arr_length, int *resultarray) {
    int length = arr_length;

    thrust::device_ptr<int> d_input = thrust::device_malloc<int>(length);
    thrust::device_ptr<int> d_output = thrust::device_malloc<int>(length);

    cudaMemcpy(d_input.get(), inarray, length * sizeof(int), cudaMemcpyHostToDevice);

    thrust::inclusive_scan(d_input, d_input + length, d_output);

    cudaThreadSynchronize();

    cudaMemcpy(resultarray, d_output.get(), length * sizeof(int), cudaMemcpyDeviceToHost);

    thrust::device_free(d_input);
    thrust::device_free(d_output);
}

void radixSort(int *array, int size) {
    int significantDigit = 1;

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

    cudaMalloc((void **)&inputArray, sizeof(int) * size);
    cudaMalloc((void **)&indexArray, sizeof(int) * size);
    cudaMalloc((void **)&radixArray, sizeof(int) * size);
    cudaMalloc((void **)&outputArray, sizeof(int) * size);
    cudaMalloc((void **)&semiSortArray, sizeof(int) * size);
    cudaMalloc((void **)&bucketArray, sizeof(int) * RADIX);
    cudaMalloc((void **)&blockBucketArray, sizeof(int) * RADIX * BLOCKSIZE);

    cudaMemcpy(inputArray, array, sizeof(int) * size, cudaMemcpyHostToDevice);

    int largestNum;
    thrust::device_ptr<int> d_in = thrust::device_pointer_cast(inputArray);
    thrust::device_ptr<int> d_out;
    d_out = thrust::max_element(d_in, d_in + size);
    largestNum = *d_out;
    printf("\tLargestNumThrust : %d\n", largestNum);

    while (largestNum / significantDigit > 0) {
        int bucket[RADIX] = {0};
        cudaMemcpy(bucketArray, bucket, sizeof(int) * RADIX, cudaMemcpyHostToDevice);

        histogramKernel<<<blockCount, threadCount>>>(inputArray, blockBucketArray, radixArray, size, significantDigit);
        cudaThreadSynchronize();

        combineBucket<<<1, RADIX>>>(blockBucketArray, bucketArray);
        cudaThreadSynchronize();

        cudaScanThrust(bucketArray, RADIX, bucketArray);
        cudaThreadSynchronize();

        indexArrayKernel<<<blockCount, threadCount>>>(radixArray, bucketArray, indexArray, size, significantDigit);
        cudaThreadSynchronize();

        semiSortKernel<<<blockCount, threadCount>>>(inputArray, semiSortArray, indexArray, size, significantDigit);
        cudaThreadSynchronize();

        copyKernel<<<blockCount, threadCount>>>(inputArray, semiSortArray, size);
        cudaThreadSynchronize();

        significantDigit *= RADIX;
    }

    cudaMemcpy(array, semiSortArray, sizeof(int) * size, cudaMemcpyDeviceToHost);

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
        array[i] = (rand() % max_digit);
    }

    printf("\nUnsorted List: ");
    printArray(array, size);

    radixSort(array, size);

    printf("\nSorted List:");
    printArray(array, size);

    printf("\n");

    return 0;
}