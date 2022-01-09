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

#define SIZE 8192 * 12
#define THREADSIZE 1024
#define BLOCKSIZE ((SIZE - 1) / THREADSIZE + 1)
#define RADIX 10
#define MAXSM 12
#define FILE_TO_OPEN "OURLASTCODE_shared_measures.csv"

void radixSort(int *array, int size) {
    int significantDigit = 1;
    cudaEvent_t start, stop;
    int threadCount;
    int blockCount;

    int min, max;

    cudaStream_t stream[MAXSM];

    for (int i = 0; i <= MAXSM; i++)
        cudaStreamCreate(&stream[i]);
    threadCount = THREADSIZE;
    blockCount = size / 12;
    int max_digit = 4;
    // da calcolare bene
    int *outputArray;
    int *inputArray;

    int *myarraynew;
    cudaMallocHost((void **)&myarraynew, size * sizeof(int));

    CUDA_CHECK(cudaMalloc((void **)&inputArray, sizeof(int) * size));

    cudaMemcpyAsync(inputArray, array, 8192, cudaMemcpyHostToDevice, stream[1]);

    cudaThreadSynchronize();

    cudaMemcpyAsync(myarraynew, inputArray, 8192, cudaMemcpyDeviceToHost, stream[1]);
    cudaThreadSynchronize();
    printf("ora verifco se so fort %d size", new_size_first);
    for (int l = 0; l < 2048; l++) {
        if (array[l] != myarraynew[l]) {
            printf("questo %d e diverso da %d \n", array[l], myarraynew[l]);
        }
    }
}

int main() {
    printf("\n\nRunning Radix Sort Example in C!\n");
    printf("----------------------------------\n");

    int size = SIZE;
    int *array;
    cudaMallocHost((void **)&array, size * sizeof(int));
    int i;
    int max_digit = 9999;
    srand(time(NULL));

    for (i = 0; i < size; i++) {
        if (i % 2)
            array[i] = (rand() % max_digit);
        else
            array[i] = (rand() % max_digit);
    }

    // printf("\nUnsorted List: ");
    // printArray(array, size);

    radixSort(array, size);
    /*for (int i = 1; i < size; i++)
        if (array[i - 1] > array[i])
            printf("SE SCASSATT O PUNTATOR");*/

    // printf("\nSorted List:");
    // printArray(array, size);

    printf("\n");

    return 0;
}