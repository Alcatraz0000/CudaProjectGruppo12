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

#define GRIDSIZE ((SIZE - 1) / BLOCKSIZE + 1)
#define RADIX 10
#define MAXSM 12
#define FILE_TO_OPEN "STEAMS_Global_measure.csv"

__global__ void copyKernel(int *inArray, int *semiSortArray, int arrayLength) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;

    if (index < arrayLength) {
        inArray[index] = semiSortArray[index];
    }
}

__global__ void reduceMaxMin(int *g_idata, int *g_maxdata, int *g_mindata, int *smaxdata, int *smindata) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    smaxdata[i] = g_idata[i];
    smindata[i] = g_idata[i];
    __syncthreads();  // do reduction in shared mem
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) {
            if (smaxdata[i + s] > smaxdata[i]) {
                smaxdata[i] = smaxdata[i + s];
            }
            if (smindata[i + s] < smindata[i]) {
                smindata[i] = smindata[i + s];
            }
        }
    }  // write result for this block to global mem

    if (threadIdx.x == 0) {
        g_maxdata[blockIdx.x] = smaxdata[blockIdx.x * blockDim.x];
        g_mindata[blockIdx.x] = smindata[blockIdx.x * blockDim.x];
    }
}

__global__ void reduceMaxMin_Service(int *g_maxdata, int *g_mindata, int *max, int *min, int *smaxdata, int *smindata) {
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
    }  // write result for this block to global mem
    if (tid == 0) {
        *max = smaxdata[0];
        *min = smindata[0];
    }
}

__global__ void histogramKernel(int *inArray, int *outArray, int *radixArray, int arrayLength, int significantDigit, int minElement, int *inArrayShared, int *outArrayShared, int *radixArrayShared) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int thread = threadIdx.x;
    int blockIndex = blockIdx.x * RADIX;

    int radix;
    int arrayElement;
    int i;
    if (index > arrayLength) {
        return;
    }
    if (thread == 0) {
        for (i = 0; i < RADIX; i++) {
            outArrayShared[blockIndex + i] = 0;
        }
    }

    if (index < arrayLength) {
        inArrayShared[index] = inArray[index];
    }

    __syncthreads();

    if (index < arrayLength) {
        arrayElement = inArrayShared[index] - minElement;
        radix = ((arrayElement / significantDigit) % 10);
        radixArrayShared[index] = radix;
        atomicAdd(&outArrayShared[blockIndex + radix], 1);
    }

    if (index < arrayLength) {
        radixArray[index] = radixArrayShared[index];
    }
    __syncthreads();
    // forse possimao fare il casino che diventa supermegaultravelocissimo !!!!!!
    if (thread == 0) {
        for (i = 0; i < RADIX; i++) {
            outArray[blockIndex + i] += outArrayShared[blockIndex + i];
        }
    }
}
__global__ void combineBucket(int *blockBucketArray, int *bucketArray, int new_block_size, int *bucketArrayShared) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;

    int i;

    bucketArrayShared[index] = 0;

    for (i = index; i < RADIX * new_block_size; i = i + RADIX) {
        atomicAdd(&bucketArrayShared[index], blockBucketArray[i]);
    }
    __syncthreads();
    if (threadIdx.x == 0) {
        for (i = 1; i < RADIX; i++) {
            bucketArrayShared[i] += bucketArrayShared[i - 1];
        }
    }
    __syncthreads();

    atomicAdd(&bucketArray[index], bucketArrayShared[index]);
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
    for (i = 0; i < 50; i++)
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

__global__ void resetBucket(int *bucket) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    bucket[index] = 0;
}

void radixSort(int *array, int size) {
    int significantDigit = 1;
    cudaEvent_t start, stop;
    int threadCount;
    int blockCount;
    int pocket;
    int radix;
    int min, max;

    cudaStream_t stream[MAXSM];

    for (int i = 0; i <= MAXSM; i++)
        cudaStreamCreate(&stream[i]);
    threadCount = BLOCKSIZE;
    blockCount = GRIDSIZE;

    int max_digit_value;

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

    int *inArrayShared;
    int *outArrayShared;
    int *radixArrayShared;
    int *smaxdata;
    int *smindata;

    int *bucketArrayShared;

    int new_size_first = size / MAXSM;
    int my_size, offset = 0;
    int new_block_size;
    CUDA_CHECK(cudaMalloc((void **)&bucketArrayShared, sizeof(int) * RADIX * MAXSM));

    CUDA_CHECK(cudaMalloc((void **)&inArrayShared, sizeof(int) * size));
    CUDA_CHECK(cudaMalloc((void **)&outArrayShared, sizeof(int) * RADIX * GRIDSIZE));
    CUDA_CHECK(cudaMalloc((void **)&radixArrayShared, sizeof(int) * size));

    CUDA_CHECK(cudaMalloc((void **)&inputArray, sizeof(int) * size));
    CUDA_CHECK(cudaMalloc((void **)&indexArray, sizeof(int) * size));

    CUDA_CHECK(cudaMalloc((void **)&smaxdata, sizeof(int) * size));
    CUDA_CHECK(cudaMalloc((void **)&smindata, sizeof(int) * size));

    CUDA_CHECK(cudaMalloc((void **)&g_maxdata, sizeof(int) * GRIDSIZE));
    CUDA_CHECK(cudaMalloc((void **)&g_mindata, sizeof(int) * GRIDSIZE));

    CUDA_CHECK(cudaMalloc((void **)&radixArray, sizeof(int) * size));

    CUDA_CHECK(cudaMalloc((void **)&outputArray, sizeof(int) * size));

    CUDA_CHECK(cudaMalloc((void **)&semiSortArray, sizeof(int) * size));
    CUDA_CHECK(cudaMalloc((void **)&bucketArray, sizeof(int) * RADIX));
    CUDA_CHECK(cudaMalloc((void **)&blockBucketArray, sizeof(int) * RADIX * GRIDSIZE));

    cudaMalloc((void **)&largestNum, sizeof(int));
    cudaMalloc((void **)&smallestNum, sizeof(int));

    for (int j = 1; j <= MAXSM; j++) {
        cudaMemcpyAsync(inputArray + new_size_first * (j - 1), array + new_size_first * (j - 1), new_size_first * sizeof(int), cudaMemcpyHostToDevice, stream[j]);
    }

    cudaError_t mycudaerror;
    mycudaerror = cudaGetLastError();
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start, 0);

    reduceMaxMin<<<blockCount, threadCount>>>(inputArray, g_maxdata, g_mindata, smaxdata, smindata);
    mycudaerror = cudaGetLastError();
    if (mycudaerror != cudaSuccess) {
        fprintf(stderr, "%s\n", cudaGetErrorString(mycudaerror));
        exit(1);
    }
    reduceMaxMin_Service<<<1, BLOCKSIZE>>>(g_maxdata, g_mindata, largestNum, smallestNum, smaxdata, smindata);
    mycudaerror = cudaGetLastError();
    if (mycudaerror != cudaSuccess) {
        fprintf(stderr, "%s\n", cudaGetErrorString(mycudaerror));
        exit(1);
    }

    cudaMemcpy(&max, largestNum, sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(&min, smallestNum, sizeof(int), cudaMemcpyDeviceToHost);

    int *bucket = (int *)malloc(RADIX * sizeof(int));
    int *CPUradixArray = (int *)malloc(size * sizeof(int));
    int *CPUindexArray = (int *)malloc(size * sizeof(int));

    max_digit_value = max - min;

    while (max_digit_value / significantDigit > 0) {
        resetBucket<<<GRIDSIZE, RADIX>>>(blockBucketArray);
        resetBucket<<<1, RADIX>>>(bucketArray);
        resetBucket<<<GRIDSIZE, BLOCKSIZE>>>(radixArrayShared);
        for (int j = 1; j <= MAXSM; j++) {
            my_size = new_size_first;
            offset = new_size_first * (j - 1);

            new_block_size = (my_size - 1) / BLOCKSIZE + 1;

            histogramKernel<<<new_block_size, BLOCKSIZE, 0, stream[j]>>>(inputArray + offset, blockBucketArray + (j - 1) * new_block_size * RADIX, radixArray + offset, my_size, significantDigit, min, inArrayShared + offset, outArrayShared, radixArrayShared + offset);

            mycudaerror = cudaGetLastError();
            if (mycudaerror != cudaSuccess) {
                fprintf(stderr, "%s\n", cudaGetErrorString(mycudaerror));
                exit(1);
            }

            // calcolo la frequenza per ogni cifra, sommando quelle di tutti i block.
            // fondamentalmente sommo all'array delle frequenze il precedente, come facevamo nel vecchio algortimo. A[i-1] = A[i]
            combineBucket<<<1, RADIX, 0, stream[j]>>>(blockBucketArray + (j - 1) * new_block_size * RADIX, bucketArray, new_block_size, bucketArrayShared + RADIX * (j - 1));

            mycudaerror = cudaGetLastError();
            if (mycudaerror != cudaSuccess) {
                fprintf(stderr, "%s\n", cudaGetErrorString(mycudaerror));
                exit(1);
            }
        }

        // reduce bucketArray

        // salva gli indici in cui memorizzare gli elementi ordinati --> fa la magia :D

        cudaMemcpy(CPUradixArray, radixArray, sizeof(int) * size, cudaMemcpyDeviceToHost);

        cudaMemcpy(bucket, bucketArray, sizeof(int) * RADIX, cudaMemcpyDeviceToHost);

        for (int c = 0; c < size; c++) {
            radix = CPUradixArray[size - c - 1];
            pocket = --bucket[radix];
            CPUindexArray[size - c - 1] = pocket;
        }
        cudaMemcpy(indexArray, CPUindexArray, sizeof(int) * size, cudaMemcpyHostToDevice);

        cudaThreadSynchronize();

        mycudaerror = cudaGetLastError();
        if (mycudaerror != cudaSuccess) {
            fprintf(stderr, "%s\n", cudaGetErrorString(mycudaerror));
            exit(1);
        }
        for (int j = 1; j <= MAXSM; j++) {
            my_size = new_size_first;
            offset = new_size_first * (j - 1);

            new_block_size = (my_size - 1) / BLOCKSIZE + 1;
            // salva gli elementi nella corretta posizione ordinati.
            semiSortKernel<<<new_block_size, BLOCKSIZE, 0, stream[j]>>>(inputArray + offset, semiSortArray, indexArray + offset, my_size, significantDigit);
            mycudaerror = cudaGetLastError();
            if (mycudaerror != cudaSuccess) {
                fprintf(stderr, "%s\n", cudaGetErrorString(mycudaerror));
                exit(1);
            }
        }
        cudaThreadSynchronize();
        for (int j = 1; j <= MAXSM; j++) {
            my_size = new_size_first;
            offset = new_size_first * (j - 1);

            new_block_size = (my_size - 1) / BLOCKSIZE + 1;
            copyKernel<<<new_block_size, BLOCKSIZE, 0, stream[j]>>>(inputArray + offset, semiSortArray + offset, my_size);

            mycudaerror = cudaGetLastError();
            if (mycudaerror != cudaSuccess) {
                fprintf(stderr, "%s\n", cudaGetErrorString(mycudaerror));
                exit(1);
            }
        }
        significantDigit *= RADIX;
    }
    cudaMemcpy(array, inputArray, sizeof(int) * size, cudaMemcpyDeviceToHost);
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
    for (int i = 1; i < size; i++)
        if (array[i - 1] > array[i])
            printf("SE SCASSATT O PUNTATOR");

    // printf("\nSorted List:");
    // printArray(array, size);

    printf("\n");

    return 0;
}