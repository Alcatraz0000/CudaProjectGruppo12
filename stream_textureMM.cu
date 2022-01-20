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

#define SIZE 8192 * 12 * 12 * 12
#define THREADSIZE 512
#define BLOCKSIZE ((SIZE - 1) / THREADSIZE + 1)
#define RADIX 10
#define MAXSM 12
#define BLOCKxSM (2048 / THREADSIZE)
#define FILE_TO_OPEN "OURLASTCODE_shared_measures.csv"

texture<int, 1> texture_semiSortArray;  // donotremove
__device__ float fetch_radixArrayElement(int value) {
    return tex1Dfetch(texture_semiSortArray, value);
}

__global__ void copyKernel(int *inArray, int offsette, int arrayLength) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;

    if (index < arrayLength) {
        inArray[index] = fetch_radixArrayElement(index + offsette);
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
    for (unsigned int s = blockDim.x / BLOCKxSM; s > 0; s >>= 1) {
        if (tid < s) {
            if (smaxdata[tid + s] > smaxdata[tid]) {
                smaxdata[tid] = smaxdata[tid + s];
            }
            if (smindata[tid + s] < smindata[tid]) {
                smindata[tid] = smindata[tid + s];
            }
        }
    }

    // write result for this block to global mem
    if (tid == 0) {
        g_maxdata[blockIdx.x] = smaxdata[0];
        g_mindata[blockIdx.x] = smindata[0];
    }
}
__global__ void reduceMaxMin_Service(int *g_maxdata, int *g_mindata, int *max, int *min) {
    __shared__ int smaxdata[(THREADSIZE)];  // each thread loads one element from global to shared mem unsigned
    __shared__ int smindata[(THREADSIZE)];
    int tid = threadIdx.x;
    smaxdata[tid] = g_maxdata[tid];
    smindata[tid] = g_mindata[tid];
    for (unsigned int s = 1; s < BLOCKSIZE / THREADSIZE; s++) {
        int index = THREADSIZE * s + tid;
        if (smaxdata[tid] < g_maxdata[index])
            smaxdata[tid] = g_maxdata[index];
        if (smindata[tid] > g_mindata[index])
            smindata[tid] = g_mindata[index];
    }
    __syncthreads();  // do reduction in shared mem
    for (unsigned int s = blockDim.x / BLOCKxSM; s > 0; s >>= 1) {
        if (tid < s) {
            if (smaxdata[tid + s] > smaxdata[tid]) {
                smaxdata[tid] = smaxdata[tid + s];
            }
            if (smindata[tid + s] < smindata[tid]) {
                smindata[tid] = smindata[tid + s];
            }
        }
    }
    // write result for this block to global mem
    if (tid == 0) {
        *max = smaxdata[0];
        *min = smindata[0];
    }
}
__global__ void resetBucket(int *bucket) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    bucket[index] = 0;
}

__global__ void histogramKernel(int *inArray, int *outArray, int *radixArray, int arrayLength, int significantDigit, int minElement) {
    __shared__ int inArrayShared[THREADSIZE];
    __shared__ int outArrayShared[RADIX];
    __shared__ int radixArrayShared[THREADSIZE];
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int thread = threadIdx.x;
    int blockIndex = blockIdx.x * RADIX;
    if (index > arrayLength) {
        return;
    }
    int radix;
    int arrayElement;
    int i;

    if (thread < RADIX) {
        outArrayShared[thread] = 0;
    }
    __syncthreads();
    if (index < arrayLength) {
        inArrayShared[thread] = inArray[index];

        arrayElement = inArrayShared[thread] - minElement;
        radix = ((arrayElement / significantDigit) % 10);
        radixArrayShared[thread] = radix;
        atomicAdd(&outArrayShared[radix], 1);

        radixArray[index] = radixArrayShared[thread];
    }
    __syncthreads();

    if (thread == 0) {
        for (i = 0; i < RADIX; i++) {
            outArray[blockIndex + i] += outArrayShared[i];
        }
    }
}

__global__ void combineBucket(int *blockBucketArray, int *bucketArray, int block_size) {
    __shared__ int bucketArrayShared[RADIX];

    int index = blockIdx.x * blockDim.x + threadIdx.x;

    int i;

    bucketArrayShared[index] = 0;

    for (i = index; i < RADIX * block_size; i = i + RADIX) {
        atomicAdd(&bucketArrayShared[index], blockBucketArray[i]);
    }
    __syncthreads();
    if (threadIdx.x == 0) {
        for (i = 1; i < RADIX; i++) {
            bucketArrayShared[i] += bucketArrayShared[i - 1];
        }
    }
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
        fprintf(fp, "N, BlockSize, GridSize, gflops, time_sec\n");
    }
    fprintf(fp, "%f, %d, %d, %f, %.5f\n", N, THREADSIZE, BLOCKSIZE, gflops, time / 1000);
    fclose(fp);
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
    threadCount = THREADSIZE;
    blockCount = BLOCKSIZE;

    int max_digit;

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

    int new_size_first = size / MAXSM + size % MAXSM;
    int new_size_second = size / MAXSM;
    int my_size, offset = 0;
    int new_block_size;

    CUDA_CHECK(cudaMalloc((void **)&inputArray, sizeof(int) * size));
    CUDA_CHECK(cudaMalloc((void **)&indexArray, sizeof(int) * size));

    CUDA_CHECK(cudaMalloc((void **)&g_maxdata, sizeof(int) * BLOCKSIZE));
    CUDA_CHECK(cudaMalloc((void **)&g_mindata, sizeof(int) * BLOCKSIZE));

    CUDA_CHECK(cudaMalloc((void **)&radixArray, sizeof(int) * size));

    CUDA_CHECK(cudaMalloc((void **)&outputArray, sizeof(int) * size));

    CUDA_CHECK(cudaMalloc((void **)&semiSortArray, sizeof(int) * size));
    CUDA_CHECK(cudaMalloc((void **)&bucketArray, sizeof(int) * RADIX));
    CUDA_CHECK(cudaMalloc((void **)&blockBucketArray, sizeof(int) * RADIX * BLOCKSIZE));

    cudaMalloc((void **)&largestNum, sizeof(int));
    cudaMalloc((void **)&smallestNum, sizeof(int));

    for (int j = 1; j <= MAXSM; j++) {
        if (j == 1) {
            cudaMemcpyAsync(inputArray, array, new_size_first * sizeof(int), cudaMemcpyHostToDevice, stream[j]);

        } else {
            cudaMemcpyAsync(inputArray + new_size_second * (j - 1) + size % MAXSM, array + new_size_second * (j - 1) + size % MAXSM, new_size_second * sizeof(int), cudaMemcpyHostToDevice, stream[j]);
        }
    }

    cudaError_t mycudaerror;
    mycudaerror = cudaGetLastError();
    cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<int>();
    mycudaerror = cudaBindTexture(0, texture_semiSortArray, semiSortArray, channelDesc);
    if (mycudaerror != cudaSuccess) printf("can not bind semiSortArray to texture \n");

    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start, 0);

    reduceMaxMin<<<blockCount, threadCount>>>(inputArray, g_maxdata, g_mindata);
    mycudaerror = cudaGetLastError();
    if (mycudaerror != cudaSuccess) {
        fprintf(stderr, "%s\n", cudaGetErrorString(mycudaerror));
        exit(1);
    }
    reduceMaxMin_Service<<<1, THREADSIZE>>>(g_maxdata, g_mindata, largestNum, smallestNum);
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

    max_digit = max - min;
    for (int j = 1; j <= MAXSM; j++) {
        if (j == 1) {
            my_size = new_size_first;
            offset = 0;
        } else {
            my_size = new_size_second;
            offset = new_size_second * (j - 1) + size % MAXSM;
        }
    }

    while (max_digit / significantDigit > 0) {
        resetBucket<<<BLOCKSIZE, RADIX>>>(blockBucketArray);
        resetBucket<<<1, RADIX>>>(bucketArray);
        for (int j = 1; j <= MAXSM; j++) {
            if (j == 1) {
                my_size = new_size_first;
                offset = 0;
            } else {
                my_size = new_size_second;
                offset = new_size_second * (j - 1) + size % MAXSM;
            }

            new_block_size = (my_size - 1) / THREADSIZE + 1;

            histogramKernel<<<new_block_size, THREADSIZE, 0, stream[j]>>>(inputArray + offset, blockBucketArray + (j - 1) * new_block_size * RADIX, radixArray + offset, my_size, significantDigit, min);

            mycudaerror = cudaGetLastError();
            if (mycudaerror != cudaSuccess) {
                fprintf(stderr, "%s\n", cudaGetErrorString(mycudaerror));
                exit(1);
            }

            // calcolo la frequenza per ogni cifra, sommando quelle di tutti i block.
            // fondamentalmente sommo all'array delle frequenze il precedente, come facevamo nel vecchio algortimo. A[i-1] = A[i]
            combineBucket<<<1, RADIX, 0, stream[j]>>>(blockBucketArray + (j - 1) * new_block_size * RADIX, bucketArray, new_block_size);

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
            if (j == 1) {
                my_size = new_size_first;
                offset = 0;
            } else {
                my_size = new_size_second;
                offset = new_size_second * (j - 1) + size % MAXSM;
            }
            new_block_size = (my_size - 1) / THREADSIZE + 1;
            // salva gli elementi nella corretta posizione ordinati.
            semiSortKernel<<<new_block_size, THREADSIZE, 0, stream[j]>>>(inputArray + offset, semiSortArray, indexArray + offset, my_size, significantDigit);
            mycudaerror = cudaGetLastError();
            if (mycudaerror != cudaSuccess) {
                fprintf(stderr, "%s\n", cudaGetErrorString(mycudaerror));
                exit(1);
            }
        }
        cudaThreadSynchronize();
        for (int j = 1; j <= MAXSM; j++) {
            if (j == 1) {
                my_size = new_size_first;
                offset = 0;
            } else {
                my_size = new_size_second;
                offset = new_size_second * (j - 1) + size % MAXSM;
            }
            new_block_size = (my_size - 1) / THREADSIZE + 1;
            copyKernel<<<new_block_size, THREADSIZE, 0, stream[j]>>>(inputArray + offset, offset, my_size);

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
    int *array;
    cudaMallocHost((void **)&array, size * sizeof(int));
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
        if (array[i - 1] > array[i]) {
            printf("SE SCASSATT O PUNTATOR");
            break;
        }

    // printf("\nSorted List:");
    // printArray(array, size);

    printf("\n");

    return 0;
}