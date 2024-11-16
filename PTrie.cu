#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <vector>
#include <string>
#include <cuda_runtime.h>
#include "PTrie.hpp"

#define ALPHABET_SIZE 4  // DNA components: A,T,C,G

// Declare the texture object globally
cudaTextureObject_t texObj2D;

__global__ void searchKernel(cudaTextureObject_t texObj, char *text, int *d_match_count, int textSize);
__device__ __host__ int charToIndex(char c);  // Convert DNA Bases to Indices

void printTable(int** table, int numStates);

// Constructor
PTrie::PTrie(const std::vector<std::string>& patterns) {
    int maxState = calculateMaxStates(patterns);
    // Create empty STT
    cur_STT = createSTT(maxState);
    cur_STT->maxStates = maxState;

    // Insert patterns into STT
    for (const std::string& pattern : patterns) {
        insertPattern(pattern.c_str());
    }
//printTable(cur_STT->table, cur_STT->numStates);
}

// Wrapper function to launch kernel and perform matching
int PTrie::search(const char *text, int textSize) {
    // Allocate device memory
    char *d_text;
    int *d_match_count;

    // Allocate host memory for results
    int match_count = 0; // Host variable for the match count

    // Allocate device memory
    cudaMalloc(&d_text, textSize * sizeof(char));
    cudaMalloc(&d_match_count, sizeof(int));
    cudaMemset(d_match_count, 0, sizeof(int));  // Initialize match count to 0 on the device
    // Copy data to device
    cudaMemcpy(d_text, text, textSize * sizeof(char), cudaMemcpyHostToDevice);

    // Step1: Allocate 2D CUDA array for STT to use texture object
    cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<int>();  // cudaChannelFormatDesc structure describes the format of a CUDA array
    cudaArray_t d_sttArray2D;
    cudaMallocArray(&d_sttArray2D, &channelDesc, ALPHABET_SIZE + 1, cur_STT->numStates);  // handle, element format (int), width, height

    // Step 2: Copy the STT from host to CUDA 2D array
    for (int i = 0; i < cur_STT->numStates; i++) {
        cudaMemcpy2DToArray(d_sttArray2D, 0, i, cur_STT->table[i],
                            (ALPHABET_SIZE + 1) * sizeof(int),
                            (ALPHABET_SIZE + 1) * sizeof(int), 1, cudaMemcpyHostToDevice);
    }

    // Step 3: Set up texture object
    cudaResourceDesc resDesc = {};  // Create a resource descriptor that defines the type of data the texture object will access
    resDesc.resType = cudaResourceTypeArray;  // Indicate that the resource is a CUDA array
    resDesc.res.array.array = d_sttArray2D;  // Assign the previously allocated 2D CUDA array as the resource

    cudaTextureDesc texDesc = {};  // Set up the texture descriptor, which defines how the texture is accessed
    texDesc.addressMode[0] = cudaAddressModeClamp;  // cudaAddressModeClamp: Prevent out-of-bounds access
    texDesc.addressMode[1] = cudaAddressModeClamp;
    texDesc.filterMode = cudaFilterModePoint;  // Specify point sampling (no interpolation)
    texDesc.readMode = cudaReadModeElementType;  // Esure that the texture object reads the raw data as integers
    texDesc.normalizedCoords = 0;  // The coordinates are not normalized

    // Create the texture object that allows accessing the data in d_sttArray2D with optimized caching and memory access
    cudaCreateTextureObject(&texObj2D, &resDesc, &texDesc, nullptr);

    // Step 4: Launch kernel with texture object
    int blockSize = 256;
    int gridSize = (textSize + blockSize - 1) / blockSize;
    searchKernel<<<gridSize, blockSize, blockSize * sizeof(int)>>>(texObj2D, d_text, d_match_count, textSize);  // blockSize * sizeof(int): dynamic shared memory allocation

    // Copy results back to the host
    cudaMemcpy(&match_count, d_match_count, sizeof(int), cudaMemcpyDeviceToHost);

    // Cleanup
    cudaDestroyTextureObject(texObj2D);
    cudaFreeArray(d_sttArray2D);
    cudaFree(d_text);
    cudaFree(d_match_count);

    return match_count;
}

// Destructor
PTrie::~PTrie() {
    freeSTT();
}

// Calculate upper bound of the number of states
int PTrie::calculateMaxStates(const std::vector<std::string>& patterns) {
    int cnt = 0;
    for (const std::string& pattern : patterns) {
        cnt += pattern.length();
    }
    return cnt;
}

// Construct the State Transition Table with the max number of states
STT* PTrie::createSTT(int maxStates) {
    STT *stt = (STT*)malloc(sizeof(STT));
    stt->numStates = 1; // Starting with initial state
    stt->table = (int **)malloc(maxStates * sizeof(int *));

    // Allocate and initialize table for each state
    for (int i = 0; i < maxStates; i++) {
        // Default value 0 indicates no transition
        // The extra last col is 1 if it is the end of a patter 
        stt->table[i] = (int *)calloc(ALPHABET_SIZE + 1, sizeof(int));
    }

    return stt;
}

// Insert a pattern into the State Transition Table
void PTrie::insertPattern(const char *pattern) {
    int state = 0;
    for (int i = 0; pattern[i] != '\0'; i++) {
        int c = charToIndex(pattern[i]);
        if (c == -1) continue; // Skip invalid characters

        if (cur_STT->table[state][c] == 0) {
            cur_STT->table[state][c] = cur_STT->numStates++; // Add new state
        }
        state = cur_STT->table[state][c];
    }
    cur_STT->table[state][ALPHABET_SIZE] = 1; // Mark the end state of the pattern
}

// Free all STT resources
void PTrie::freeSTT() {
    for (int i = 0; i < cur_STT->maxStates; i++) {
        free(cur_STT->table[i]);
    }
    free(cur_STT->table);
    free(cur_STT);
}

// Search function(kernel)
__global__ void searchKernel(cudaTextureObject_t texObj, char *text, int *d_match_count, int textSize) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int local_match_count = 0; // Per-thread match count

    // Declare shared memory for reduction
    extern __shared__ int shared_match_counts[];  // Dynamic shared memory allocation

    if (idx < textSize) {
        int state = 0;
        for (int i = idx; i < textSize; i++) {
            int index = charToIndex(text[i]);
            if (index == -1) break; // Stop if an invalid character is encountered

            // Fetch the next state using the 2D texture object
            state = tex2D<int>(texObj2D, index, state);
            if (state == 0) break;  // No valid transition

            // Check if the current state is a terminal (accepting) state
            if (tex2D<int>(texObj, ALPHABET_SIZE, state) == 1) {
                local_match_count++;
            }
        }
    }

    shared_match_counts[threadIdx.x] = local_match_count;
    __syncthreads();

    // Perform reduction to sum counts
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
                shared_match_counts[threadIdx.x] += shared_match_counts[threadIdx.x + stride];
        }
        __syncthreads();
    }

    // One thread per block updates the global match count
    if (threadIdx.x == 0) {
        atomicAdd(d_match_count, shared_match_counts[0]);
    }
}

__device__ __host__ int charToIndex(char c) {
    switch (c) {
        case 'A': return 0;
        case 'T': return 1;
        case 'C': return 2;
        case 'G': return 3;
        default: return -1; // Invalid character
    }
}

void printTable(int** table, int numStates) {
    for (int rol = 0; rol < numStates; rol++)
    {
        for (int col = 0; col <= ALPHABET_SIZE; col++)
        {
            printf(" %d ", table[rol][col]);
        }
        printf("\n");
	}
}