#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <vector>
#include <string>
#include <cuda_runtime.h>
#include "PTrie.hpp"

#define ALPHABET_SIZE 4  // DNA components: A,T,C,G

__global__ void searchKernel(STT *d_stt, char *text, int *d_match_count, int textSize);

void printTable(int** table, int numStates);

    // Constructor
    PTrie::PTrie(const std::vector<std::string>& patterns) {
        int maxState = calculateMaxStates(patterns);
        // Create empty STT
        cur_STT = createSTT(maxState);
        cur_STT->maxStates = maxState;

        // Insert patterns into STT
        for (const std::string& pattern : patterns) {
            insertPattern( pattern.c_str());
        }
	//printTable(cur_STT->table, cur_STT->numStates);
    }

    // Wrapper function to launch kernel and perform matching
    int PTrie::search(const char *text, int textSize) {
        // Allocate device memory
        STT *d_stt;
        char *d_text;
        int *d_match_count;

        // Allocate host memory for results
        int match_count = 0; // Host variable for the match count

        // Allocate device memory
        cudaMalloc(&d_stt, sizeof(STT));
        cudaMalloc(&d_text, textSize * sizeof(char));
        cudaMalloc(&d_match_count, sizeof(int));

        // Initialize match count to 0 on the device
        cudaMemset(d_match_count, 0, sizeof(int));
        // Copy data to device
        cudaMemcpy(d_text, text, textSize * sizeof(char), cudaMemcpyHostToDevice);

        // Deep copy STT from host to device
        // Allocate device memory for the table (array of pointers)
        int **d_table;  // a pointer to a pointer of type int
        cudaMalloc(&d_table, cur_STT->numStates * sizeof(int *));
        for (int i = 0; i < cur_STT->numStates; i++) {
            int *d_row;  // a pointer declared on CPU that points to a row in GPU used to store an array of int
            cudaMalloc(&d_row, (ALPHABET_SIZE + 1) * sizeof(int));
            cudaMemcpy(d_row, cur_STT->table[i], (ALPHABET_SIZE + 1) * sizeof(int), cudaMemcpyHostToDevice);
            cudaMemcpy(&d_table[i], &d_row, sizeof(int *), cudaMemcpyHostToDevice);
        }

        // Copy the table pointer to the device STT struct
        STT h_stt = *cur_STT;  // Copy the host STT to a temporary struct
        h_stt.table = d_table; // Set the table pointer to the device table
        cudaMemcpy(d_stt, &h_stt, sizeof(STT), cudaMemcpyHostToDevice);

        // Set up CUDA kernel
        int blockSize = 256;
        int gridSize = (textSize + blockSize - 1) / blockSize;
        searchKernel<<<gridSize, blockSize>>>(d_stt, d_text, d_match_count, textSize);

        // Copy results back to the host
        cudaMemcpy(&match_count, d_match_count, sizeof(int), cudaMemcpyDeviceToHost);

        // Cleanup
        cudaFree(d_text);
        cudaFree(d_match_count);
        cudaFree(d_stt);

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
            int c = pattern[i] - 'a'; // char at index i
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
__global__ void searchKernel(STT *d_stt, char *text, int *d_match_count, int textSize) {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        int local_match_count = 0; // Per-thread match count

        // Declare shared memory for reduction
        __shared__ int shared_match_counts[256];

        if (idx < textSize) {
                int state = 0;
                for (int i = idx; i < textSize; i++) {
                        state = d_stt->table[state][text[i] - 'a'];
                        if (state == 0) break; // No valid transition

                        // Check if the current state is a terminal (accepting) state
                        if (d_stt->table[state][ALPHABET_SIZE] == 1) {
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