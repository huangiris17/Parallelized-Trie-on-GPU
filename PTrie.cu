#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <vector>
#include <cuda_runtime.h>

#define ALPHABET_SIZE 26  // lowercase letters

// Structure for the state transition table (STT)
struct STT {
    int numStates;           // Number of states in the STT
    int maxStates;           // Max number of states in the STT
    int **table;             // 2D table for storing state transitions
};


class PTrie {

    private: 
    STT* cur_STT;

    public:
    // Constructor
    PTrie(const std::vector<std::string>& patterns) {
        int maxState = calculateMaxStates(patterns);
        // Create empty STT
        cur_STT = createSTT(maxState);
        cur_STT->maxStates = maxState;

        // Insert patters into STT
        for (const std::string& pattern : patterns) {
            insertPattern(cur_STT, pattern.c_str());
        }
    }

    // Wrapper function to launch kernel and perform matching
    int search(const char *text, int textSize) {
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

    private:
    // Destructor
    ~PTrie() {
        freeSTT();
    }

    // Calculate upper bound of the number of states
    int calculateMaxStates(const std::vector<std::string>& patterns) {
        int cnt = 0;
        for (const std::string& pattern : patterns) {
            cnt += pattern.length();
        }
        return cnt;
    }

    // Construct the State Transition Table with the max number of states
    STT* createSTT(int maxStates) {
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
    void insertPattern(const char *pattern) {
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
    void freeSTT() {
        for (int i = 0; i < cur_STT->maxStates; i++) {
            free(cur_STT->table[i]);
        }
        free(cur_STT->table);
        free(cur_STT);
    }

    // Search function(kernel)
    __global__ void searchKernel(STT *d_stt, char *text, int *d_match_count, int textSize) {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx < textSize) {
            int state = 0;
            for (int i = idx; i < textSize; i++) {
                state = d_stt->table[state][(unsigned char)text[i]];
                if (state == 0) break; // No valid transition
                
                // Check if the current state is a terminal (accepting) state
                if (d_stt->table[state][ALPHABET_SIZE] == 1) {
                    atomicAdd(d_match_count, 1); // Atomically increment the match count
                    break; // Exit after the first match to avoid double-counting
                }
            }
        }
    }
}