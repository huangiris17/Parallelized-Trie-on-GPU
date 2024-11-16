#ifndef PTRIE_TEXTURE_HPP
#define PTRIE_TEXTURE_HPP

#include <vector>
#include <string>
#include <iostream>
#include <cuda_runtime.h>

#define ALPHABET_SIZE 4  // DNA components: A, T, C, G

// Structure for the state transition table (STT)
struct STT {
    int numStates;           // Number of states in the STT
    int maxStates;           // Max number of states in the STT
    int **table;             // 2D table for storing state transitions (pointer to an array of pointers)
};

// Class to handle the construction and searching using the PTrie and texture memory
class PTrie {
private:
    STT* cur_STT;                      // Pointer to the current state transition table (STT)
    cudaTextureObject_t texObj2D;      // Texture object for the STT
    cudaArray_t d_sttArray2D;          // CUDA array for the 2D STT

public:
    // Constructor to initialize the PTrie with a set of patterns
    PTrie(const std::vector<std::string>& patterns);

    // Destructor to free allocated resources
    ~PTrie();

    // GPU search function for pattern matching
    int search(const char *text, int textSize);

private:
    // Calculate the upper bound of the number of states based on the patterns
    int calculateMaxStates(const std::vector<std::string>& patterns);

    // Construct the State Transition Table with the max number of states
    STT* createSTT(int maxStates);

    // Insert a pattern into the State Transition Table
    void insertPattern(const char *pattern);

    // Free all resources associated with the STT
    void freeSTT();
};

#endif // PTRIE_TEXTURE_HPP