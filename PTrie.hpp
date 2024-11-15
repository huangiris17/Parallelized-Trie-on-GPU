#ifndef PTRIE_HPP
#define PTRIE_HPP

#include <vector>
#include <string>
#include <iostream>
// #include <cuda_runtime.h>

#define ALPHABET_SIZE 26  // For lowercase letters

// Structure for the state transition table (STT)
struct STT {
    int numStates;           // Number of states in the STT
    int maxStates;           // Max number of states in the STT
    int **table;             // 2D table for storing state transitions (table: a pointer pointing to an array of pointers)
};

class PTrie {

    private:
        STT* cur_STT;

    public:
        // Constructor
        PTrie(const std::vector<std::string>& patterns);

        // Destructor
        ~PTrie();

        // GPU search function for pattern matching
        int search(const char *text, int textSize);

    private:
        // Calculate upper bound of the number of states
        int calculateMaxStates(const std::vector<std::string>& patterns);

        // Construct the State Transition Table with the max number of states
        STT* createSTT(int maxStates);

        // Insert a pattern into the State Transition Table
        void insertPattern(const char *pattern);

        // Free all STT resources
        void freeSTT();
};

#endif