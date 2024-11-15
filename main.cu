#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "PTrie.hpp"
#include <vector>
#include <string>
#include <iostream>
#include <fstream>
#include "seq_PTrie.hpp"
#include <time.h> 

// Function declarations
void runParallel(std::string textFilename, std::string patternFilename);
void runSequential(std::string textFilename, std::string patternFilename);
std::string readTextFromFile(const std::string& filename);
std::vector<std::string> readPatternsFromFile(const std::string& filename);

// to measure time taken by a specific part of the code 
double time_taken;
clock_t start, end;


int main(int argc, char *argv[]) {

    // Check if the correct number of arguments are provided
    if (argc != 4) {
        std::cerr << "Usage: " << argv[0] << " <mode> <text_filename> <pattern_filename>\n";
        std::cerr << "Mode: 0 for parallel, 1 for sequential\n";
        return 1;
    }

    // Read the mode from the command-line argument
    int mode = std::stoi(argv[1]);
    std::string textFilename = argv[2];
    std::string patternFilename = argv[3];

    if (mode == 0) {
        std::cout << "Running parallel implementation...\n";
        runParallel(textFilename, patternFilename);
    } else if (mode == 1) {
        std::cout << "Running sequential implementation...\n";
        runSequential(textFilename, patternFilename);
    } else {
        std::cerr << "Invalid mode! Use 0 for parallel, 1 for sequential.\n";
        return 1;
    }
    return 0;
}


void runParallel(std::string textFilename, std::string patternFilename) {
    // std::vector<std::string> patterns = {"rat", "rate", "rats", "hat", "hate", "house", "bat", "batter", "batted", "bottle"};
    // Read input DNA sequence from a file
    std::string text = readTextFromFile(textFilename);
    if (text.empty()) {
        std::cerr << "Error: No DNA sequence found in " << textFilename << std::endl;
        return;
    }

    // Read patterns from a file
    std::vector<std::string> patterns = readPatternsFromFile(patternFilename);
    if (patterns.empty()) {
        std::cerr << "Error: No patterns found in " << patternFilename << std::endl;
        return;
    }

    // Construct the PTrie instance by passing in the pre-constructed patterns
    PTrie ptrie(patterns);

    start = clock();
    int count = ptrie.search(text.c_str(), text.length());
    if (count == 0) {
        std::cout << "Pattern not found.\n";
    } else {
        std::cout << "Pattern found with count " << count << "\n";
    }  
    end = clock();  
    time_taken = ((double)(end - start))/ CLOCKS_PER_SEC;
    printf("Time taken = %lf\n", time_taken);
}


void runSequential(std::string textFilename, std::string patternFilename) {
    // Read input DNA sequence from a file
    std::string text = readTextFromFile(textFilename);
    if (text.empty()) {
        std::cerr << "Error: No DNA sequence found in " << textFilename << std::endl;
        return;
    }

    // Read patterns from a file
    std::vector<std::string> patterns = readPatternsFromFile(patternFilename);
    if (patterns.empty()) {
        std::cerr << "Error: No patterns found in " << patternFilename << std::endl;
        return;
    }

    // Construct the sequential Aho-Corasick Trie
    AhoCorasick ac;
    for (const auto& pattern : patterns) {
        ac.insert(pattern);
    }

    // Build the failure links after inserting all patterns
    ac.buildFailureLinks();

    start = clock();
    // Perform the search in the DNA text
    ac.search(text);
    end = clock();  
    time_taken = ((double)(end - start))/ CLOCKS_PER_SEC;
    printf("Time taken = %lf\n", time_taken);
}


// Read DNA sequence from a file
std::string readTextFromFile(const std::string& filename) {
    std::ifstream file(filename);
    std::string text;
    if (file.is_open()) {
        file.seekg(0, std::ios::end);  // Move file pointer to the end of the file to determinr the size of the file
        text.reserve(file.tellg());  // Reserve memory to store contents of the file
        file.seekg(0, std::ios::beg);  // Move file pointer back to the beginning of the file
        text.assign((std::istreambuf_iterator<char>(file)), std::istreambuf_iterator<char>());  // Read the entire content of the file into the text string
        file.close();
    } else {
        std::cerr << "Error: Could not open file " << filename << std::endl;
    }
    return text;
}

// Read patterns from a file
std::vector<std::string> readPatternsFromFile(const std::string& filename) {
    std::ifstream file(filename);
    std::vector<std::string> patterns;
    std::string line;
    if (file.is_open()) {
        while (std::getline(file, line)) {
            patterns.push_back(line);
        }
        file.close();
    } else {
        std::cerr << "Error: Could not open file " << filename << std::endl;
    }
    return patterns;
}


// nvcc -std=c++11 -o main main.cu PTrie.cu seq_PTrie.cpp
// ./main 0 DNA_text_1000.txt Patterns_3bp_10.txt