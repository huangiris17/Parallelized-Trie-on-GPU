#include "DNAUtils.hpp"
#include <fstream>
#include <iostream>
#include <cstdlib>
#include <ctime>

// Function to generate a random DNA sequence of a given length
std::string generateRandomDNA(int length) {
    const char DNA_Bases[] = {'A', 'T', 'C', 'G'};
    std::string sequence;
    sequence.reserve(length);
    for (int i = 0; i < length; i++) {
        sequence += DNA_Bases[rand() % 4];
    }
    return sequence;
}

// Function to generate multiple random DNA patterns
std::vector<std::string> generatePatterns(int numPatterns, int patternLength) {
    std::vector<std::string> patterns;
    for (int i = 0; i < numPatterns; i++) {
        patterns.push_back(generateRandomDNA(patternLength));
    }
    return patterns;
}

// Function to save content to a file
void saveToFile(const std::string& filename, const std::string& content) {
    std::ofstream outFile(filename);
    if (outFile.is_open()) {
        outFile << content;
        outFile.close();
    } else {
        std::cerr << "Error opening file: " << filename << std::endl;
    }
}