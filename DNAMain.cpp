#include "DNAUtils.hpp"
#include <iostream>
#include <vector>
#include <fstream> 

// Define the test cases 
struct TestCase {
    int textLength;
    int patternLength;
    int numPatterns;
};


int main() {
    // Seed the random number generator
    srand(static_cast<unsigned int>(time(0)));

    // Test cases with different N, M values
    std::vector<TestCase> testCases = {
        {100, 3, 10},           // Mini test case
        {1000, 10, 20},         // Small test case
        {10000, 20, 50},      // Medium test case
        {1000000, 100, 500},   // Large test case
        {10000000, 500, 2000}  // Very large test case
    };
    
    // Generate random patterns
    for (const auto& testCase: testCases) {
        // Generate random DNA text
        std::string text = generateRandomDNA(testCase.textLength);
        std::string textFilename = "DNA_text_" + std::to_string(testCase.textLength) + ".txt";
        saveToFile(textFilename, text);
        std::cout << "Generated input text: " << textFilename << std::endl;

        // Generate random patterns
        std::vector<std::string> patterns = generatePatterns(testCase.numPatterns, testCase.patternLength);
        std::string patternFilename = "Patterns_" + std::to_string(testCase.patternLength) + "bp_" + std::to_string(testCase.numPatterns) + ".txt";

        // Save patterns to file
        std::ofstream patternFile(patternFilename);
        if (patternFile.is_open()) {
            for (const auto& pattern : patterns) {
                patternFile << pattern << "\n";
            }
            patternFile.close();
        }
        std::cout << "Generated patterns: " << patternFilename << std::endl;

    }
    return 0;
}

// g++ -std=c++11 -o DNAMain DNAMain.cpp DNAUtils.cpp