#ifndef DNA_UTILS_HPP
#define DNA_UTILS_HPP

#include <string>
#include <vector>

// Function to generate a random DNA sequence of a given length
std::string generateRandomDNA(int length);

// Function to generate multiple random DNA patterns
std::vector<std::string> generatePatterns(int numPatterns, int patternLength);

// Function to save a sequence or patterns to a file
void saveToFile(const std::string& filename, const std::string& content);

#endif // DNA_UTILS_HPP