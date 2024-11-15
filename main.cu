#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "PTrie.hpp"
#include <vector>
#include <string>
#include <iostream>
// #include <cuda_runtime.h>

int main() {

    // Construct a vector of pattern for testing
    // Old patterns: {"rat", "rate", "rats", "hat", "hate", "house", "bat", "batter", "batted", "bottle"}
    std::vector<std::string> patterns = {"rat", "rate", "rats", "hat", "hate", "house", "bat", "batter", "batted", "bottle"};

    // Construct the PTrie instance by passing in the pre-constructed patterns
    PTrie ptrie(patterns);

    // Example non-existent pattern search
    std::vector<std::string> test_pattern = {"hatehouserats"};
    int count = ptrie.search(test_pattern[0].c_str(), test_pattern[0].length());
    if (count == 0) {
        std::cout << "Pattern: " << test_pattern[0] << "\" not found.\n";
    } else {
        std::cout << "Pattern: " << test_pattern[0] << " found with count " << count << "\n";
    }

    // Example non-existent pattern search
    std::vector<std::string> non_existent_pattern = {"nonexistent"};
    count = ptrie.search(non_existent_pattern[0].c_str(), non_existent_pattern[0].length());
    if (count == 0) {
        std::cout << "Pattern: " << non_existent_pattern[0] << " correctly not found.\n";
    } else {
        std::cout << "Error: Non-existent pattern: " << non_existent_pattern[0] << " found in the trie.\n";
    }

    return 0;
}