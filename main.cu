#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "PTrie.hpp"
// #include <cuda_runtime.h>

int main() {

    // Construct a vector of pattern for testing
    std::vector<std::string> patterns = {"rat", "rate", "rats", "hat", "hate", "house", "bat", "batter", "batted", "bottle"};

    // Construct the PTrie instance by passing in the pre-constructed patterns
    PTrie Ptrie(patterns);

    // Search each pattern and validate the returned values
    for (const auto& pattern : patterns) {
        int count = Ptrie.search(pattern.c_str(), pattern.length());
        if (count > 0) {
            std::cout << count << " matched patterns found.\n";
        } else {
            std::cout << "0 matched patterns found.\n";
        }
    }

    // Example non-existent pattern search
    std::vector<std::string> non_existent_pattern = {"nonexistent"};
    int count = Ptrie.search(non_existent_pattern[0].c_str(), non_existent_pattern[0].length());
    if (count == 0) {
        std::cout << "Pattern: " << non_existent_pattern[0] << "\" correctly not found.\n";
    } else {
        std::cout << "Error: Non-existent pattern: " << non_existent_pattern[0] << " found in the trie.\n";
    }

    return 0;
}