#include "seq_ahoCorasick.hpp"
#include <iostream>

int main() {
    AhoCorasick ac;

    // Insert patterns
    ac.insert("he");
    ac.insert("she");
    ac.insert("his");
    ac.insert("hers");

    // Build failure links after inserting patterns
    ac.buildFailureLinks();

    // Search patterns in a given text
    std::string text = "ushers";
    ac.search(text);

    return 0;
}

// g++ -o program main.cpp seq_ahoCorasick.cpp -std=c++11
// ./program