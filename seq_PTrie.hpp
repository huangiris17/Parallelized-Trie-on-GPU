#ifndef AHOCORASICK_HPP
#define AHOCORASICK_HPP
#include <unordered_map>
#include <vector>
#include <string>

struct TrieNode {
    std::unordered_map<char, int> children;  // {char: state}; Need to change to GPU-friendly format!
    int fail_link = 0;
    bool is_final = false;  // Whether this state represent the end of a pattern
    std::vector<std::string> matched_patterns;  // List of patterns matched at this stage
};


class AhoCorasick {
public:
    AhoCorasick();  // Constructor
    void insert(const std::string& pattern);
    void buildFailureLinks();  // Need to change to GPU-friendly format!
    void search(const std::string& text);  // Need to change to GPU-friendly format!

private:
    std::vector<TrieNode> trie;  // STT
};

#endif // AHOCORASICK_HPP