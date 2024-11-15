#include "seq_PTrie.hpp"
#include <queue>
#include <iostream>

// Initialize an AhoCorasick object with the root node
AhoCorasick::AhoCorasick() {
    trie.emplace_back(TrieNode());
}

// Method insert
void AhoCorasick::insert(const std::string& pattern) {
    int curr_state = 0;  // root of the trie
    for (char c: pattern) {
        // If the transition to c doesn't exist, add it to the trie (STT)
        if (trie[curr_state].children.find(c) == trie[curr_state].children.end()) {
            trie[curr_state].children[c] = trie.size();
            trie.emplace_back(TrieNode());
        }
        // update current_state to the next state in the Trie
        curr_state = trie[curr_state].children[c];
    }
    trie[curr_state].is_final = true;
    trie[curr_state].matched_patterns.push_back(pattern);
}

// Method build failure links
void AhoCorasick::buildFailureLinks() {
    // Use BFS to construct failure links
    std::queue<int> q;
    // Add first level nodes of the trie to queue
    for (const auto& child: trie[0].children) {
        int child_state = child.second;
        trie[child_state].fail_link = 0;  // First level failure links point to the root
        q.push(child_state);
    }
    // Construct failure links for test of the nodes
    while (!q.empty()) {
        int curr_state = q.front();
        q.pop();
        // Construct the failure links for curr_state's all children
        for (const auto& child: trie[curr_state].children) {
            char c = child.first;
            int next_state = child.second;
            q.push(next_state);
            
            // Use the curr_state’s failure link as a starting point to avoid re-checking the entire pattern from scratch
            int failure_state = trie[curr_state].fail_link;
            while (failure_state != 0 && trie[failure_state].children.find(c) == trie[failure_state].children.end()) {
                failure_state = trie[failure_state].fail_link;
            }
            // If fail_state has a child for "c", set it as the failure link for next_state
            if (trie[failure_state].children.find(c) != trie[failure_state].children.end()) {
                failure_state = trie[failure_state].children[c];
            }
            trie[next_state].fail_link = failure_state;
            // Update the matched_patterns of next_state to reflect it can still reach the pattern via a failure link
            if (trie[failure_state].is_final) {
                trie[next_state].matched_patterns.insert(
                    trie[next_state].matched_patterns.end(),
                    trie[failure_state].matched_patterns.begin(),  // range
                    trie[failure_state].matched_patterns.end()  // range
                );
            }
        }
    }
}

// Method search
void AhoCorasick::search(const std::string& text) {
    int curr_state = 0;
    int count = 0;
    for (size_t i = 0; i < text.size(); i++) {
        char c = text[i];

        // If failure, use the failure links to retry other paths
        while (curr_state != 0 && trie[curr_state].children.find(c) == trie[curr_state].children.end()) {
            curr_state = trie[curr_state].fail_link;
        }
        // If we found a match to "c" in the current state, continue trying this path
        if (trie[curr_state].children.find(c) != trie[curr_state].children.end()) {
            curr_state = trie[curr_state].children[c];
        }
        // Find a match!
        if (trie[curr_state].is_final) {
            for (const std::string& pattern : trie[curr_state].matched_patterns) {
                // std::cout << "Pattern \"" << pattern << "\" found at index " << i - pattern.size() + 1 << std::endl;
                count++;
            }
        }
    }
    std::cout << count << std::endl;
}