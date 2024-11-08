#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cstdio>
#include <cstring>

#define MAX_NODES (1024*1024)
#define CHAR_SET_SIZE 26 // lowercase letters for now

__device__ __host__ inline int charToIndex(char c) {
    return c - 'a';
}

struct TrieNode {
    unsigned int bitmap;
    int children[CHAR_SET_SIZE];
    bool is_end_of_word;
};

class Trie{
    public:

    TrieNode* d_trie;
    int* d_node_index;
    int total_nodes;

    Trie(int max_nodes = MAX_NODES) : total_nodes(max_nodes){
        size_t num_bytes = total_nodes * sizeof(TrieNode);
        cudaMalloc((void**)&d_trie, num_bytes);

        TrieNode h_root;
        h_root.bitmap = 0;
        memset(h_root.children, -1, sizeof(h_root.children));
        h_root.is_end_of_word = false;

        cudaMemcpy(d_trie, &h_root, sizeof(TrieNode), cudaMemcpyHostToDevice);

        int h_node_index = 1;

        cudaMalloc((void**)&d_node_index, sizeof(int));
        cudaMemcpy(d_node_index, &h_node_index, sizeof(int), cudaMemcpyHostToDevice);
    }

    // Destructor
    ~Trie(){
        cudaFree(d_trie);
        cudaFree(d_node_index);
    }
};
