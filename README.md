# Parallelized Trie library (PTrie)

## Project Description
This project implements a parallelized Trie library, PTrie, on CUDA to search for DNA patterns in a given text file. The program supports both parallel (GPU-based) and sequential (CPU-based) implementations to evaluate the speedup (CPU Time / GPU Time)

---

## How to Compile the code
1. Ensure having CUDA installed on the system
2. To compile the program, run:
    ```
    make
    ```

---

## How to run the code
* After compiling, run the program with the following command:  
    ```
    ./main <mode> <text_filename> <pattern_filename> <blockSize>
    ```
* Commend-line arguments
    1. ```<mode>```: 
        * ```0```: Run the parallel implementation (PTrie library) on the GPU
        * ```1```: Run the sequential implementation on the CPU
    2. ```<text_filename>```: 
        * Path the the file containing the DNA text sequence
        * Example: ```DNA_text_100000000.txt```
    3. ```<pattern_filename>```:
        * Path to the file containing DNA patterns to search for
        * Example: ```Patterns_3bp_10.txt```
    4. ```<blockSize>```:
        * Only for parallel mode; this value does not affect the sequential code (```mode 0```)
        * The size of each CUDA block, which determines GPU thread grouping
        * Example: 256