# Compiler and flags
NVCC := nvcc
NVCCFLAGS := -std=c++11

# Executable
EXEC := main

# Source files
SOURCES := main.cu PTrie.cu seq_PTrie.cpp

# Rule to build the executable
all: $(EXEC)

$(EXEC): $(SOURCES)
	$(NVCC) $(NVCCFLAGS) -o $(EXEC) $(SOURCES)

# Clean up the build
clean:
	rm -f $(EXEC)