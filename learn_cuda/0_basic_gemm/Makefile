NVCC = nvcc
NVCC_FLAGS = -O3 -std=c++17 -arch=sm_80
INCLUDES = -I../../include -I../common
LIBS = -lcudart -lcublas

# Target executable
TARGET = basic_gemm

# Source files
SOURCES = basic_gemm.cu

# Build target
$(TARGET): $(SOURCES)
	$(NVCC) $(NVCC_FLAGS) $(INCLUDES) -o $@ $^ $(LIBS)

# Run the program
run: $(TARGET)
	./$(TARGET)

# Clean build artifacts
clean:
	rm -f $(TARGET)

.PHONY: clean run