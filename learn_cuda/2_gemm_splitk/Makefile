NVCC = nvcc
NVCC_FLAGS = -O3 -std=c++17 -arch=sm_80 --expt-relaxed-constexpr
INCLUDES = -I../../include -I../common
LIBS = -lcudart -lcublas

# Targets
TARGETS = gemm_splitk gemm_splitk_simple

# Source files
SOURCES = gemm_splitk.cu gemm_splitk_simple.cu

# Build all targets
all: $(TARGETS)

gemm_splitk: gemm_splitk.cu
	$(NVCC) $(NVCC_FLAGS) $(INCLUDES) -o $@ $< $(LIBS)

gemm_splitk_simple: gemm_splitk_simple.cu
	$(NVCC) $(NVCC_FLAGS) $(INCLUDES) -o $@ $< $(LIBS)

splitk_demo: splitk_demo.cu
	$(NVCC) $(NVCC_FLAGS) $(INCLUDES) -o $@ $< $(LIBS)

# Run with default parameters
run: gemm_splitk_simple
	./gemm_splitk_simple

# Run with different K dimensions
test_k: $(TARGET)
	@echo "=== Testing different K dimensions ==="
	@echo "\n--- Small K (1024) ---"
	./$(TARGET) 1024 1024 1024 8
	@echo "\n--- Medium K (4096) ---"
	./$(TARGET) 1024 1024 4096 8
	@echo "\n--- Large K (8192) ---"
	./$(TARGET) 1024 1024 8192 8
	@echo "\n--- Very Large K (16384) ---"
	./$(TARGET) 1024 1024 16384 8

# Run with different split-k slices
test_slices: $(TARGET)
	@echo "=== Testing different Split-K slices (K=8192) ==="
	@echo "\n--- 1 slice (no split) ---"
	./$(TARGET) 1024 1024 8192 1
	@echo "\n--- 4 slices ---"
	./$(TARGET) 1024 1024 8192 4
	@echo "\n--- 8 slices ---"
	./$(TARGET) 1024 1024 8192 8
	@echo "\n--- 16 slices ---"
	./$(TARGET) 1024 1024 8192 16

# Run with skinny matrices
test_skinny: $(TARGET)
	@echo "=== Testing skinny matrices ==="
	./$(TARGET) 128 128 8192 8
	./$(TARGET) 256 256 8192 8
	./$(TARGET) 512 512 8192 8

# Clean
clean:
	rm -f $(TARGET)

.PHONY: run test_k test_slices test_skinny clean