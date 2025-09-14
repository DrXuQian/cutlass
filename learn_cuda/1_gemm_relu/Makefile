NVCC = nvcc
NVCC_FLAGS = -O3 -std=c++17 -arch=sm_80 --expt-relaxed-constexpr
INCLUDES = -I../../include -I../common
LIBS = -lcudart -lcublas

# Targets
TARGETS = gemm_relu gemm_relu_cutlass

all: $(TARGETS)

# Basic GEMM + ReLU implementations
gemm_relu: gemm_relu.cu
	$(NVCC) $(NVCC_FLAGS) $(INCLUDES) -o $@ $< $(LIBS)

# CUTLASS version with fused epilogue
gemm_relu_cutlass: gemm_relu_cutlass.cu
	$(NVCC) $(NVCC_FLAGS) $(INCLUDES) -o $@ $< $(LIBS)

# Run basic version
run: gemm_relu
	./gemm_relu

# Run CUTLASS version
run_cutlass: gemm_relu_cutlass
	./gemm_relu_cutlass

# Run all
run_all: $(TARGETS)
	@echo "=== Running Basic GEMM + ReLU ==="
	./gemm_relu
	@echo "\n\n=== Running CUTLASS GEMM + ReLU ==="
	./gemm_relu_cutlass

# Clean
clean:
	rm -f $(TARGETS)

.PHONY: all clean run run_cutlass run_all