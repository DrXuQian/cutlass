# Makefile for Machete GEMM Example
#
# Usage:
#   make              - Build the example
#   make clean        - Clean build artifacts
#   make run          - Build and run with default parameters
#   make sweep        - Build and run sweep over multiple sizes
#

# CUDA compiler
NVCC := nvcc

# CUTLASS root directory (adjust if needed)
CUTLASS_ROOT := ..

# Include paths
INCLUDES := -I$(CUTLASS_ROOT)/include \
            -I$(CUTLASS_ROOT)/tools/util/include \
            -I$(CUTLASS_ROOT)/examples/common \
            -I.

# Compiler flags
NVCC_FLAGS := -O3 \
              -std=c++17 \
              --expt-relaxed-constexpr \
              -gencode arch=compute_90,code=sm_90

# Linker flags
LDFLAGS := -lcuda

# Source and target
SRC := machete_gemm_example.cu
TARGET := machete_gemm_example

# Default target
all: $(TARGET)

$(TARGET): $(SRC)
	$(NVCC) $(NVCC_FLAGS) $(INCLUDES) $< -o $@ $(LDFLAGS)

# Run with default parameters (M=1, single token decode)
run: $(TARGET)
	./$(TARGET) --m=1 --n=2048 --k=2048 --g=128 --mode=1

# Run sweep over typical LLM sizes
sweep: $(TARGET)
	@echo "=== Single Token Decode (M=1) ==="
	./$(TARGET) --m=1 --n=2048 --k=2048 --g=128
	./$(TARGET) --m=1 --n=4096 --k=4096 --g=128
	./$(TARGET) --m=1 --n=8192 --k=8192 --g=128
	@echo ""
	@echo "=== Small Batch Decode ==="
	./$(TARGET) --m=4 --n=4096 --k=4096 --g=128
	./$(TARGET) --m=8 --n=4096 --k=4096 --g=128
	./$(TARGET) --m=16 --n=4096 --k=4096 --g=128
	@echo ""
	@echo "=== Prefill (Larger M) ==="
	./$(TARGET) --m=128 --n=4096 --k=4096 --g=128
	./$(TARGET) --m=256 --n=4096 --k=4096 --g=128
	./$(TARGET) --m=512 --n=4096 --k=4096 --g=128

# LLaMA-7B FFN sizes
llama: $(TARGET)
	@echo "=== LLaMA-7B FFN Sizes ==="
	./$(TARGET) --m=1 --n=11008 --k=4096 --g=128
	./$(TARGET) --m=1 --n=4096 --k=11008 --g=128
	./$(TARGET) --m=128 --n=11008 --k=4096 --g=128
	./$(TARGET) --m=128 --n=4096 --k=11008 --g=128

clean:
	rm -f $(TARGET)

.PHONY: all run sweep llama clean
