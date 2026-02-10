# Makefile for TensorRT-LLM MoE Kernel Extraction
#
# Usage:
#   make              - Build the test
#   make clean        - Clean build artifacts

NVCC := nvcc

# CUTLASS root directory
CUTLASS_ROOT := ..

# Include paths
INCLUDES := -I. \
            -I./tensorrt_llm/cutlass_extensions/include \
            -I$(CUTLASS_ROOT)/include \
            -I$(CUTLASS_ROOT)/tools/util/include

# Compiler flags
NVCC_FLAGS := -O3 \
              -std=c++17 \
              --expt-relaxed-constexpr \
              -gencode arch=compute_90,code=sm_90 \
              -arch=sm_90a \
              -DCUTLASS_ARCH_MMA_SM90_SUPPORTED \
              -DCOMPILE_HOPPER_TMA_GROUPED_GEMMS \
              -DENABLE_BF16

# Linker flags
LDFLAGS := -lcuda

# Source and target
TARGET := moe_fp16_test
BENCH_TARGET := moe_fp16_bench

# Common source files
COMMON_SRCS := tensorrt_llm/common/tllmException.cpp

all: $(TARGET) $(BENCH_TARGET)

# Compile common object files
tllmException.o: tensorrt_llm/common/tllmException.cpp
	$(NVCC) $(NVCC_FLAGS) $(INCLUDES) -c $< -o $@

$(TARGET): moe_fp16_test.cu
	$(NVCC) $(NVCC_FLAGS) $(INCLUDES) $< -o $@ $(LDFLAGS)

$(BENCH_TARGET): moe_fp16_bench.cu tllmException.o
	$(NVCC) $(NVCC_FLAGS) $(INCLUDES) $< tllmException.o -o $@ $(LDFLAGS)

clean:
	rm -f $(TARGET) $(BENCH_TARGET) *.o

.PHONY: all clean
