# Makefile for CuTe examples

CUTLASS_PATH = ..
CUDA_PATH ?= /usr/local/cuda

NVCC = $(CUDA_PATH)/bin/nvcc

# Use SM80 for Ampere
ARCH = sm_80

INCLUDES = -I$(CUTLASS_PATH)/include \
           -I$(CUTLASS_PATH)/tools/util/include \
           -I$(CUDA_PATH)/include

NVCC_FLAGS = -std=c++17 \
             -arch=$(ARCH) \
             --expt-relaxed-constexpr \
             -O2 \
             $(INCLUDES)

# Host-only examples (no GPU kernel needed)
HOST_EXAMPLES = retile_demo layout_products_demo

# GPU examples
GPU_EXAMPLES = mma_ldmatrix_sm80 mma_ldmatrix_real

# SM90 examples (Hopper)
SM90_EXAMPLES = mma_ldmatrix_stmatrix wgmma_tma_warp_specialized_sm90

.PHONY: all clean gpu sm90

all: $(HOST_EXAMPLES)

gpu: $(GPU_EXAMPLES)

sm90: $(SM90_EXAMPLES)

retile_demo: retile_demo.cu
	$(NVCC) $(NVCC_FLAGS) -o $@ $<

layout_products_demo: layout_products_demo.cu
	$(NVCC) $(NVCC_FLAGS) -o $@ $<

mma_ldmatrix_sm80: mma_ldmatrix_sm80.cu
	$(NVCC) $(NVCC_FLAGS) -o $@ $<

mma_ldmatrix_real: mma_ldmatrix_real.cu
	$(NVCC) $(NVCC_FLAGS) -o $@ $<

mma_ldmatrix_stmatrix: mma_ldmatrix_stmatrix.cu
	$(NVCC) $(NVCC_FLAGS) -arch=sm_90 -o $@ $<

wgmma_tma_warp_specialized_sm90: wgmma_tma_warp_specialized_sm90.cu
	$(NVCC) $(NVCC_FLAGS) -arch=sm_90a -o $@ $<

clean:
	rm -f $(HOST_EXAMPLES) $(GPU_EXAMPLES) $(SM90_EXAMPLES)
