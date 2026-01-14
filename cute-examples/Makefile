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

.PHONY: all clean

all: $(HOST_EXAMPLES)

retile_demo: retile_demo.cu
	$(NVCC) $(NVCC_FLAGS) -o $@ $<

layout_products_demo: layout_products_demo.cu
	$(NVCC) $(NVCC_FLAGS) -o $@ $<

clean:
	rm -f $(HOST_EXAMPLES)
