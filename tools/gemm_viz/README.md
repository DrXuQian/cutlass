# GEMM Decomposition Visualizer

An interactive web-based tool for visualizing the decomposition of matrix multiplication (GEMM) operations into their constituent phases:

1. **CTA Scheduler** - Which CTA handles which output tile
2. **G2S (Global → Shared)** - Threads cooperatively load A and B tiles from global memory to shared memory
3. **S2R (Shared → Register)** - Each thread loads its portion via ldmatrix
4. **MMA (Compute)** - Tensor core or FMA computation
5. **R2S (Register → Shared)** - Write C back (epilogue)

## Quick Start

1. Serve the directory with a local HTTP server:
   ```bash
   cd tools/gemm_viz
   python3 -m http.server 8000
   ```

2. Open in browser: http://localhost:8000

3. The visualizer will load `gemm_data.json` automatically, or you can load your own JSON file.

## Generating JSON Data

The JSON data is generated from `examples/cute/tutorial/sgemm_sm80.cu`:

```bash
# Compile the example
nvcc -std=c++17 -I./include -I./tools/util/include -O2 \
     -o sgemm_sm80_dump ./examples/cute/tutorial/sgemm_sm80.cu \
     --expt-relaxed-constexpr -arch=sm_80

# Generate JSON for different problem sizes
./sgemm_sm80_dump --dump-json --dump-m 128 --dump-n 128 --dump-k 64 > gemm_data.json
./sgemm_sm80_dump --dump-json --dump-m 256 --dump-n 256 --dump-k 128 > large_gemm.json
```

## Generating G2S PDFs (no JSON)

If you want a direct visualization of **G2S (gmem→smem)** derived from the **CUTE layouts** (no JSON),
use `tools/gemm_viz/g2s_pdf.cu`.

This tool generates PDFs via **LaTeX/TikZ internally** (similar style to `print_latex`), but you don’t need to edit TeX.

This emits both:
- **per-stage multi-page PDFs** (easy to scroll)
- **per-step single-page PDFs** (easy to convert to GIF)

Build:
```bash
nvcc -std=c++17 -O2 -I./include -I./tools/util/include -I./examples \
  --expt-relaxed-constexpr --extended-lambda -arch=sm_80 \
  tools/gemm_viz/g2s_pdf.cu -o tools/gemm_viz/g2s_pdf
```

Run:
```bash
./tools/gemm_viz/g2s_pdf --outdir tools/gemm_viz/out_g2s --cell-mm 1.8
```

Notes:
- Requires `pdflatex` in `$PATH` (TeX Live).
- Default output directory is `<exe_dir>/out_g2s` (next to `tools/gemm_viz/g2s_pdf`).
- Use `--pipe 0/1/2` to generate a single pipeline stage while iterating on layout.
- By default, each page overlays **per-thread boxes** labeled with `tid` to show which thread copies which 16B vector.

Outputs:
- `tools/gemm_viz/out_g2s/g2sA_stage_pipe0_ktile0.pdf` (A-only, and pipe1/pipe2)
- `tools/gemm_viz/out_g2s/g2sB_stage_pipe0_ktile0.pdf` (B-only, and pipe1/pipe2)
- `tools/gemm_viz/out_g2s/g2s_stage_pipe0_ktile0.pdf` (interleaved: A step0, B step0, A step1, B step1, ...)
- `tools/gemm_viz/out_g2s/steps/pipe0_ktile0/A_step_000.pdf` ... `A_step_007.pdf` (and pipe1/pipe2)
- `tools/gemm_viz/out_g2s/steps/pipe0_ktile0/B_step_000.pdf` ... `B_step_007.pdf` (and pipe1/pipe2)
- `tools/gemm_viz/out_g2s/layouts/copyA.pdf` and `tools/gemm_viz/out_g2s/layouts/copyB.pdf`

## Controls

- **Phase**: Select which decomposition phase to visualize
- **Stage**: Step through iterations within the current phase
- **Thread**: Select which thread to inspect (0-127)
- **Playback**: Animate through stages automatically

### Stage semantics (important)

- **G2S**: `stage = pipe_stage * (pairs_per_thread) + cp.async_index`  
  This lets you see both the pipeline stage (`pipe_stage`) and the per-stage `cp.async` sequence.
- **S2R**: `stage = k_block * (cp_count) + cp_index`  
  This highlights one “column” (cp index) of the `ldmatrix`/S2R view at a time.

## Understanding the Views

### Global Memory Panel
Shows the input matrices A, B and output matrix C at the global memory level.

### Shared Memory Panel
Shows the CTA tile in shared memory:
- **sA**: Shared memory tile for matrix A (bM × bK)
- **sB**: Shared memory tile for matrix B (bN × bK)

### Register Panel
Shows per-thread register state:
- **REG A**: Fragment of A loaded via ldmatrix
- **REG B**: Fragment of B loaded via ldmatrix
- **REG C**: Accumulator fragment for output

### Inspector
Detailed information about the current selection/hover.

### Timeline
Visual representation of the pipeline stages.

## JSON Schema

The input JSON follows schema version 3 from `sgemm_sm80.cu --dump-json`:

```json
{
  "schema_version": 3,
  "problem": { "M": 128, "N": 128, "K": 64, ... },
  "cta": { "bM": 128, "bN": 128, "bK": 64, "pipe": 3 },
  "g2s": { ... },
  "s2r": { ... },
  "mma": { ... },
  "threads": [
    {
      "thread": 0,
      "warp": 0,
      "lane": 0,
      "a_g2s_mk": [...],  // G2S mapping for A
      "b_g2s_nk": [...],  // G2S mapping for B
      "a_s2r": [...],     // S2R mapping for A
      "b_s2r": [...],     // S2R mapping for B
      "c_mn": [...]       // C fragment mapping
    },
    ...
  ]
}
```

## Key Concepts Visualized

### CTA Scheduler
Shows how the output matrix C is partitioned into CTA tiles. Each CTA computes a bM×bN tile of C.

### G2S (cp.async)
- 128 threads cooperatively copy 128×64 tiles of A and B
- Each thread handles specific (m,k) or (n,k) elements
- Uses `SM80_CP_ASYNC_CACHEALWAYS<uint128_t>` for 16-byte loads

### S2R (ldmatrix)
- Uses `SM75_U32x4_LDSM_N` copy atom
- K-dimension is partitioned into k_blocks (4 blocks of 16 elements each)
- Each thread loads 8 elements per ldmatrix instruction
- Shows bank conflict-free access patterns via swizzled layout

### MMA
- Uses `SM80_16x8x16_F16F16F16F16_TN` tensor core instruction
- Each thread owns a fragment of the C accumulator
- Fragment shape: (v=4, mma_m=4, mma_n=8) = 128 elements per thread
