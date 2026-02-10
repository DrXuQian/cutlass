# CuTe Visualizer Demo (static page)

This folder contains a zero-dependency demo page to visualize **tiledMMA registers** in `sgemm_sm80`:

- **REG A** / **REG B**: the per-thread fragments consumed by `gemm(mma, A, B, C)` (vary with the K-loop iteration)
- **REG C**: the per-thread accumulator fragment (same indices throughout the K loop; values accumulate)

Use the **Thread** and **Iter** controls to see which `(m,n,k)` coordinates each register element corresponds to.

## Open the demo

- Recommended (so the page can fetch the bundled `ldmatrix.json`):
  ```bash
  python3 -m http.server 8000
  ```
  Then open `http://localhost:8000/tools/cute_viz_demo/`.

- You can also open `tools/cute_viz_demo/index.html` directly (it will fall back to mock data).

## Load real data from CuTe (JSON)

`examples/cute/tutorial/sgemm_sm80.cu` supports a host-only JSON dump:

```bash
nvcc -std=c++17 -arch=sm_80 --expt-relaxed-constexpr -O2 \
  -I./include -I./tools/util/include \
  examples/cute/tutorial/sgemm_sm80.cu -o sgemm_sm80_dump

./sgemm_sm80_dump --dump-json --dump-k 256 > ldmatrix.json
```

Then (re)load the sample in the page, or use **Advanced → Load JSON** and select `ldmatrix.json`.
