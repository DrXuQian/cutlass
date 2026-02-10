/* CuTe Visualizer Demo (static, no build step)
 *
 * Register-only view for sgemm_sm80 tiledMMA:
 *   - REG A fragment (per iteration)
 *   - REG B fragment (per iteration)
 *   - REG C accumulator fragment (thread-local, constant over K loop)
 *
 * Input JSON is produced by:
 *   examples/cute/tutorial/sgemm_sm80.cu --dump-json
 */

function $(id) {
  return document.getElementById(id);
}

function clampInt(x, lo, hi) {
  const n = Number(x);
  if (!Number.isFinite(n)) return lo;
  return Math.max(lo, Math.min(hi, Math.trunc(n)));
}

function isSchemaV3(data) {
  return (
    data &&
    data.schema_version === 3 &&
    data.problem &&
    data.cta &&
    data.pipeline &&
    data.s2r &&
    data.mma &&
    Array.isArray(data.k_blocks) &&
    Array.isArray(data.threads)
  );
}

function buildMockV3() {
  const bM = 128;
  const bN = 128;
  const bK = 64;
  const pipe = 3;
  const K = 256;

  const kBlockCount = 4;
  const kBlockLen = 16;
  const cpCount = 8;
  const mmaMCount = 4;
  const mmaNCount = 4;

  const k_blocks = [];
  for (let kb = 0; kb < kBlockCount; ++kb) {
    const k_vals = Array.from({ length: kBlockLen }, (_, i) => kb * kBlockLen + i);
    k_blocks.push({ k_block: kb, k_len: kBlockLen, k_vals });
  }

  const threads = [];
  for (let thr = 0; thr < 128; ++thr) {
    const warp = Math.floor(thr / 32);
    const lane = thr % 32;
    const a_s2r = [];
    const b_s2r = [];
    for (let kb = 0; kb < kBlockCount; ++kb) {
      const m = Array.from({ length: mmaMCount }, (_, mm) =>
        Array.from({ length: cpCount }, () => mm * 32 + lane),
      );
      const n = Array.from({ length: mmaNCount }, (_, nn) =>
        Array.from({ length: cpCount }, () => nn * 32 + lane),
      );
      const k_col = Array.from({ length: mmaMCount }, () =>
        Array.from({ length: cpCount }, (_, cp) => (cp * 2) % kBlockLen),
      );
      a_s2r.push({ k_block: kb, m, k_col });
      b_s2r.push({ k_block: kb, n, k_col });
    }

    const c_mn = [];
    const c_v = 4;
    const c_m = 4;
    const c_n = 8;
    for (let v = 0; v < c_v; ++v) {
      for (let mm = 0; mm < c_m; ++mm) {
        for (let nn = 0; nn < c_n; ++nn) {
          c_mn.push((v * 7 + mm * 5 + lane) % bM, (v * 11 + nn * 3 + warp * 8) % bN);
        }
      }
    }

    threads.push({ thread: thr, warp, lane, a_s2r, b_s2r, c_mn });
  }

  return {
    schema_version: 3,
    focus: "mock_sgemm_sm80_regs",
    problem: { M: bM, N: bN, K, transA: "T", transB: "N", ldA: K, ldB: K, ldC: bM },
    cta: { bM, bN, bK, pipe },
    pipeline: { k_tile_count: Math.floor((K + bK - 1) / bK), k_block_count: kBlockCount },
    s2r: { atom: "SM75_U32x4_LDSM_N", cp_count: cpCount, mma_m_count: mmaMCount, mma_n_count: mmaNCount, k_block_count: kBlockCount, k_block_len: kBlockLen },
    mma: {
      op: "SM80_16x8x16_F16F16F16F16_TN",
      tile_shape: "mock",
      layoutA_TV: "mock",
      layoutB_TV: "mock",
      layoutC_TV: "mock",
      thr_layout_vmnk: "mock",
      c_v_count: 4,
      c_m_count: 4,
      c_n_count: 8,
    },
    k_blocks,
    threads,
  };
}

function normalizeV3(data) {
  if (!isSchemaV3(data)) {
    throw new Error("Unsupported JSON. Expected schema_version=3 output from sgemm_sm80.cu --dump-json.");
  }
  const threads = [...data.threads].sort((a, b) => (a.thread ?? 0) - (b.thread ?? 0));
  return { ...data, threads };
}

function buildTableGrid(container, opts) {
  const { rows, cols, rowLabel, colLabel, cellText, cellTitle, onHover, onClick } = opts;

  container.textContent = "";
  const grid = document.createElement("div");
  grid.className = "grid";
  grid.style.gridTemplateColumns = `72px repeat(${cols}, 78px)`;
  grid.style.gridAutoRows = "26px";

  const corner = document.createElement("div");
  corner.className = "cell header";
  corner.textContent = "";
  grid.appendChild(corner);

  for (let c = 0; c < cols; ++c) {
    const h = document.createElement("div");
    h.className = "cell header";
    h.textContent = colLabel(c);
    grid.appendChild(h);
  }

  for (let r = 0; r < rows; ++r) {
    const lab = document.createElement("div");
    lab.className = "cell rowlab";
    lab.textContent = rowLabel(r);
    grid.appendChild(lab);

    for (let c = 0; c < cols; ++c) {
      const cell = document.createElement("div");
      cell.className = "cell";
      cell.textContent = cellText(r, c);
      const title = cellTitle?.(r, c);
      if (title) cell.title = title;
      if (onHover) cell.addEventListener("mouseenter", () => onHover(r, c, cell));
      if (onClick) cell.addEventListener("click", () => onClick(r, c, cell));
      grid.appendChild(cell);
    }
  }

  container.appendChild(grid);
}

function main() {
  const state = {
    data: null,
    thread: 0,
    iter: 0,
    pinned: null,
  };

  function setInspector(lines) {
    $("inspector").textContent = Array.isArray(lines) ? lines.join("\n") : String(lines);
  }

  function getThreadObj(thread) {
    const obj = state.data.threads[thread];
    if (!obj) throw new Error(`thread ${thread} not present`);
    return obj;
  }

  function getKBlockObj(kBlock) {
    const obj = state.data.k_blocks[kBlock];
    if (!obj) throw new Error(`k_block ${kBlock} not present`);
    return obj;
  }

  function getThreadBlock(threadObj, which, kBlock) {
    const key = which === "A" ? "a_s2r" : "b_s2r";
    const block = (threadObj[key] ?? []).find((x) => x.k_block === kBlock);
    if (!block) throw new Error(`thread ${threadObj.thread} missing ${key} for k_block ${kBlock}`);
    return block;
  }

  function totalIters() {
    const t = state.data.pipeline.k_tile_count;
    const b = state.data.pipeline.k_block_count;
    return Math.max(1, t * b);
  }

  function iterToKB(iter) {
    const kBlockCount = state.data.pipeline.k_block_count;
    const kTile = Math.floor(iter / kBlockCount);
    const kBlock = iter % kBlockCount;
    return { kTile, kBlock };
  }

  function updateThreadInfo() {
    const t = getThreadObj(state.thread);
    $("threadInfo").textContent = `warp ${t.warp}, lane ${t.lane}`;
  }

  function updateIterInfo() {
    const { kTile, kBlock } = iterToKB(state.iter);
    const baseK = kTile * state.data.cta.bK;
    const kb = getKBlockObj(kBlock);
    const kMin = Math.min(...kb.k_vals) + baseK;
    const kMax = Math.max(...kb.k_vals) + baseK;
    $("iterInfo").textContent = `k_tile ${kTile}, k_block ${kBlock} · k∈[${kMin}..${kMax}]`;
  }

  function renderRegA() {
    const t = getThreadObj(state.thread);
    const { kTile, kBlock } = iterToKB(state.iter);
    const baseK = kTile * state.data.cta.bK;
    const kb = getKBlockObj(kBlock);
    const blk = getThreadBlock(t, "A", kBlock);
    const rows = blk.m.length;
    const cols = blk.m[0]?.length ?? 0;

    $("regAMeta").textContent = `iter ${state.iter}/${totalIters() - 1} · (k_tile=${kTile}, k_block=${kBlock}) · k_vals=[${kb.k_vals.map((x) => x + baseK).join(", ")}]`;

    buildTableGrid($("regAGrid"), {
      rows,
      cols,
      rowLabel: (r) => `m${r}`,
      colLabel: (c) => `cp${c}`,
      cellText: (r, c) => {
        const m = blk.m[r][c];
        const kCol = blk.k_col[r][c];
        const kAbs = kb.k_vals[kCol];
        return `${m},${baseK + kAbs}`;
      },
      cellTitle: (r, c) => {
        const m = blk.m[r][c];
        const kCol = blk.k_col[r][c];
        const kAbs = kb.k_vals[kCol];
        return `A reg: idx(mma_m=${r}, cp=${c}) -> (m=${m}, k=${baseK + kAbs})  (k_abs=${kAbs}, k_col=${kCol})`;
      },
      onHover: (r, c) => {
        if (state.pinned) return;
        const m = blk.m[r][c];
        const kCol = blk.k_col[r][c];
        const kAbs = kb.k_vals[kCol];
        setInspector([
          `thread=${t.thread} (warp=${t.warp}, lane=${t.lane})`,
          `iter=${state.iter}  k_tile=${kTile}  k_block=${kBlock}`,
          ``,
          `REG A: (mma_m=${r}, cp=${c})`,
          `(m,k)=(${m},${baseK + kAbs})  (k_abs=${kAbs}, k_col=${kCol})`,
        ]);
      },
    });
  }

  function renderRegB() {
    const t = getThreadObj(state.thread);
    const { kTile, kBlock } = iterToKB(state.iter);
    const baseK = kTile * state.data.cta.bK;
    const kb = getKBlockObj(kBlock);
    const blk = getThreadBlock(t, "B", kBlock);
    const rows = blk.n.length;
    const cols = blk.n[0]?.length ?? 0;

    $("regBMeta").textContent = `iter ${state.iter}/${totalIters() - 1} · (k_tile=${kTile}, k_block=${kBlock}) · k_vals=[${kb.k_vals.map((x) => x + baseK).join(", ")}]`;

    buildTableGrid($("regBGrid"), {
      rows,
      cols,
      rowLabel: (r) => `n${r}`,
      colLabel: (c) => `cp${c}`,
      cellText: (r, c) => {
        const n = blk.n[r][c];
        const kCol = blk.k_col[r][c];
        const kAbs = kb.k_vals[kCol];
        return `${n},${baseK + kAbs}`;
      },
      cellTitle: (r, c) => {
        const n = blk.n[r][c];
        const kCol = blk.k_col[r][c];
        const kAbs = kb.k_vals[kCol];
        return `B reg: idx(mma_n=${r}, cp=${c}) -> (n=${n}, k=${baseK + kAbs})  (k_abs=${kAbs}, k_col=${kCol})`;
      },
      onHover: (r, c) => {
        if (state.pinned) return;
        const n = blk.n[r][c];
        const kCol = blk.k_col[r][c];
        const kAbs = kb.k_vals[kCol];
        setInspector([
          `thread=${t.thread} (warp=${t.warp}, lane=${t.lane})`,
          `iter=${state.iter}  k_tile=${kTile}  k_block=${kBlock}`,
          ``,
          `REG B: (mma_n=${r}, cp=${c})`,
          `(n,k)=(${n},${baseK + kAbs})  (k_abs=${kAbs}, k_col=${kCol})`,
        ]);
      },
    });
  }

  function renderRegC() {
    const t = getThreadObj(state.thread);
    const vCount = state.data.mma.c_v_count;
    const mCount = state.data.mma.c_m_count;
    const nCount = state.data.mma.c_n_count;
    const pairs = t.c_mn ?? [];

    $("regCMeta").textContent = `thread=${t.thread} · fragment shape=(${vCount},${mCount},${nCount}) · (m,n) pairs=${pairs.length / 2}`;

    const container = $("regCGrid");
    container.textContent = "";

    for (let v = 0; v < vCount; ++v) {
      const header = document.createElement("div");
      header.className = "meta";
      header.textContent = `v=${v}`;
      container.appendChild(header);

      const wrap = document.createElement("div");
      wrap.className = "gridWrap";
      container.appendChild(wrap);

      const base = v * mCount * nCount * 2;
      buildTableGrid(wrap, {
        rows: mCount,
        cols: nCount,
        rowLabel: (r) => `m${r}`,
        colLabel: (c) => `n${c}`,
        cellText: (r, c) => {
          const idx = base + (r * nCount + c) * 2;
          return `${pairs[idx]},${pairs[idx + 1]}`;
        },
        cellTitle: (r, c) => {
          const idx = base + (r * nCount + c) * 2;
          const m = pairs[idx];
          const n = pairs[idx + 1];
          return `C reg: idx(v=${v}, mma_m=${r}, mma_n=${c}) -> (m=${m}, n=${n})`;
        },
        onHover: (r, c) => {
          if (state.pinned) return;
          const idx = base + (r * nCount + c) * 2;
          const m = pairs[idx];
          const n = pairs[idx + 1];
          setInspector([
            `thread=${t.thread} (warp=${t.warp}, lane=${t.lane})`,
            `iter=${state.iter} (A/B change with iter; C accumulates)`,
            ``,
            `REG C: (v=${v}, mma_m=${r}, mma_n=${c})`,
            `(m,n)=(${m},${n})`,
          ]);
        },
      });
    }
  }

  function renderSummary() {
    const kTileCount = state.data.pipeline.k_tile_count;
    const kBlockCount = state.data.pipeline.k_block_count;
    const mma = state.data.mma;
    setInspector([
      `Loaded: focus=${state.data.focus} schema=${state.data.schema_version}`,
      `problem: M=${state.data.problem.M} N=${state.data.problem.N} K=${state.data.problem.K}`,
      `cta: bM=${state.data.cta.bM} bN=${state.data.cta.bN} bK=${state.data.cta.bK} pipe=${state.data.cta.pipe}`,
      `mma: op=${mma.op}  tile_shape=${mma.tile_shape}`,
      `loop: k_tile_count=${kTileCount}  k_block_count=${kBlockCount}  mma_iters=${kTileCount * kBlockCount}`,
      ``,
      `mma.thr_layout_vmnk: ${mma.thr_layout_vmnk}`,
      `mma.layoutA_TV: ${mma.layoutA_TV}`,
      `mma.layoutB_TV: ${mma.layoutB_TV}`,
      `mma.layoutC_TV: ${mma.layoutC_TV}`,
      ``,
      `Hover cells to see exact register-to-(m,n,k) mapping.`,
    ]);
  }

  function renderAll() {
    if (!state.data) return;
    updateThreadInfo();
    updateIterInfo();
    renderRegA();
    renderRegB();
    renderRegC();
    renderSummary();
  }

  function setData(data) {
    state.data = normalizeV3(data);

    // Thread controls
    const maxThread = state.data.threads.length - 1;
    $("threadSlider").max = String(maxThread);
    $("threadNumber").max = String(maxThread);
    state.thread = clampInt(state.thread, 0, maxThread);
    $("threadSlider").value = String(state.thread);
    $("threadNumber").value = String(state.thread);

    // Iteration controls (linearize k_tile x k_block)
    const maxIter = totalIters() - 1;
    $("iterSlider").max = String(maxIter);
    $("iterNumber").max = String(maxIter);
    state.iter = clampInt(state.iter, 0, maxIter);
    $("iterSlider").value = String(state.iter);
    $("iterNumber").value = String(state.iter);

    renderAll();
  }

  async function loadJsonFile(file) {
    const text = await file.text();
    setData(JSON.parse(text));
  }

  async function loadBundled() {
    const resp = await fetch("ldmatrix.json", { cache: "no-cache" });
    if (!resp.ok) throw new Error(`HTTP ${resp.status}`);
    setData(await resp.json());
  }

  // Wire controls
  const onThreadChange = (v) => {
    if (!state.data) return;
    state.thread = clampInt(v, 0, state.data.threads.length - 1);
    $("threadSlider").value = String(state.thread);
    $("threadNumber").value = String(state.thread);
    renderAll();
  };
  $("threadSlider").addEventListener("input", (e) => onThreadChange(e.target.value));
  $("threadNumber").addEventListener("change", (e) => onThreadChange(e.target.value));

  const onIterChange = (v) => {
    if (!state.data) return;
    const maxIter = totalIters() - 1;
    state.iter = clampInt(v, 0, maxIter);
    $("iterSlider").value = String(state.iter);
    $("iterNumber").value = String(state.iter);
    renderAll();
  };
  $("iterSlider").addEventListener("input", (e) => onIterChange(e.target.value));
  $("iterNumber").addEventListener("change", (e) => onIterChange(e.target.value));

  $("fileInput").addEventListener("change", async (e) => {
    const file = e.target.files?.[0];
    if (!file) return;
    await loadJsonFile(file);
  });

  $("btnBundled").addEventListener("click", async () => {
    try {
      await loadBundled();
    } catch (err) {
      setInspector(
        [
          `Failed to load bundled sample (ldmatrix.json).`,
          `Error: ${String(err)}`,
          ``,
          `Tip: serve the repo with:`,
          `  python3 -m http.server 8000`,
          `then open:`,
          `  http://localhost:8000/tools/cute_viz_demo/`,
        ].join("\n"),
      );
    }
  });

  $("btnMock").addEventListener("click", () => setData(buildMockV3()));

  // Default view: try bundled sample, fallback to mock.
  loadBundled().catch(() => setData(buildMockV3()));
}

main();
