/**
 * GEMM Decomposition Visualizer
 *
 * Visualizes the stages of a GEMM operation:
 * 1. CTA Scheduler - which CTA handles which output tile
 * 2. G2S (Global to Shared) - threads cooperatively load A and B tiles
 * 3. S2R (Shared to Register) - each thread loads its portion via ldmatrix
 * 4. MMA - tensor core or FMA computation
 * 5. R2S (Register to Shared) - write C back (optional epilogue)
 */

const $ = id => document.getElementById(id);

function clamp(x, lo, hi) {
  return Math.max(lo, Math.min(hi, x));
}

// Color utilities
const WARP_COLORS = [
  '#3b82f6', '#8b5cf6', '#ec4899', '#f97316'
];

function threadToColor(tid, alpha = 1) {
  const warp = Math.floor(tid / 32);
  const base = WARP_COLORS[warp % WARP_COLORS.length];
  if (alpha >= 1) return base;
  // Convert hex to rgba
  const r = parseInt(base.slice(1, 3), 16);
  const g = parseInt(base.slice(3, 5), 16);
  const b = parseInt(base.slice(5, 7), 16);
  return `rgba(${r},${g},${b},${alpha})`;
}

// Main application state
const state = {
  data: null,
  phase: 'cta',
  stage: 0,
  thread: 0,
  playing: false,
  playInterval: null,
  speed: 500,
  cache: {
    smemA: new Map(), // key "m,k,p" -> { off, bank }
    smemB: new Map(), // key "n,k,p" -> { off, bank }
    g2sPairsPerThread: 0,
    g2sVectorLen: 8
  }
};

// ============ Data Loading ============

function isValidSchema(data) {
  return data &&
    data.schema_version >= 3 &&
    data.problem &&
    data.cta &&
    data.threads;
}

function getG2SVectorLen(data) {
  const bytes = data?.g2s?.bytes_per_inst;
  const elem = data?.g2s?.elem_bytes;
  if (typeof bytes === 'number' && typeof elem === 'number' && elem > 0) {
    const v = Math.floor(bytes / elem);
    if (v > 0 && v <= 32) return v;
  }
  return 8;
}

function getG2SPairsPerThread(data) {
  const cpCount = data?.g2s?.cp_count;
  const cpyM = data?.g2s?.cpy_m;
  const cpyK = data?.g2s?.cpy_k;
  if ([cpCount, cpyM, cpyK].every(x => typeof x === 'number' && x > 0)) {
    return cpCount * cpyM * cpyK;
  }
  const t0 = data?.threads?.[0];
  const len = (t0?.a_g2s_mk?.length || 0) / 2;
  return Math.max(0, Math.floor(len));
}

function buildSmemCoordMap(smemObj, mnKey) {
  const map = new Map();
  if (!smemObj) return map;
  const minOff = smemObj.min_off;
  const offToMn = smemObj[mnKey];
  const offToK = smemObj.off_to_k;
  const offToP = smemObj.off_to_p;
  const offToBank = smemObj.off_to_bank;
  if (![minOff, offToMn, offToK, offToP, offToBank].every(x => x !== undefined)) return map;
  for (let idx = 0; idx < offToMn.length; idx++) {
    const mn = offToMn[idx];
    const k = offToK[idx];
    const p = offToP[idx];
    const bank = offToBank[idx];
    if (mn < 0 || k < 0 || p < 0 || bank < 0) continue;
    const off = minOff + idx;
    map.set(`${mn},${k},${p}`, { off, bank });
  }
  return map;
}

function loadData(data) {
  if (!isValidSchema(data)) {
    setInspector('Invalid JSON schema. Expected schema_version >= 3 from sgemm_sm80 --dump-json');
    return;
  }

  state.data = data;
  state.cache.g2sPairsPerThread = getG2SPairsPerThread(data);
  state.cache.g2sVectorLen = getG2SVectorLen(data);
  state.cache.smemA = buildSmemCoordMap(data.smem?.a, 'off_to_m');
  state.cache.smemB = buildSmemCoordMap(data.smem?.b, 'off_to_n');

  // Update controls
  const maxThread = data.threads.length - 1;
  $('threadSlider').max = maxThread;
  $('threadNumber').max = maxThread;
  state.thread = clamp(state.thread, 0, maxThread);

  updateStageRange();
  renderAll();
}

function updateStageRange() {
  if (!state.data) return;

  let maxStage = 0;
  let stageDesc = '';

  switch (state.phase) {
    case 'cta':
      // Number of CTAs
      const ctaM = Math.ceil(state.data.problem.M / state.data.cta.bM);
      const ctaN = Math.ceil(state.data.problem.N / state.data.cta.bN);
      maxStage = ctaM * ctaN - 1;
      stageDesc = `${ctaM}x${ctaN} CTAs`;
      break;
    case 'g2s':
      // Stage = pipe_stage * (pairs per thread) + cp.async instruction index
      const pipeStages = state.data.cta?.pipe || 1;
      maxStage = Math.max(0, pipeStages * (state.cache.g2sPairsPerThread || 1) - 1);
      stageDesc = `${pipeStages} pipes × ${state.cache.g2sPairsPerThread || 1} instr`;
      break;
    case 's2r':
      // Stage = k_block * cp_count + cp_index
      const kbCount = state.data.s2r?.k_block_count || 1;
      const cpCount = state.data.s2r?.cp_count || 1;
      maxStage = Math.max(0, kbCount * cpCount - 1);
      stageDesc = `${kbCount} k_blocks × ${cpCount} cp`;
      break;
    case 'mma':
      maxStage = (state.data.pipeline?.k_block_count || 1) - 1;
      stageDesc = `${maxStage + 1} k_blocks`;
      break;
    case 'r2s':
      maxStage = 0; // Single stage for now
      stageDesc = 'epilogue';
      break;
  }

  $('stageSlider').max = maxStage;
  $('stageNumber').max = maxStage;
  state.stage = clamp(state.stage, 0, maxStage);
  $('stageSlider').value = state.stage;
  $('stageNumber').value = state.stage;

  // Show phase info
  console.log(`Phase: ${state.phase}, maxStage: ${maxStage}, desc: ${stageDesc}`);
}

// ============ Build mock data for testing ============

function buildMockData() {
  const bM = 128, bN = 128, bK = 64;
  const M = 128, N = 128, K = 64;
  const pipe = 3;
  const kBlockCount = 4;
  const kBlockLen = 16;
  const cpCount = 8;
  const mmaMCount = 4;
  const mmaNCount = 4;

  // G2S mapping: each thread copies specific (m,k) pairs
  const g2sPairsPerThread = 8; // simplified

  const threads = [];
  for (let thr = 0; thr < 128; thr++) {
    const warp = Math.floor(thr / 32);
    const lane = thr % 32;

    // G2S: generate (m,k) pairs this thread copies
    const a_g2s_mk = [];
    const b_g2s_nk = [];
    for (let i = 0; i < g2sPairsPerThread; i++) {
      const m = (thr * g2sPairsPerThread + i) % bM;
      const k = Math.floor((thr * g2sPairsPerThread + i) / bM) % bK;
      a_g2s_mk.push(m, k);
      const n = (thr * g2sPairsPerThread + i) % bN;
      b_g2s_nk.push(n, k);
    }

    // S2R
    const a_s2r = [];
    const b_s2r = [];
    for (let kb = 0; kb < kBlockCount; kb++) {
      const m = Array.from({length: mmaMCount}, (_, mm) =>
        Array.from({length: cpCount}, () => (mm * 32 + lane) % bM)
      );
      const n = Array.from({length: mmaNCount}, (_, nn) =>
        Array.from({length: cpCount}, () => (nn * 32 + lane) % bN)
      );
      const k_col = Array.from({length: mmaMCount}, () =>
        Array.from({length: cpCount}, (_, cp) => cp % kBlockLen)
      );
      a_s2r.push({ k_block: kb, m, k_col });
      b_s2r.push({ k_block: kb, n, k_col });
    }

    // C fragment
    const c_mn = [];
    for (let v = 0; v < 4; v++) {
      for (let mm = 0; mm < 4; mm++) {
        for (let nn = 0; nn < 8; nn++) {
          const mIdx = (warp * 16 + lane % 4 * 2 + v % 2) + mm * 8;
          const nIdx = (Math.floor(lane / 4) * 8 + v * 2 + nn);
          c_mn.push(mIdx % bM, nIdx % bN);
        }
      }
    }

    threads.push({
      thread: thr,
      warp,
      lane,
      a_g2s_mk,
      b_g2s_nk,
      a_s2r,
      b_s2r,
      c_mn
    });
  }

  // K blocks
  const k_blocks = [];
  for (let kb = 0; kb < kBlockCount; kb++) {
    const k_vals = Array.from({length: kBlockLen}, (_, i) => kb * kBlockLen + i);
    k_blocks.push({
      k_block: kb,
      k_len: kBlockLen,
      k_vals,
      a_smem_offsets: [],
      a_smem_banks: [],
      b_smem_offsets: [],
      b_smem_banks: []
    });
  }

  return {
    schema_version: 3,
    focus: 'mock_sgemm_sm80',
    problem: { M, N, K, transA: 'T', transB: 'N', ldA: K, ldB: K, ldC: M },
    cta: { bM, bN, bK, pipe },
    pipeline: { k_tile_count: 1, k_block_count: kBlockCount },
    g2s: {
      atom: 'SM80_CP_ASYNC_CACHEALWAYS<uint128_t>',
      bytes_per_inst: 16,
      elem_bytes: 2,
      cp_count: 1,
      cpy_m: 8,
      cpy_k: 1
    },
    s2r: {
      atom: 'SM75_U32x4_LDSM_N',
      cp_count: cpCount,
      mma_m_count: mmaMCount,
      mma_n_count: mmaNCount,
      k_block_count: kBlockCount,
      k_block_len: kBlockLen
    },
    mma: {
      op: 'SM80_16x8x16_F16F16F16F16_TN',
      c_v_count: 4,
      c_m_count: 4,
      c_n_count: 8
    },
    k_blocks,
    threads
  };
}

// ============ Rendering ============

function setInspector(text) {
  $('inspector').textContent = Array.isArray(text) ? text.join('\n') : text;
}

function clearAllGrids() {
  $('gAGrid').innerHTML = '';
  $('gBGrid').innerHTML = '';
  $('gCGrid').innerHTML = '';
  $('sAGrid').innerHTML = '';
  $('sBGrid').innerHTML = '';
  $('regAGrid').innerHTML = '';
  $('regBGrid').innerHTML = '';
  $('regCGrid').innerHTML = '';
  $('bankGrid').innerHTML = '';
}

function renderBankGrid(bankItemsByBank) {
  const bankGrid = $('bankGrid');
  bankGrid.innerHTML = '';
  for (let b = 0; b < 32; b++) {
    const cell = document.createElement('div');
    cell.className = 'bank-cell';

    const items = bankItemsByBank[b] || [];
    if (items.length > 1) cell.classList.add('conflict');

    const id = document.createElement('div');
    id.className = 'bank-id';
    id.textContent = `B${b}`;
    cell.appendChild(id);

    const data = document.createElement('div');
    data.className = 'bank-data';
    items.slice(0, 10).forEach(it => {
      const di = document.createElement('div');
      di.className = 'data-item';
      di.textContent = it.label;
      di.style.background = threadToColor(it.thread, 0.9);
      data.appendChild(di);
    });

    if (items.length > 10) {
      const more = document.createElement('div');
      more.className = 'data-item';
      more.textContent = `+${items.length - 10} more`;
      more.style.background = 'rgba(255,255,255,0.15)';
      data.appendChild(more);
    }

    cell.appendChild(data);
    cell.title = items.length ? items.map(x => x.label).join('\n') : 'No accesses';
    bankGrid.appendChild(cell);
  }
}

function renderSmallGrid(container, rows, cols, cellSizePx, cellFn) {
  container.innerHTML = '';
  const grid = document.createElement('div');
  grid.className = 'grid';
  grid.style.gridTemplateColumns = `repeat(${cols}, ${cellSizePx}px)`;
  for (let r = 0; r < rows; r++) {
    for (let c = 0; c < cols; c++) {
      const cell = document.createElement('div');
      cell.className = 'cell';
      cell.style.width = `${cellSizePx}px`;
      cell.style.height = `${cellSizePx}px`;
      cell.style.fontSize = `${Math.max(5, Math.floor(cellSizePx / 2.2))}px`;
      cellFn(cell, r, c);
      grid.appendChild(cell);
    }
  }
  container.appendChild(grid);
}

function renderCTAPhase() {
  if (!state.data) return;

  const { M, N } = state.data.problem;
  const { bM, bN } = state.data.cta;
  const ctaM = Math.ceil(M / bM);
  const ctaN = Math.ceil(N / bN);

  // Update shapes
  $('gAShape').textContent = `(${M}x${state.data.problem.K})`;
  $('gBShape').textContent = `(${state.data.problem.K}x${N})`;
  $('gCShape').textContent = `(${M}x${N})`;

  // Render C matrix with CTA tiles highlighted
  const gCGrid = $('gCGrid');
  gCGrid.innerHTML = '';

  const cellsM = Math.min(M, 64); // Limit display size
  const cellsN = Math.min(N, 64);

  const grid = document.createElement('div');
  grid.className = 'grid';
  grid.style.gridTemplateColumns = `repeat(${cellsN}, 12px)`;

  for (let m = 0; m < cellsM; m++) {
    for (let n = 0; n < cellsN; n++) {
      const cell = document.createElement('div');
      cell.className = 'cell';
      cell.style.width = '12px';
      cell.style.height = '12px';
      cell.style.fontSize = '6px';

      const ctaIdxM = Math.floor(m / bM);
      const ctaIdxN = Math.floor(n / bN);
      const ctaIdx = ctaIdxM * ctaN + ctaIdxN;

      if (ctaIdx === state.stage) {
        cell.classList.add('active');
      } else {
        // Color by CTA
        cell.style.background = threadToColor(ctaIdx * 32, 0.4);
      }

      cell.title = `(${m},${n}) CTA(${ctaIdxM},${ctaIdxN})`;
      grid.appendChild(cell);
    }
  }

  gCGrid.appendChild(grid);

  // Stage info
  const ctaIdxM = Math.floor(state.stage / ctaN);
  const ctaIdxN = state.stage % ctaN;
  $('stageInfo').textContent = `CTA(${ctaIdxM},${ctaIdxN}) → C[${ctaIdxM*bM}:${(ctaIdxM+1)*bM}, ${ctaIdxN*bN}:${(ctaIdxN+1)*bN}]`;

  setInspector([
    `CTA Scheduler Phase`,
    ``,
    `Problem: M=${M}, N=${N}, K=${state.data.problem.K}`,
    `CTA Tile: ${bM} x ${bN} x ${state.data.cta.bK}`,
    `Total CTAs: ${ctaM} x ${ctaN} = ${ctaM * ctaN}`,
    ``,
    `Current CTA: (${ctaIdxM}, ${ctaIdxN})`,
    `Output tile: C[${ctaIdxM*bM}:${(ctaIdxM+1)*bM}, ${ctaIdxN*bN}:${(ctaIdxN+1)*bN}]`,
    `A slice: A[${ctaIdxM*bM}:${(ctaIdxM+1)*bM}, :]`,
    `B slice: B[:, ${ctaIdxN*bN}:${(ctaIdxN+1)*bN}]`
  ]);
}

function renderG2SPhase() {
  if (!state.data) return;

  const { bM, bN, bK } = state.data.cta;
  const thr = state.data.threads[state.thread];

  const pairsPerThr = state.cache.g2sPairsPerThread || 1;
  const pipeStages = state.data.cta?.pipe || 1;
  const pipe = Math.floor(state.stage / pairsPerThr);
  const instr = state.stage % pairsPerThr;

  const cpyK = state.data.g2s?.cpy_k || 1;
  const cpyM = state.data.g2s?.cpy_m || 1;
  const ck = instr % cpyK;
  const cm = Math.floor(instr / cpyK) % cpyM;
  const cp = Math.floor(instr / (cpyK * cpyM));

  const vecLen = state.cache.g2sVectorLen || 8;

  const stageA = new Map(); // "m,k" -> [thread]
  const stageB = new Map(); // "n,k" -> [thread]
  state.data.threads.forEach(t => {
    const baseA = t.a_g2s_mk;
    const baseB = t.b_g2s_nk;
    if (baseA && baseA.length >= 2 * (instr + 1)) {
      const m = baseA[2 * instr];
      const k0 = baseA[2 * instr + 1];
      for (let dk = 0; dk < vecLen; dk++) {
        const k = k0 + dk;
        if (k < 0 || k >= bK) continue;
        const key = `${m},${k}`;
        if (!stageA.has(key)) stageA.set(key, []);
        stageA.get(key).push(t.thread);
      }
    }
    if (baseB && baseB.length >= 2 * (instr + 1)) {
      const n = baseB[2 * instr];
      const k0 = baseB[2 * instr + 1];
      for (let dk = 0; dk < vecLen; dk++) {
        const k = k0 + dk;
        if (k < 0 || k >= bK) continue;
        const key = `${n},${k}`;
        if (!stageB.has(key)) stageB.set(key, []);
        stageB.get(key).push(t.thread);
      }
    }
  });

  const showM = Math.min(bM, 32);
  const showN = Math.min(bN, 32);
  const showK = Math.min(bK, 32);

  $('gAShape').textContent = `(${bM}x${bK}) tile`;
  $('gBShape').textContent = `(${bN}x${bK}) tile`;
  $('gCShape').textContent = `(not used in G2S)`;

  renderSmallGrid($('gAGrid'), showM, showK, 12, (cell, m, k) => {
    const key = `${m},${k}`;
    const threads = stageA.get(key) || [];
    if (threads.length) {
      cell.style.background = threadToColor(threads[0], 0.85);
      if (threads.includes(state.thread)) cell.classList.add('active');
      cell.title = `gA(${m},${k}) ← T[${threads.join(',')}]`;
    } else {
      cell.title = `gA(${m},${k})`;
    }
  });
  renderSmallGrid($('gBGrid'), showN, showK, 12, (cell, n, k) => {
    const key = `${n},${k}`;
    const threads = stageB.get(key) || [];
    if (threads.length) {
      cell.style.background = threadToColor(threads[0], 0.85);
      if (threads.includes(state.thread)) cell.classList.add('active');
      cell.title = `gB(${n},${k}) ← T[${threads.join(',')}]`;
    } else {
      cell.title = `gB(${n},${k})`;
    }
  });

  $('sAShape').textContent = `(${bM}x${bK}) pipe=${pipe}`;
  $('sBShape').textContent = `(${bN}x${bK}) pipe=${pipe}`;
  renderSmallGrid($('sAGrid'), showM, showK, 14, (cell, m, k) => {
    const key = `${m},${k}`;
    const threads = stageA.get(key) || [];
    if (threads.length) {
      cell.style.background = threadToColor(threads[0], 0.85);
      if (threads.includes(state.thread)) cell.classList.add('active');
      const smem = state.cache.smemA.get(`${m},${k},${pipe}`);
      const extra = smem ? ` off=${smem.off} bank=${smem.bank}` : '';
      cell.title = `sA(m=${m},k=${k},p=${pipe}) ← T[${threads.join(',')}]${extra}`;
    } else {
      cell.title = `sA(m=${m},k=${k},p=${pipe})`;
    }
  });
  renderSmallGrid($('sBGrid'), showN, showK, 14, (cell, n, k) => {
    const key = `${n},${k}`;
    const threads = stageB.get(key) || [];
    if (threads.length) {
      cell.style.background = threadToColor(threads[0], 0.85);
      if (threads.includes(state.thread)) cell.classList.add('active');
      const smem = state.cache.smemB.get(`${n},${k},${pipe}`);
      const extra = smem ? ` off=${smem.off} bank=${smem.bank}` : '';
      cell.title = `sB(n=${n},k=${k},p=${pipe}) ← T[${threads.join(',')}]${extra}`;
    } else {
      cell.title = `sB(n=${n},k=${k},p=${pipe})`;
    }
  });

  const bankItems = Array.from({ length: 32 }, () => []);
  stageA.forEach((threads, key) => {
    const [m, k] = key.split(',').map(Number);
    threads.forEach(tid => {
      const smem = state.cache.smemA.get(`${m},${k},${pipe}`);
      if (!smem) return;
      bankItems[smem.bank].push({ thread: tid, label: `A T${tid} (${m},${k})` });
    });
  });
  stageB.forEach((threads, key) => {
    const [n, k] = key.split(',').map(Number);
    threads.forEach(tid => {
      const smem = state.cache.smemB.get(`${n},${k},${pipe}`);
      if (!smem) return;
      bankItems[smem.bank].push({ thread: tid, label: `B T${tid} (${n},${k})` });
    });
  });
  renderBankGrid(bankItems);

  $('stageInfo').textContent = `pipe=${pipe}/${pipeStages - 1}, instr=${instr}/${pairsPerThr - 1} (cp=${cp}, cm=${cm}, ck=${ck})`;

  setInspector([
    `G2S (Global → Shared Memory) Phase`,
    ``,
    `Thread ${state.thread} (warp ${thr?.warp}, lane ${thr?.lane})`,
    ``,
    `Copy Atom: ${state.data.g2s?.atom || 'unknown'}`,
    `Bytes per instruction: ${state.data.g2s?.bytes_per_inst || 16}`,
    `Vector length: ${vecLen} elements`,
    `Pipe stage: ${pipe} (out of ${pipeStages})`,
    ``,
    `Current instr index: ${instr} (cp=${cp}, cm=${cm}, ck=${ck})`,
    `A base (m,k0): (${thr?.a_g2s_mk?.[2*instr]}, ${thr?.a_g2s_mk?.[2*instr+1]})`,
    `B base (n,k0): (${thr?.b_g2s_nk?.[2*instr]}, ${thr?.b_g2s_nk?.[2*instr+1]})`
  ]);
}

function renderS2RPhase() {
  if (!state.data) return;

  const { bM, bN, bK } = state.data.cta;
  const thr = state.data.threads[state.thread];
  const cpCount = state.data.s2r?.cp_count || 1;
  const kb = Math.floor(state.stage / cpCount);
  const cpSel = state.stage % cpCount;

  const aBlk = thr?.a_s2r?.find(x => x.k_block === kb);
  const bBlk = thr?.b_s2r?.find(x => x.k_block === kb);

  if (!aBlk || !bBlk) {
    setInspector(`No S2R data for k_block ${kb}`);
    return;
  }

  // Render Register A
  const regAGrid = $('regAGrid');
  regAGrid.innerHTML = '';

  const mmaM = aBlk.m.length;
  const cpCountBlk = aBlk.m[0]?.length || 0;

  $('regAShape').textContent = `(${mmaM} x ${cpCountBlk})`;

  const tableA = document.createElement('table');
  tableA.className = 'reg-table';

  // Header
  const headerA = document.createElement('tr');
  headerA.innerHTML = '<th></th>' + Array.from({length: cpCountBlk}, (_, i) => `<th>cp${i}</th>`).join('');
  tableA.appendChild(headerA);

  // Data rows
  for (let mm = 0; mm < mmaM; mm++) {
    const row = document.createElement('tr');
    row.innerHTML = `<th>m${mm}</th>`;
    for (let cp = 0; cp < cpCountBlk; cp++) {
      const m = aBlk.m[mm][cp];
      const kCol = aBlk.k_col[mm][cp];
      const td = document.createElement('td');
      td.className = 'val';
      if (cp === cpSel) td.classList.add('highlight');
      td.textContent = `${m},${kCol}`;
      td.title = `REG_A[${mm},${cp}] → sA(m=${m}, k_col=${kCol})`;
      row.appendChild(td);
    }
    tableA.appendChild(row);
  }

  regAGrid.appendChild(tableA);

  // Render Register B
  const regBGrid = $('regBGrid');
  regBGrid.innerHTML = '';

  const mmaN = bBlk.n.length;

  $('regBShape').textContent = `(${mmaN} x ${cpCountBlk})`;

  const tableB = document.createElement('table');
  tableB.className = 'reg-table';

  const headerB = document.createElement('tr');
  headerB.innerHTML = '<th></th>' + Array.from({length: cpCountBlk}, (_, i) => `<th>cp${i}</th>`).join('');
  tableB.appendChild(headerB);

  for (let nn = 0; nn < mmaN; nn++) {
    const row = document.createElement('tr');
    row.innerHTML = `<th>n${nn}</th>`;
    for (let cp = 0; cp < cpCountBlk; cp++) {
      const n = bBlk.n[nn][cp];
      const kCol = bBlk.k_col[nn][cp];
      const td = document.createElement('td');
      td.className = 'val';
      if (cp === cpSel) td.classList.add('highlight');
      td.textContent = `${n},${kCol}`;
      td.title = `REG_B[${nn},${cp}] → sB(n=${n}, k_col=${kCol})`;
      row.appendChild(td);
    }
    tableB.appendChild(row);
  }

  regBGrid.appendChild(tableB);

  // Highlight in smem views
  renderS2RSmemHighlight(aBlk, bBlk, cpSel);

  // k_block info
  const kBlock = state.data.k_blocks?.[kb];
  const kVals = kBlock?.k_vals || [];

  $('stageInfo').textContent = `k_block=${kb}/${state.data.s2r?.k_block_count - 1}, cp=${cpSel}/${cpCount - 1}, k_vals=[${kVals.slice(0, 8).join(',')}${kVals.length > 8 ? '...' : ''}]`;

  setInspector([
    `S2R (Shared → Register) Phase`,
    ``,
    `Thread ${state.thread} (warp ${thr?.warp}, lane ${thr?.lane})`,
    `k_block: ${kb}/${state.data.s2r?.k_block_count - 1}`,
    `k_block_len: ${state.data.s2r?.k_block_len}`,
    `cp_index: ${cpSel}/${cpCount - 1}`,
    ``,
    `Copy Atom: ${state.data.s2r?.atom} (ldmatrix)`,
    ``,
    `REG_A shape: (mma_m=${mmaM}, cp=${cpCountBlk})`,
    `REG_B shape: (mma_n=${mmaN}, cp=${cpCountBlk})`,
    ``,
    `Sample A[0,0] = sA(m=${aBlk.m[0]?.[0]}, k_col=${aBlk.k_col[0]?.[0]})`,
    `Sample B[0,0] = sB(n=${bBlk.n[0]?.[0]}, k_col=${bBlk.k_col[0]?.[0]})`
  ]);
}

function renderS2RSmemHighlight(aBlk, bBlk, cpSel) {
  const { bM, bN, bK } = state.data.cta;

  // Treat "stage" as one ldmatrix column (cpSel)
  const aAccess = [];
  const bAccess = [];
  for (let mm = 0; mm < aBlk.m.length; mm++) {
    aAccess.push({ m: aBlk.m[mm][cpSel], kCol: aBlk.k_col[mm][cpSel] });
  }
  for (let nn = 0; nn < bBlk.n.length; nn++) {
    bAccess.push({ n: bBlk.n[nn][cpSel], kCol: bBlk.k_col[nn][cpSel] });
  }

  // Render sA with highlights
  const sAGrid = $('sAGrid');
  sAGrid.innerHTML = '';

  const showM = Math.min(bM, 32);
  const showK = Math.min(state.data.s2r?.k_block_len || 16, 16);

  const gridA = document.createElement('div');
  gridA.className = 'grid';
  gridA.style.gridTemplateColumns = `repeat(${showK}, 14px)`;

  for (let m = 0; m < showM; m++) {
    for (let k = 0; k < showK; k++) {
      const cell = document.createElement('div');
      cell.className = 'cell';
      cell.style.width = '14px';
      cell.style.height = '14px';
      cell.style.fontSize = '6px';

      const hit = aAccess.some(x => x.m === m && x.kCol === k);
      if (hit) {
        cell.classList.add('source');
        cell.textContent = '●';
      }

      cell.title = `sA(${m}, k_col=${k})`;
      gridA.appendChild(cell);
    }
  }

  sAGrid.appendChild(gridA);
  $('sAShape').textContent = `(${bM}x${showK}) k_block=${Math.floor(state.stage / (state.data.s2r?.cp_count || 1))} cp=${cpSel}`;

  // Render sB
  const sBGrid = $('sBGrid');
  sBGrid.innerHTML = '';

  const showN = Math.min(bN, 32);

  const gridB = document.createElement('div');
  gridB.className = 'grid';
  gridB.style.gridTemplateColumns = `repeat(${showK}, 14px)`;

  for (let n = 0; n < showN; n++) {
    for (let k = 0; k < showK; k++) {
      const cell = document.createElement('div');
      cell.className = 'cell';
      cell.style.width = '14px';
      cell.style.height = '14px';
      cell.style.fontSize = '6px';

      const hit = bAccess.some(x => x.n === n && x.kCol === k);
      if (hit) {
        cell.classList.add('source');
        cell.textContent = '●';
      }

      cell.title = `sB(${n}, k_col=${k})`;
      gridB.appendChild(cell);
    }
  }

  sBGrid.appendChild(gridB);
  $('sBShape').textContent = `(${bN}x${showK}) k_block=${Math.floor(state.stage / (state.data.s2r?.cp_count || 1))} cp=${cpSel}`;

  // Bank view (use per-k_block bank tables if available)
  const kb = Math.floor(state.stage / (state.data.s2r?.cp_count || 1));
  const bankItems = Array.from({ length: 32 }, () => []);
  const kBlock = state.data.k_blocks?.[kb];
  if (kBlock?.a_smem_banks && kBlock?.b_smem_banks) {
    aAccess.forEach(x => {
      if (x.m < 0 || x.m >= kBlock.a_smem_banks.length) return;
      if (x.kCol < 0 || x.kCol >= kBlock.a_smem_banks[x.m].length) return;
      const bank = kBlock.a_smem_banks[x.m][x.kCol];
      if (bank >= 0 && bank < 32) bankItems[bank].push({ thread: state.thread, label: `A T${state.thread} (${x.m},k${x.kCol})` });
    });
    bAccess.forEach(x => {
      if (x.n < 0 || x.n >= kBlock.b_smem_banks.length) return;
      if (x.kCol < 0 || x.kCol >= kBlock.b_smem_banks[x.n].length) return;
      const bank = kBlock.b_smem_banks[x.n][x.kCol];
      if (bank >= 0 && bank < 32) bankItems[bank].push({ thread: state.thread, label: `B T${state.thread} (${x.n},k${x.kCol})` });
    });
  }
  renderBankGrid(bankItems);
}

function renderMMAPhase() {
  if (!state.data) return;

  const { bM, bN } = state.data.cta;
  const thr = state.data.threads[state.thread];
  const mma = state.data.mma;

  // Get C fragment mapping
  const cMn = thr?.c_mn || [];
  const cV = mma?.c_v_count || 4;
  const cM = mma?.c_m_count || 4;
  const cN = mma?.c_n_count || 8;

  $('regCShape').textContent = `(${cV} x ${cM} x ${cN})`;

  // Render C fragment
  const regCGrid = $('regCGrid');
  regCGrid.innerHTML = '';

  for (let v = 0; v < cV; v++) {
    const vHeader = document.createElement('div');
    vHeader.style.cssText = 'font-size: 10px; color: #aaa; margin: 4px 0 2px;';
    vHeader.textContent = `v=${v}`;
    regCGrid.appendChild(vHeader);

    const table = document.createElement('table');
    table.className = 'reg-table';

    const header = document.createElement('tr');
    header.innerHTML = '<th></th>' + Array.from({length: cN}, (_, i) => `<th>n${i}</th>`).join('');
    table.appendChild(header);

    for (let mm = 0; mm < cM; mm++) {
      const row = document.createElement('tr');
      row.innerHTML = `<th>m${mm}</th>`;

      for (let nn = 0; nn < cN; nn++) {
        const idx = (v * cM * cN + mm * cN + nn) * 2;
        const m = cMn[idx] ?? -1;
        const n = cMn[idx + 1] ?? -1;

        const td = document.createElement('td');
        td.className = 'val';
        td.textContent = `${m},${n}`;
        td.title = `REG_C[v=${v}, m=${mm}, n=${nn}] → C(${m}, ${n})`;
        row.appendChild(td);
      }

      table.appendChild(row);
    }

    regCGrid.appendChild(table);
  }

  // Show which output elements this thread computes
  const gCGrid = $('gCGrid');
  gCGrid.innerHTML = '';

  const showM = Math.min(bM, 64);
  const showN = Math.min(bN, 64);

  // Build set of (m,n) this thread owns
  const owned = new Set();
  for (let i = 0; i < cMn.length; i += 2) {
    owned.add(`${cMn[i]},${cMn[i+1]}`);
  }

  const grid = document.createElement('div');
  grid.className = 'grid';
  grid.style.gridTemplateColumns = `repeat(${showN}, 10px)`;

  for (let m = 0; m < showM; m++) {
    for (let n = 0; n < showN; n++) {
      const cell = document.createElement('div');
      cell.className = 'cell';
      cell.style.width = '10px';
      cell.style.height = '10px';
      cell.style.fontSize = '5px';

      if (owned.has(`${m},${n}`)) {
        cell.classList.add('computing');
      }

      cell.title = `C(${m},${n})`;
      grid.appendChild(cell);
    }
  }

  gCGrid.appendChild(grid);
  $('gCShape').textContent = `(${bM}x${bN}) output tile`;

  // k_block info
  const kb = state.stage;
  const kBlock = state.data.k_blocks?.[kb];

  $('stageInfo').textContent = `MMA k_block=${kb}`;

  setInspector([
    `MMA (Matrix Multiply-Accumulate) Phase`,
    ``,
    `Thread ${state.thread} (warp ${thr?.warp}, lane ${thr?.lane})`,
    `k_block: ${kb}/${state.data.pipeline?.k_block_count - 1}`,
    ``,
    `MMA Operation: ${mma?.op}`,
    `C Fragment Shape: (v=${cV}, mma_m=${cM}, mma_n=${cN})`,
    `Total elements per thread: ${cV * cM * cN}`,
    ``,
    `This thread computes C elements:`,
    `First few: ${cMn.slice(0, 12).reduce((acc, v, i) => {
      if (i % 2 === 0) return acc + (i > 0 ? ', ' : '') + '(' + v;
      return acc + ',' + v + ')';
    }, '')}...`
  ]);
}

function renderR2SPhase() {
  if (!state.data) return;

  setInspector([
    `R2S (Register → Shared for C) Phase`,
    ``,
    `This phase involves writing accumulator results back.`,
    `For simple epilogues, results go directly to gmem.`,
    `For complex epilogues (like split-K), may use smem.`,
    ``,
    `Thread ${state.thread}`,
    `Accumulator elements: ${state.data.mma?.c_v_count * state.data.mma?.c_m_count * state.data.mma?.c_n_count}`
  ]);

  $('stageInfo').textContent = 'Epilogue';
}

function renderTimeline() {
  const timeline = $('timeline');
  timeline.innerHTML = '';

  if (!state.data) return;

  const phases = [
    { id: 'cta', name: 'CTA', detail: 'scheduler' },
    { id: 'g2s', name: 'G2S', detail: 'gmem→smem' },
    { id: 's2r', name: 'S2R', detail: 'smem→reg' },
    { id: 'mma', name: 'MMA', detail: 'compute' },
    { id: 'r2s', name: 'R2S', detail: 'epilogue' }
  ];

  phases.forEach(p => {
    const stage = document.createElement('div');
    stage.className = `timeline-stage ${p.id}`;
    if (state.phase === p.id) stage.classList.add('active');

    stage.innerHTML = `
      <div class="stage-name">${p.name}</div>
      <div class="stage-detail">${p.detail}</div>
    `;

    stage.addEventListener('click', () => {
      state.phase = p.id;
      $('phaseSelect').value = p.id;
      updateStageRange();
      renderAll();
    });

    timeline.appendChild(stage);
  });
}

function renderAll() {
  if (!state.data) {
    setInspector('No data loaded. Click "Load Sample" or load a JSON file.');
    return;
  }

  // Update thread info
  const thr = state.data.threads[state.thread];
  $('threadInfo').textContent = thr ? `warp ${thr.warp}, lane ${thr.lane}` : '';

  clearAllGrids();

  // Render phase-specific content
  switch (state.phase) {
    case 'cta': renderCTAPhase(); break;
    case 'g2s': renderG2SPhase(); break;
    case 's2r': renderS2RPhase(); break;
    case 'mma': renderMMAPhase(); break;
    case 'r2s': renderR2SPhase(); break;
  }

  renderTimeline();
}

// ============ Playback ============

function startPlayback() {
  if (state.playing) return;
  state.playing = true;
  $('btnPlay').textContent = '⏸ Pause';
  $('btnPlay').classList.add('playing');

  state.playInterval = setInterval(() => {
    const max = parseInt($('stageSlider').max);
    if (state.stage >= max) {
      stopPlayback();
      return;
    }
    state.stage++;
    $('stageSlider').value = state.stage;
    $('stageNumber').value = state.stage;
    renderAll();
  }, state.speed);
}

function stopPlayback() {
  state.playing = false;
  $('btnPlay').textContent = '▶ Play';
  $('btnPlay').classList.remove('playing');
  if (state.playInterval) {
    clearInterval(state.playInterval);
    state.playInterval = null;
  }
}

// ============ Event Handlers ============

function setupEventHandlers() {
  // Phase selection
  $('phaseSelect').addEventListener('change', e => {
    state.phase = e.target.value;
    state.stage = 0;
    updateStageRange();
    renderAll();
  });

  // Stage controls
  $('stageSlider').addEventListener('input', e => {
    state.stage = parseInt(e.target.value);
    $('stageNumber').value = state.stage;
    renderAll();
  });

  $('stageNumber').addEventListener('change', e => {
    const max = parseInt($('stageSlider').max);
    state.stage = clamp(parseInt(e.target.value) || 0, 0, max);
    $('stageSlider').value = state.stage;
    $('stageNumber').value = state.stage;
    renderAll();
  });

  // Thread controls
  $('threadSlider').addEventListener('input', e => {
    state.thread = parseInt(e.target.value);
    $('threadNumber').value = state.thread;
    renderAll();
  });

  $('threadNumber').addEventListener('change', e => {
    const max = parseInt($('threadSlider').max);
    state.thread = clamp(parseInt(e.target.value) || 0, 0, max);
    $('threadSlider').value = state.thread;
    $('threadNumber').value = state.thread;
    renderAll();
  });

  // Playback controls
  $('btnPlay').addEventListener('click', () => {
    if (state.playing) {
      stopPlayback();
    } else {
      startPlayback();
    }
  });

  $('btnPrev').addEventListener('click', () => {
    stopPlayback();
    if (state.stage > 0) {
      state.stage--;
      $('stageSlider').value = state.stage;
      $('stageNumber').value = state.stage;
      renderAll();
    }
  });

  $('btnNext').addEventListener('click', () => {
    stopPlayback();
    const max = parseInt($('stageSlider').max);
    if (state.stage < max) {
      state.stage++;
      $('stageSlider').value = state.stage;
      $('stageNumber').value = state.stage;
      renderAll();
    }
  });

  $('speedSlider').addEventListener('input', e => {
    state.speed = parseInt(e.target.value);
  });

  // Load buttons
  $('btnLoadSample').addEventListener('click', async () => {
    setInspector('Loading gemm_data.json...');
    try {
      const resp = await fetch('gemm_data.json', { cache: 'no-cache' });
      if (!resp.ok) throw new Error(`HTTP ${resp.status}`);
      const data = await resp.json();
      loadData(data);
      setInspector([
        `Loaded gemm_data.json successfully!`,
        ``,
        `Problem: M=${data.problem.M}, N=${data.problem.N}, K=${data.problem.K}`,
        `CTA Tile: ${data.cta.bM} x ${data.cta.bN} x ${data.cta.bK}`,
        `Threads: ${data.threads.length}`,
        ``,
        `Use Phase selector to switch between stages.`,
        `Use Stage slider to step through iterations.`
      ]);
    } catch (err) {
      console.log('Sample not found, using mock data:', err);
      setInspector(`Failed to load gemm_data.json: ${err.message}\nUsing mock data instead.`);
      loadData(buildMockData());
    }
  });

  $('fileInput').addEventListener('change', async e => {
    const file = e.target.files?.[0];
    if (!file) return;
    try {
      const text = await file.text();
      loadData(JSON.parse(text));
    } catch (err) {
      setInspector(`Error loading file: ${err.message}`);
    }
  });
}

// ============ Initialize ============

function init() {
  setupEventHandlers();

  setInspector('Initializing... Loading gemm_data.json');

  // Try to load sample, fall back to mock
  fetch('gemm_data.json', { cache: 'no-cache' })
    .then(resp => {
      if (!resp.ok) throw new Error(`HTTP ${resp.status}`);
      return resp.json();
    })
    .then(data => {
      loadData(data);
      console.log('Loaded gemm_data.json:', data.problem);
      setInspector([
        `Loaded gemm_data.json`,
        ``,
        `Problem: M=${data.problem.M}, N=${data.problem.N}, K=${data.problem.K}`,
        `CTA Tile: ${data.cta.bM} x ${data.cta.bN} x ${data.cta.bK}`,
        `CTAs: ${Math.ceil(data.problem.M/data.cta.bM)} x ${Math.ceil(data.problem.N/data.cta.bN)}`,
        ``,
        `Current Phase: ${state.phase.toUpperCase()}`,
        `Max Stage: ${$('stageSlider').max}`,
        ``,
        `Tip: Switch to G2S/S2R/MMA phase for more stages.`
      ]);
    })
    .catch(err => {
      console.log('Using mock data:', err);
      loadData(buildMockData());
      setInspector([
        `Could not load gemm_data.json (${err.message})`,
        `Using mock data instead.`,
        ``,
        `To generate real data:`,
        `  ./sgemm_sm80 --dump-json > gemm_data.json`,
        ``,
        `Serve with: python3 -m http.server 8000`
      ]);
    });
}

init();
