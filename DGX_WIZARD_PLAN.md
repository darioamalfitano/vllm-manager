# DGX Installation Wizard — Implementation Plan

## Context

The `vllm_manager_web.py` single-file app (3314 lines) has a DGX Spark tab with basic cluster management (workers, Ray, multi-node inference). This plan adds a **6-step guided wizard** that walks users from hardware detection through vLLM server launch, with ~40 engine parameters with tooltips, all DGX-specific.

The wizard becomes a new tab ("Wizard DGX") alongside the existing "DGX Spark" tab, visible only when `is_dgx=True`.

---

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    Wizard Flow                          │
│                                                         │
│  Step 0        Step 1       Step 2       Step 3         │
│  HW Detect ──► Install ──► Params ──► Model Select      │
│                                           │             │
│                                           ▼             │
│                              Step 4       Step 5        │
│                              Ray ──────► Verify+Launch  │
└─────────────────────────────────────────────────────────┘

Data flow:
  ConfigManager.wizard{} ◄──► /api/dgx/wizard/* ◄──► JS wizState{}
                                                        │
  VLLMProcess.start() ◄── /api/dgx/wizard/launch ◄─────┘
                            (assembles CLI args from wizState)
```

Reuses heavily:
- `_install_vllm_dgx()` (line 989) — parametrized, not duplicated
- `_install_vllm_dgx_remote()` (line 1650) — for worker install via wizard
- Worker endpoints `/api/dgx/workers/*` (line 1870+) — worker CRUD
- `/api/models/search`, `/api/models/local` (line 1251+) — model search
- `_verify_worker()` (line 1845) — worker verification
- `_ssh_cmd()` / `_ssh_cmd_b64()` (line 1618/1626) — SSH helpers
- `VLLMProcess.start()` (line 626) — extended with new params
- `DGX_MULTINODE_PRESETS` (line 518) — preset data

---

## Implementation Steps (in order)

### 1. ConfigManager — wizard state methods (~15 lines)
**File:** `vllm_manager_web.py` — after line 588 (end of `ConfigManager` class)

Add 3 methods:
```python
def get_wizard(self):
    return self.data.setdefault("wizard", {})

def save_wizard_step(self, step_key, step_data):
    wiz = self.data.setdefault("wizard", {})
    wiz[step_key] = step_data
    self.save()

def reset_wizard(self):
    self.data["wizard"] = {}
    self.save()
```

### 2. DGX_WIZARD_PARAMS_SCHEMA — engine parameter definitions (~250 lines)
**File:** `vllm_manager_web.py` — after line 523 (after `DGX_MULTINODE_PRESETS`)

Define a Python list `DGX_WIZARD_PARAMS_SCHEMA` containing 8 categories, each with `key`, `label`, `params` list. Each param: `key`, `type` (text/number/select/checkbox/range), `default`, `label`, `tooltip`, `dgx_tip`, `advanced` (bool), and `options` (for selects).

**Categories and parameter counts:**
1. **Configurazione Modello** (9): model, tokenizer, dtype, max_model_len, trust_remote_code, revision, quantization, enforce_eager, seed
2. **Memoria & Performance** (6): gpu_memory_utilization, cpu_offload_gb, kv_cache_dtype, block_size, enable_prefix_caching, calculate_kv_scales
3. **Parallelismo** (7): tensor_parallel_size, pipeline_parallel_size, data_parallel_size, distributed_executor_backend, enable_expert_parallel, max_parallel_loading_workers, nnodes
4. **Serving & Scheduling** (7): host, port, max_num_seqs, max_num_batched_tokens, enable_chunked_prefill, chat_template, scheduling_policy
5. **Attention & Backend** (3): attention_backend, disable_cascade_attn, disable_sliding_window
6. **LoRA** (3, advanced): enable_lora, max_loras, max_lora_rank
7. **Speculative Decoding** (2, advanced): speculative_model, num_speculative_tokens
8. **Osservabilita** (2, advanced): disable_log_stats, enable_metrics

Full tooltip text and DGX-specific tips as specified in the draft plan.

### 3. Parametrize `_install_vllm_dgx()` (~30 lines changed)
**File:** `vllm_manager_web.py` — line 989

Change signature to accept optional overrides:
```python
def _install_vllm_dgx(vllm_commit=None, triton_commit=None, pytorch_index=None, install_dir=None):
    vllm_commit = vllm_commit or DGX_VLLM_COMMIT
    triton_commit = triton_commit or DGX_TRITON_COMMIT
    pytorch_index = pytorch_index or DGX_PYTORCH_INDEX
    install_dir = install_dir or DGX_INSTALL_DIR_DEFAULT
```
Then replace all hardcoded `DGX_*` constants inside the function with these local vars. The existing `_install_vllm_dgx_remote()` stays as-is — the wizard's worker install reuses the existing `/api/dgx/workers/install` endpoint.

### 4. Extend `VLLMProcess.start()` (~40 lines changed)
**File:** `vllm_manager_web.py` — line 626

Add new keyword arguments:
```python
def start(self, model, gpu_mem_util=0.90, max_model_len=None,
          dtype="auto", enable_prefix_caching=True, extra_args="",
          host="0.0.0.0", port=8000, chat_template="",
          tp_size=1, attention_backend="",
          # New wizard params:
          pp_size=1, dp_size=None, max_num_seqs=None,
          max_num_batched_tokens=None, enable_chunked_prefill=False,
          kv_cache_dtype=None, block_size=None, cpu_offload_gb=None,
          trust_remote_code=False, enforce_eager=False,
          quantization=None, seed=None, tokenizer=None,
          enable_lora=False, max_loras=None, max_lora_rank=None,
          speculative_model=None, num_speculative_tokens=None,
          distributed_executor_backend=None, nnodes=1,
          scheduling_policy=None, disable_log_stats=False):
```

In the command construction block (lines 639-661), add CLI flag mappings for each new param.

### 5. API endpoints (~300 lines)
**File:** `vllm_manager_web.py` — after line 2068 (after `api_dgx_ray_stop_cluster`)

**Wizard state:**
- `GET /api/dgx/wizard/state` — returns `config.get_wizard()`
- `POST /api/dgx/wizard/state` — saves full wizard state
- `POST /api/dgx/wizard/reset` — calls `config.reset_wizard()`

**Step 0 — Hardware detection:**
- `GET /api/dgx/wizard/hw-detect` — runs nvidia-smi, nvcc, df, free, ibdev2netdev, NCCL check, ~/vllm-install/ existence. Returns structured JSON with pass/fail per check.

**Step 1 — Installation:**
- `POST /api/dgx/wizard/install-config` — saves install config to wizard state
- `POST /api/dgx/wizard/install-start` — calls parametrized `_install_vllm_dgx()` in background thread
- `GET /api/dgx/wizard/install-status` — returns install state

**Step 2 — Parameters:**
- `GET /api/dgx/wizard/params-schema` — returns `DGX_WIZARD_PARAMS_SCHEMA` as JSON
- `POST /api/dgx/wizard/params-save` — saves params to wizard state

**Step 3 — Model:**
- `POST /api/dgx/wizard/model-save` — saves selected model
- `POST /api/dgx/wizard/model-download` — triggers download in background
- Reuses existing `/api/models/search` and `/api/models/local`

**Step 4 — Ray/Network:**
- `GET /api/dgx/wizard/net-detect` — runs ibdev2netdev, suggests NCCL env vars
- `POST /api/dgx/wizard/ray-config` — saves Ray/NCCL config
- Reuses existing `/api/dgx/workers/*` endpoints

**Step 5 — Verify & Launch:**
- `POST /api/dgx/wizard/verify-all` — local checks + remote `_verify_worker()` per worker
- `POST /api/dgx/wizard/launch` — assembles all params, calls `vllm_proc.start()`

### 6. CSS — wizard styles (~80 lines)
**File:** `vllm_manager_web.py` — before line 2199 (before `</style>`)

Classes: `.wizard-stepper`, `.step-circle` (pending/active/completed/error states), `.step-line`, `.wizard-panel`, `.wizard-nav`, `.param-category`, `.param-row`, `.param-tip` + `::after` tooltip, `.hw-check-item`, `.wizard-summary`, `.advanced-toggle`

### 7. HTML — wizard tab and 6 step panels (~350 lines)
**File:** `vllm_manager_web.py`

**7a. Tab button** — line 2224, add before existing DGX button:
```html
<button class="tab-btn" onclick="switchTab('dgx-wizard')">Wizard DGX</button>
```

**7b. Wizard panel** — insert before line 2502:
- Stepper bar (6 numbered circles with labels)
- Step 0: Hardware Detection (button + results grid)
- Step 1: Installation (config fields + progress)
- Step 2: Engine Parameters (empty container, JS-rendered from schema)
- Step 3: Model Selection (presets + HF search + local)
- Step 4: Ray Cluster (network detection + NCCL vars + worker table)
- Step 5: Verify & Launch (checklist + summary + launch button)
- Navigation bar (Back / Next)

### 8. JavaScript — wizard logic (~300 lines)
**File:** `vllm_manager_web.py` — before line 3284 (before `</script>`)

Navigation: `wizInit()`, `wizShowStep(n)`, `wizNext()`, `wizBack()`, `wizGoToStep(n)`, `wizUpdateStepper()`, `wizSaveState()`

Step handlers: `wizDetectHW()`, `wizSaveInstallConfig()`, `wizStartInstall()`, `wizLoadParamsSchema()`, `wizRenderParamCategory()`, `wizToggleAdvanced()`, `wizResetCategoryDefaults()`, `wizSaveParams()`, `wizSearchModel()`, `wizSelectPreset()`, `wizDownloadModel()`, `wizSaveModel()`, `wizDetectNetwork()`, `wizSaveRayConfig()`, `wizVerifyAll()`, `wizRenderSummary()`, `wizLaunch()`

---

## Key Design Decisions

1. **Single file** — tutto in `vllm_manager_web.py`
2. **No duplicazione** — wizard parametrizza `_install_vllm_dgx()`, non lo copia
3. **Riuso endpoint esistenti** — worker CRUD, model search, Ray cluster
4. **Rendering dinamico params** — JS genera HTML dallo schema API
5. **Stato persistente** — wizard state in config.json, sopravvive ai reload
6. **Tab DGX Spark preservato** — wizard e' un tab aggiuntivo

---

## Stima dimensioni

~1350 righe aggiunte (da ~3314 a ~4660):
- Schema: ~250 | API: ~300 | CSS: ~80 | HTML: ~350 | JS: ~300 | Backend: ~70

---

## Verifica

1. `python3 vllm_manager_web.py` (con `is_dgx=True` forzato per test)
2. Verificare tab "Wizard DGX" visibile
3. Testare tutti i 6 step con navigazione avanti/indietro
4. Verificare persistenza stato al reload pagina
5. Testare reset wizard
6. Verificare lancio server con parametri completi dallo Step 5
