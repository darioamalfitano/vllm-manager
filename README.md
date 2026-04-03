# vLLM Manager — Web App

App web (Python + Flask) per gestire server vLLM su qualsiasi piattaforma, accessibile da browser remoto.

## Piattaforme supportate

| Piattaforma | GPU | Backend vLLM | OS |
|---|---|---|---|
| **Windows WSL2** | NVIDIA (CUDA) | CUDA | Windows 11 + WSL Ubuntu |
| **Linux nativo** | NVIDIA (CUDA) | CUDA | Ubuntu/Debian |
| **macOS** | Apple Silicon | MLX (vllm-metal / vllm-mlx) | macOS 14+ |
| **DGX Spark** | GB10 Blackwell | CUDA (build custom sm_121) | DGX OS (Ubuntu 24.04, ARM64) |

---

## Avvio

```bash
pip install flask
python vllm_manager_web.py --host 0.0.0.0 --port 5000
```

Poi apri `http://<ip-server>:5000` nel browser.

La piattaforma viene rilevata automaticamente all'avvio. Se vLLM non e' installato, il pulsante **Installa vLLM** nel tab Server esegue l'installazione completa per la piattaforma rilevata.

---

## Installazione automatica vLLM

Il manager gestisce autonomamente l'installazione di vLLM in base alla piattaforma:

### Linux / Windows WSL2
- Crea virtualenv `~/vllm-env`
- Installa vLLM via `pip install vllm`

### macOS Apple Silicon
- Crea virtualenv `~/vllm-env`
- Installa `vllm-mlx` via pip

### DGX Spark (Blackwell GB10 / sm_121)

Installazione completa in 9 step (~20-30 minuti), basata su [dgx-spark-vllm-setup](https://github.com/eelbaz/dgx-spark-vllm-setup):

1. Controlli pre-installazione (GPU, CUDA, spazio disco)
2. Installazione `uv` package manager
3. Creazione virtualenv Python 3.12 in `~/vllm-install/.vllm`
4. PyTorch 2.9.0+cu130 (CUDA 13.0 per Blackwell)
5. Triton da source (branch main, supporto sm_121a)
6. Dipendenze (xgrammar, setuptools-scm, apache-tvm-ffi)
7. Clone vLLM (commit testato)
8. Patch critiche:
   - `pyproject.toml` license field
   - `CMakeLists.txt` — aggiunta SM100/SM120/SM121 ai kernel CUTLASS
   - flashinfer-python license fix
   - `use_existing_torch.py`
9. Build vLLM con `TORCH_CUDA_ARCH_LIST=12.1a`

Variabili ambiente settate automaticamente all'avvio server:
- `TORCH_CUDA_ARCH_LIST=12.1a`
- `VLLM_USE_FLASHINFER_MXFP4_MOE=1`
- `TRITON_PTXAS_PATH=/usr/local/cuda/bin/ptxas`

---

## Funzionalita (7+1 tab)

| Tab | Cosa fa |
|-----|---------|
| **Server** | Start/Stop server vLLM, scelta modello (preset per piattaforma + custom HF), parametri (gpu-memory-utilization, max-model-len, dtype, prefix-caching, chat template, attention backend, extra args), salvataggio profili, installazione vLLM |
| **GPU Monitor** | VRAM/Memoria, utilizzo GPU, temperatura, potenza in tempo reale con grafico storico 2 minuti |
| **Logs** | Log server in tempo reale (SSE), colorazione errori/warning, ricerca |
| **Models** | Ricerca modelli su HuggingFace, lista modelli scaricati localmente con dimensioni, eliminazione per liberare spazio |
| **Benchmark** | Test performance con prompt preset o custom, misura tokens/sec e TTFT (Time to First Token) |
| **WebUI** | Start/Stop container Open WebUI (Docker) |
| **Profili** | Salva/carica/elimina configurazioni server |
| **DGX Spark** | *(solo su DGX)* Cluster overview, node discovery, variabili NCCL, Ray cluster, multi-node inference, SSH config |

---

## Differenze per piattaforma

### DGX Spark
- Tab dedicato con gestione completa cluster
- Preset modelli da 70B+ (128 GB per nodo)
- Attention backend configurabile (TRITON_ATTN / FLASHINFER / XFORMERS)
- Variabili ambiente NCCL/UCX/GLOO editabili
- Ray cluster: avvio head, connessione workers
- Multi-node inference con tensor-parallel e pipeline-parallel
- Installazione automatica vLLM con patch Blackwell sm_121
- Gestione SSH per nodi remoti

### macOS Apple Silicon
- GPU Memory Utilization nascosto (non applicabile)
- "VRAM" rinominato in "Memoria" (memoria unificata)
- Preset modelli MLX quantizzati (mlx-community)
- Temperatura e potenza GPU: N/A

### Windows WSL2
- Comandi eseguiti via `wsl -d Ubuntu-22.04 -- bash`
- Docker via `docker.exe` (Docker Desktop)

### Linux nativo
- Comandi eseguiti via `bash` diretto
- Docker nativo

---

## Uso manuale (senza app)

### Avviare il server
```bash
source ~/vllm-env/bin/activate  # o ~/vllm-install/vllm_env.sh su DGX

vllm serve Qwen/Qwen2.5-7B-Instruct \
  --host 0.0.0.0 \
  --port 8000 \
  --gpu-memory-utilization 0.90 \
  --max-model-len 4096 \
  --dtype auto \
  --enable-prefix-caching
```

### Modelli consigliati

#### NVIDIA 16-24 GB VRAM (Windows/Linux)

| Modello | VRAM | Note |
|---------|------|------|
| Qwen/Qwen2.5-1.5B-Instruct | ~3 GB | Test veloce |
| Qwen/Qwen2.5-7B-Instruct | ~14 GB | Buon bilanciamento |
| meta-llama/Llama-3.1-8B-Instruct | ~16 GB | Al limite, ridurre max-model-len |
| TheBloke/Mistral-7B-Instruct-v0.2-AWQ | ~4 GB | Quantizzato AWQ, veloce |

#### macOS Apple Silicon

| Modello | Memoria | Note |
|---------|---------|------|
| Qwen/Qwen2.5-1.5B-Instruct | ~3 GB | Test veloce |
| mlx-community/Qwen2.5-7B-Instruct-4bit | ~4 GB | MLX quantizzato |

#### DGX Spark (128 GB per nodo)

| Modello | VRAM | Nodi | TP Size |
|---------|------|------|---------|
| meta-llama/Llama-3.3-70B-Instruct | ~70 GB | 1 | 1 |
| Qwen/Qwen2.5-72B-Instruct | ~72 GB | 1 | 1 |
| mistralai/Mixtral-8x22B-v0.1 | ~180 GB | 2 | 2 |
| meta-llama/Llama-3.1-405B-Instruct-FP8 | ~400 GB | 4 | 4 |

### Test via curl
```bash
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen/Qwen2.5-7B-Instruct",
    "messages": [{"role": "user", "content": "Ciao!"}],
    "temperature": 0.7,
    "max_tokens": 200
  }'
```

### Open WebUI (Docker)
```bash
docker run -d -p 3000:8080 \
  --add-host=host.docker.internal:host-gateway \
  -e OPENAI_API_BASE_URL=http://host.docker.internal:8000/v1 \
  -e OPENAI_API_KEY=not-needed \
  --name open-webui \
  --restart always \
  ghcr.io/open-webui/open-webui:main
```

---

## Config path

| Piattaforma | Path |
|---|---|
| Windows | `%APPDATA%\vLLM Manager\config.json` |
| macOS | `~/Library/Application Support/vLLM Manager/config.json` |
| Linux / DGX | `~/.config/vllm-manager/config.json` |

---

## Architettura tecnica

- **Single file**: `vllm_manager_web.py` (Python + Flask)
- **REST API**: tutti i controlli esposti come endpoint JSON (`/api/server/*`, `/api/gpu`, `/api/models/*`, `/api/dgx/*`, etc.)
- **SSE (Server-Sent Events)**: log in tempo reale via `/api/logs/stream`
- **Frontend**: HTML/CSS/JS embedded, dark theme, responsive
- **Cross-platform**: PlatformInfo (auto-detect), CommandRunner (WSL/Native/DGX), GpuMonitor (Nvidia/AppleSilicon), VLLMDetector (auto-find backend)
- **Backend**: VLLMProcess (ciclo vita server), VLLMApi (client HTTP urllib), ConfigManager (profili JSON)

---

## Chat Template

I modelli recenti (Qwen, Llama-3) hanno il chat template nel tokenizer. I modelli quantizzati vecchi (TheBloke) richiedono un template esplicito. L'app lo gestisce automaticamente:
- `(auto)` — il modello definisce il proprio
- `llama-2` — per modelli Llama 2
- `mistral` — per modelli Mistral
- `chatml` — formato ChatML generico
