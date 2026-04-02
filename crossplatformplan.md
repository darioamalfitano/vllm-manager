# vLLM Manager — Cross-Platform + DGX Spark Cluster Manager

## Contesto
L'app attuale (`vllm_manager.py`, ~800 righe) funziona SOLO su Windows via WSL2. L'obiettivo e':
1. Renderla cross-platform (Windows WSL2, Linux nativo, macOS Apple Silicon, DGX Spark)
2. Aggiungere gestione COMPLETA del DGX Spark: multi-node, Ray cluster, NCCL, networking

| Piattaforma | GPU | Memoria | vLLM Backend | OS |
|---|---|---|---|---|
| **Windows WSL2** | RTX 5070 Ti | 16 GB VRAM | CUDA | Win11 + WSL Ubuntu |
| **Linux nativo** | NVIDIA qualsiasi | variabile | CUDA | Ubuntu/Debian |
| **macOS** | Apple Silicon | 16-64 GB unificata | MLX (vllm-metal + vllm-mlx) | macOS 14+ |
| **DGX Spark** | GB10 Blackwell sm_121 | 128 GB unificata per nodo | CUDA (build custom) | DGX OS (Ubuntu 24.04, ARM64) |

---

## Architettura Cross-Platform

### 1. PlatformInfo (NUOVO, ~70 righe)

Auto-detect a startup:
- `os`: `"windows-wsl"` | `"linux"` | `"macos"`
- `arch`: `"x86_64"` | `"aarch64"`
- `gpu_type`: `"nvidia"` | `"apple-silicon"` | `"none"`
- `is_dgx`: bool (Linux ARM64 + `/etc/dgx-release` o compute cap 12.x)
- `vram_total`: MB rilevato

### 2. CommandRunner (SOSTITUISCE WSLBridge, ~200 righe)

Base astratta: `run()`, `popen()`, `run_raw()`, `shutdown()`, `is_available()`

- **WSLCommandRunner** — comandi via `wsl -d Ubuntu-22.04 -- bash` (Windows)
- **NativeCommandRunner** — `bash -c` diretto (Linux/macOS)
- **DGXCommandRunner** — estende Native, aggiunge env vars NCCL/Blackwell

### 3. GpuMonitor (REFACTOR, ~180 righe)

- **NvidiaGpuMonitor** — nvidia-smi (Windows WSL, Linux, DGX)
- **AppleSiliconGpuMonitor** — `vm_stat` + `sysctl hw.memsize` (temp/power = N/A)

### 4. VLLMDetector (NUOVO, ~100 righe)

Auto-trova installazione vLLM per piattaforma. Su macOS: supporta ENTRAMBI vllm-metal e vllm-mlx con auto-detect e scelta utente.

### 5. Preset modelli dinamici

`get_preset_models(platform_info)` con modelli diversi per:
- **DGX 1 nodo** (128 GB): Llama-3.3-70B, QwQ-32B, DeepSeek-R1-32B, Mixtral-8x22B
- **DGX multi-nodo**: Llama-3.1-405B (4 nodi), modelli 200B+ (2 nodi)
- **macOS**: Qwen2.5-1.5B/7B, Llama-3.2-3B, mlx-community quantizzati
- **Windows/Linux 16-24 GB**: preset attuali

### 6. Config path per piattaforma
- Windows: `%APPDATA%\vLLM Manager\`
- macOS: `~/Library/Application Support/vLLM Manager/`
- Linux/DGX: `~/.config/vllm-manager/`

### 7. Docker per piattaforma
- Windows: `docker.exe` (Docker Desktop)
- Linux/macOS/DGX: `docker` nativo

### 8. VLLMProcess.start() platform-aware
- **CUDA**: `python -m vllm.entrypoints.openai.api_server ...`
- **MLX-Metal**: `python -m vllm_metal.server ...`
- **MLX**: `python -m vllm_mlx.server ...`
- **DGX**: CUDA + env vars (`VLLM_ATTENTION_BACKEND`, `NCCL_SOCKET_IFNAME`, etc.)
- **DGX Multi-node**: avvia Ray head/workers poi vllm con `--tensor-parallel-size N`

---

## Tab 8: DGX Spark Manager (NUOVO)

Tab visibile SOLO quando `platform_info.is_dgx == True`. Gestisce l'intero cluster.

### 8a. Sezione "Cluster Overview"

Mostra stato del cluster in tempo reale:
- Lista nodi rilevati con hostname, IP, stato (online/offline)
- Per ogni nodo: VRAM usata/totale, temperatura, utilizzo GPU
- Topologia: singolo / 2 nodi (back-to-back) / 3 nodi (ring) / 4 nodi (switch)
- Status Ray: head node attivo?, workers connessi?, risorse totali

### 8b. Sezione "Node Discovery & Network"

**Bottone "Rileva Nodi":**
- Scansiona rete via ConnectX-7 per trovare altri DGX Spark
- Mostra: hostname, IP, interfacce di rete (enP2p1s0f1np1, etc.)
- Indica se raggiungibile via SSH

**Configurazione rete:**
- Selezione interfaccia QSFP per NCCL (`NCCL_SOCKET_IFNAME`)
- Selezione HCA per RDMA (`NCCL_IB_HCA=mlx5_0,mlx5_1`)
- Test connettivita' (ping + iperf tra nodi)
- Validazione NCCL con test all-reduce

**Env vars esposte con editor:**
```
NCCL_SOCKET_IFNAME=enP2p1s0f1np1
UCX_NET_DEVICES=enP2p1s0f1np1
GLOO_SOCKET_IFNAME=enp1s0f1np1
NCCL_IB_HCA=mlx5_0,mlx5_1
```

### 8c. Sezione "Ray Cluster"

**Bottoni:**
- "Avvia Ray Head" → `ray start --head --port=6379 --dashboard-host=0.0.0.0`
- "Connetti Worker" → SSH su nodo remoto + `ray start --address=<head-ip>:6379`
- "Disconnetti Worker" → `ray stop` sul nodo remoto
- "Stop Ray" → `ray stop` su tutti i nodi
- "Apri Ray Dashboard" → browser su `http://<head-ip>:8265`

**Status:**
- Ray dashboard link
- Risorse cluster: N nodi, N GPU, N GB totali
- Jobs attivi

### 8d. Sezione "Multi-Node Inference"

**Configurazione:**
- `--tensor-parallel-size`: slider (1 = singolo nodo, 2-4 = multi-nodo)
- `--pipeline-parallel-size`: slider
- Stima VRAM per modello (mostra se entra nel cluster)
- Attention backend: dropdown (TRITON_ATTN / FLASHINFER / XFORMERS)

**Preset multi-nodo:**
| Modello | Nodi | TP size | VRAM tot |
|---|---|---|---|
| meta-llama/Llama-3.3-70B-Instruct | 1 | 1 | ~70 GB |
| meta-llama/Llama-3.1-405B-Instruct-FP8 | 4 | 4 | ~400 GB |
| Qwen/Qwen2.5-72B-Instruct | 1 | 1 | ~72 GB |
| mistralai/Mixtral-8x22B-v0.1 | 2 | 2 | ~180 GB |

**Bottone "Start Multi-Node Server":**
1. Verifica Ray cluster attivo con N nodi
2. Imposta env vars NCCL
3. Lancia `vllm serve --model X --tensor-parallel-size N --host 0.0.0.0 --port 8000`
4. Log in tempo reale

### 8e. Sezione "vLLM Installation" (DGX auto-install)

**Se vLLM non rilevato, mostra:**
- Bottone "Installa vLLM (build from source)"
  1. Crea venv
  2. Installa PyTorch nightly ARM64 con sm_121
  3. Clona vLLM, configura `TORCH_CUDA_ARCH_LIST="12.1"`
  4. Compila (`pip install -e .`)
  5. Log progresso in tempo reale
- Bottone "Installa vLLM (Docker NVIDIA)"
  1. Pull `nvcr.io/nvidia/vllm:latest` o community image `eugr/spark-vllm`
  2. Configura container con GPU access
- Status: versione installata, backend, compute capability

### 8f. Sezione "SSH Configuration"

Per gestire i nodi remoti:
- Lista chiavi SSH configurate
- Bottone "Setup SSH" → `ssh-keygen` + `ssh-copy-id` verso nodi
- Test connessione per ogni nodo
- Campo utente remoto (default: root)

---

## UI Condizionale (tutte le piattaforme)

- **Status bar**: mostra piattaforma rilevata ("Windows WSL2 | RTX 5070 Ti | 16 GB")
- **macOS**: nasconde gpu-memory-utilization, rinomina "VRAM" → "Memoria", aggiunge scelta backend MLX
- **DGX singolo**: come Linux ma con opzione attention backend + modelli grandi
- **DGX cluster**: Tab 8 completo con multi-node
- **Windows**: bottone "Stoppa Tutto (+ Docker)"

---

## File da creare/aggiornare

| File | Azione |
|---|---|
| `vllm_manager.py` | Refactor cross-platform + Tab DGX (~2000-2200 righe) |
| `build_exe.bat` | Invariato (Windows) |
| `build_linux.sh` | NUOVO — PyInstaller + AppImage |
| `build_macos.sh` | NUOVO — PyInstaller + .app bundle |
| `build_dgx.sh` | NUOVO — PyInstaller ARM64 |
| `vllm_guide.md` | Aggiornare con tutte le piattaforme + DGX cluster |

---

## Ordine di implementazione

1. **PlatformInfo** — detection a startup
2. **CommandRunner** — WSL/Native/DGX
3. **GpuMonitor** — Nvidia/AppleSilicon
4. **VLLMDetector** — auto-find venv + backend
5. **Preset modelli dinamici**
6. **Config path + Docker** — per piattaforma
7. **VLLMProcess.start()** — CUDA/MLX/DGX-aware
8. **UI condizionale** — show/hide per piattaforma
9. **Tab 8: DGX Spark Manager** — cluster overview, node discovery, Ray, multi-node, auto-install, SSH
10. **Build scripts** — linux.sh, macos.sh, dgx.sh
11. **Aggiorna vllm_guide.md**

---

## Verifica

1. `python vllm_manager.py` su Windows — WSL2 auto-detected, 7 tab come prima
2. Su Linux nativo — NVIDIA detected, bash diretto, niente WSL
3. Su macOS — Apple Silicon, preset MLX, "Memoria" al posto di "VRAM"
4. Su DGX singolo — Tab 8 visibile, modelli 70B+, attention backend
5. Su DGX cluster (2+ nodi) — Node discovery, Ray start/stop, multi-node inference
6. Compilare .exe / .app / AppImage e verificare standalone
