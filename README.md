# vLLM Manager — Cross-Platform Desktop App

App desktop (Python + tkinter) per gestire server vLLM su qualsiasi piattaforma.

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
python vllm_manager.py
```

La piattaforma viene rilevata automaticamente all'avvio.

## Compilare standalone

| Piattaforma | Script | Output |
|---|---|---|
| Windows | `build_exe.bat` | `dist\vLLM Manager.exe` |
| Linux | `./build_linux.sh` | `dist/vllm-manager` (+ AppImage opzionale) |
| macOS | `./build_macos.sh` | `dist/vLLM Manager.app` |
| DGX Spark | `./build_dgx.sh` | `dist/vllm-manager-dgx` |

---

## Funzionalita (7+1 tab)

| Tab | Cosa fa |
|-----|---------|
| **Server** | Start/Stop server vLLM, scelta modello (preset per piattaforma + custom HF), parametri (gpu-memory-utilization, max-model-len, dtype, prefix-caching, chat template, attention backend, extra args), salvataggio profili rapido |
| **GPU Monitor** | VRAM/Memoria, utilizzo GPU, temperatura, potenza in tempo reale con grafico storico 2 minuti |
| **Logs** | Log server in tempo reale, colorazione errori/warning, ricerca (Ctrl+F / Cmd+F) |
| **Models** | Ricerca modelli su HuggingFace, lista modelli scaricati localmente con dimensioni, eliminazione per liberare spazio |
| **Benchmark** | Test performance con prompt preset o custom, misura tokens/sec e TTFT (Time to First Token) |
| **WebUI** | Start/Stop container Open WebUI (Docker), apri browser, log Docker |
| **Profili** | Salva/carica/elimina configurazioni server, esporta/importa JSON |
| **DGX Spark** | *(solo su DGX)* Cluster overview, node discovery, Ray cluster, multi-node inference, installazione vLLM, SSH config |

---

## Differenze per piattaforma

### macOS Apple Silicon
- GPU Memory Utilization nascosto (non applicabile)
- "VRAM" rinominato in "Memoria" (memoria unificata)
- Scelta backend MLX: auto / vllm-metal / vllm-mlx
- Preset modelli MLX quantizzati (mlx-community)
- Temperatura e potenza GPU: N/A

### DGX Spark
- Tab 8 dedicato con gestione completa cluster
- Preset modelli da 70B+ (128 GB per nodo)
- Attention backend configurabile (TRITON_ATTN / FLASHINFER / XFORMERS)
- Variabili ambiente NCCL/UCX/GLOO editabili
- Ray cluster: avvio head, connessione workers, dashboard
- Multi-node inference con tensor-parallel e pipeline-parallel
- Installazione vLLM da source (build sm_121) o via Docker NVIDIA
- Gestione SSH per nodi remoti

### Windows WSL2
- Comandi eseguiti via `wsl -d Ubuntu-22.04 -- bash`
- "Stoppa Tutto" termina anche WSL
- Docker via `docker.exe` (Docker Desktop)

### Linux nativo
- Comandi eseguiti via `bash` diretto
- Docker nativo

---

## Uso manuale (senza app)

### Avviare il server
```bash
# Attiva il virtualenv (se presente)
source ~/vllm-env/bin/activate

vllm serve Qwen/Qwen2.5-7B-Instruct \
  --host 0.0.0.0 \
  --port 8000 \
  --gpu-memory-utilization 0.90 \
  --max-model-len 4096 \
  --dtype auto \
  --enable-prefix-caching
```

Server accessibile su `http://localhost:8000`

### Modelli consigliati

#### NVIDIA 16-24 GB VRAM (Windows/Linux)

| Modello | VRAM | Note |
|---------|------|------|
| Qwen/Qwen2.5-1.5B-Instruct | ~3 GB | Test veloce |
| Qwen/Qwen2.5-7B-Instruct | ~14 GB | Buon bilanciamento |
| meta-llama/Llama-3.1-8B-Instruct | ~16 GB | Al limite, ridurre max-model-len |
| TheBloke/Mistral-7B-Instruct-v0.2-AWQ | ~4 GB | Quantizzato AWQ, veloce |
| TheBloke/Llama-2-13B-GPTQ | ~8 GB | 13B quantizzato |

#### macOS Apple Silicon

| Modello | Memoria | Note |
|---------|---------|------|
| Qwen/Qwen2.5-1.5B-Instruct | ~3 GB | Test veloce |
| meta-llama/Llama-3.2-3B-Instruct | ~6 GB | Compatto |
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

### Test via Python
```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8000/v1", api_key="not-needed")
response = client.chat.completions.create(
    model="Qwen/Qwen2.5-7B-Instruct",
    messages=[{"role": "user", "content": "Ciao!"}],
    temperature=0.7,
    max_tokens=200,
)
print(response.choices[0].message.content)
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
Poi apri `http://localhost:3000`

---

## Config path

| Piattaforma | Path |
|---|---|
| Windows | `%APPDATA%\vLLM Manager\config.json` |
| macOS | `~/Library/Application Support/vLLM Manager/config.json` |
| Linux / DGX | `~/.config/vllm-manager/config.json` |

---

## Architettura tecnica

- **Single file**: `vllm_manager.py` (Python + tkinter, solo stdlib)
- **Cross-platform**: PlatformInfo (auto-detect), CommandRunner (WSL/Native/DGX), GpuMonitor (Nvidia/AppleSilicon), VLLMDetector (auto-find backend)
- **Backend**: VLLMProcess (ciclo vita server), VLLMApi (client HTTP urllib), ConfigManager (profili JSON)
- **Threading**: operazioni lunghe in thread daemon, comunicazione thread->UI via `queue.Queue` + `root.after()`
- **Polling**: GPU ogni 2s, log ogni 250ms, status server ogni 5s

---

## Chat Template

I modelli recenti (Qwen, Llama-3) hanno il chat template nel tokenizer. I modelli quantizzati vecchi (TheBloke) richiedono un template esplicito. L'app lo gestisce automaticamente:
- `(auto)` — il modello definisce il proprio
- `llama-2` — per modelli Llama 2
- `mistral` — per modelli Mistral
- `chatml` — formato ChatML generico
