# Guida vLLM - RTX 5070 Ti (16GB VRAM)

## Setup

- **GPU**: NVIDIA GeForce RTX 5070 Ti (16 GB VRAM)
- **WSL**: Ubuntu-22.04, CUDA 12.8
- **vLLM**: v0.18.1 installato in `/root/vllm-env`
- **Docker**: Docker Desktop su Windows (non dentro WSL)
- **Open WebUI**: container Docker `open-webui`

---

## 1. vLLM Manager (App Desktop)

### Avvio
```
python C:\Users\dario\Documents\llm\vllm_manager.py
```

### Compilare in .exe
```
cd C:\Users\dario\Documents\llm
build_exe.bat
```
Output: `dist\vLLM Manager.exe` (~15-25 MB, standalone)

### Funzionalita (7 tab)

| Tab | Cosa fa |
|-----|---------|
| **Server** | Start/Stop server vLLM, scelta modello (5 preset + custom HF), parametri (gpu-memory-utilization, max-model-len, dtype, prefix-caching, chat template, extra args), salvataggio profili rapido |
| **GPU Monitor** | VRAM, utilizzo GPU, temperatura, potenza in tempo reale con grafico storico 2 minuti |
| **Logs** | Log server in tempo reale, colorazione errori/warning, ricerca Ctrl+F |
| **Models** | Ricerca modelli su HuggingFace, lista modelli scaricati localmente con dimensioni, eliminazione per liberare spazio |
| **Benchmark** | Test performance con prompt preset o custom, misura tokens/sec e TTFT (Time to First Token) |
| **WebUI** | Start/Stop container Open WebUI (Docker Desktop), apri browser, log Docker |
| **Profili** | Salva/carica/elimina configurazioni server, esporta/importa JSON |

### Bottoni principali
- **Start Server**: Verifica che WSL sia attivo (lo riavvia se spento), attiva il virtualenv `/root/vllm-env`, lancia vLLM
- **Stop Server**: Killa il processo vLLM dentro WSL con `pkill -9`
- **Stoppa Tutto (+ WSL)**: Killa vLLM + ferma container Docker Open WebUI (NON spegne WSL per non rompere Docker Desktop)

### Chat Template
I modelli recenti (Qwen, Llama-3) hanno il chat template nel tokenizer. I modelli quantizzati vecchi (TheBloke) richiedono un template esplicito. L'app lo gestisce automaticamente:
- `(auto)` — il modello definisce il proprio (Qwen, Llama-3)
- `llama-2` — per modelli Llama 2 (TheBloke/Llama-2-*-GPTQ)
- `mistral` — per modelli Mistral (TheBloke/Mistral-*-AWQ)
- `chatml` — formato ChatML generico

### Architettura tecnica
- **Single file**: `vllm_manager.py` (Python + tkinter, solo stdlib)
- **Backend**: WSLBridge (comandi WSL via subprocess), VLLMProcess (ciclo vita server), VLLMApi (client HTTP urllib), ConfigManager (profili in `%APPDATA%\vLLM Manager\config.json`), GpuMonitor (polling nvidia-smi)
- **Docker**: usa `docker.exe` direttamente da Windows (Docker Desktop), non via WSL
- **Threading**: operazioni lunghe in thread daemon, comunicazione thread->UI via `queue.Queue` + `root.after()`
- **Polling**: GPU ogni 2s, log ogni 250ms, status server ogni 5s

---

## 2. Uso manuale (senza app)

### Avviare il server
```bash
# In WSL Ubuntu-22.04
source ~/vllm-env/bin/activate
vllm serve Qwen/Qwen2.5-7B-Instruct \
  --host 0.0.0.0 \
  --port 8000 \
  --gpu-memory-utilization 0.90 \
  --max-model-len 4096 \
  --dtype auto \
  --enable-prefix-caching
```
Server accessibile su http://localhost:8000

### Modelli consigliati per 16GB VRAM

| Modello | VRAM | Note |
|---------|------|------|
| Qwen/Qwen2.5-1.5B-Instruct | ~3 GB | Test veloce |
| Qwen/Qwen2.5-7B-Instruct | ~14 GB | Buon bilanciamento |
| meta-llama/Llama-3.1-8B-Instruct | ~16 GB | Al limite, ridurre max-model-len |
| TheBloke/Mistral-7B-Instruct-v0.2-AWQ | ~4 GB | Quantizzato AWQ, veloce |
| TheBloke/Llama-2-13B-GPTQ | ~8 GB | 13B quantizzato, richiede --chat-template |

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

### Open WebUI (Docker Desktop)
```bash
docker run -d -p 3000:8080 \
  --add-host=host.docker.internal:host-gateway \
  -e OPENAI_API_BASE_URL=http://host.docker.internal:8000/v1 \
  -e OPENAI_API_KEY=not-needed \
  --name open-webui \
  --restart always \
  ghcr.io/open-webui/open-webui:main
```
Poi apri http://localhost:3000

---

## 3. Path importanti

| Cosa | Path |
|------|------|
| App | `C:\Users\dario\Documents\llm\vllm_manager.py` |
| Build script | `C:\Users\dario\Documents\llm\build_exe.bat` |
| Config profili | `%APPDATA%\vLLM Manager\config.json` |
| vLLM virtualenv | `/root/vllm-env` (dentro WSL) |
| Cache modelli HF | `~/.cache/huggingface/hub/` (dentro WSL) |
| CUDA | `/usr/local/cuda-12.8/` (dentro WSL) |

---

## 4. Punti da verificare / migliorare

### Da verificare
- [ ] **Benchmark accuracy**: il conteggio tokens via streaming SSE conta i chunk, non i token reali. Potrebbe non essere preciso al 100% — verificare confrontando con l'output non-streaming
- [ ] **Chat template modelli custom**: se si inserisce un modello HuggingFace custom che non ha chat template nel tokenizer, bisogna selezionare manualmente il template giusto — l'auto-detection funziona solo per i 5 preset
- [ ] **GPU power draw**: la RTX 5070 Ti su WSL restituisce `[N/A]` per il power draw da nvidia-smi — il valore mostrato e' sempre 0.0 W
- [ ] **Docker Desktop assente**: se Docker Desktop non e' installato o non e' avviato, il tab WebUI fallisce silenziosamente — aggiungere un check esplicito
- [ ] **Primo avvio modello pesante**: scaricare un modello da 15 GB puo' richiedere molto tempo — non c'e' indicazione di progresso del download (i log mostrano l'output di HF ma senza barra progresso)
- [ ] **Port conflict**: se la porta 8000 e' gia' occupata (altro vLLM, altro servizio), il server fallisce — aggiungere check porta prima dello start

### Miglioramenti possibili
- [ ] **Barra progresso download modelli**: intercettare l'output HF per mostrare % download nella UI
- [ ] **Multi-GPU**: supporto per sistemi con piu' GPU (tensor-parallel, --tp flag)
- [ ] **Preset modelli aggiornabili**: caricare la lista preset da un file JSON esterno invece che hardcoded
- [ ] **Tema scuro completo**: il tema `clam` di ttk e' chiaro — implementare un tema scuro consistente
- [ ] **Notifiche**: notifica Windows (toast) quando il server e' pronto o quando crasha
- [ ] **Auto-start**: opzione per avviare il server automaticamente all'apertura dell'app
- [ ] **History benchmark**: salvare i risultati benchmark in un file per confronti nel tempo
- [ ] **Log export**: bottone per esportare i log in un file .txt
- [ ] **Aggiornamento vLLM**: bottone per aggiornare vLLM (`pip install -U vllm`) dal manager
- [ ] **System tray**: minimizzare nella system tray invece di chiudere
- [ ] **Quantizzazione on-the-fly**: supporto per flag come `--quantization awq/gptq` per modelli non pre-quantizzati
- [ ] **Open WebUI env vars**: configurare `OPENAI_API_BASE_URL` nel container Docker dall'app
