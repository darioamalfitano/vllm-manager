# Guida completa — vLLM Manager Web

Guida passo-passo per installare, configurare e utilizzare il vLLM Manager su qualsiasi piattaforma.

---

## 1. Requisiti di sistema

### Tutti

| Requisito | Minimo |
|-----------|--------|
| Python | 3.10+ |
| pip | Qualsiasi versione recente |
| Rete | Accesso internet (per scaricare modelli) |
| Browser | Qualsiasi browser moderno |

### Linux / Windows WSL2

| Requisito | Minimo |
|-----------|--------|
| GPU | NVIDIA con CUDA support |
| VRAM | 4 GB+ (dipende dal modello) |
| Driver NVIDIA | 535+ |
| CUDA Toolkit | 12.1+ |

### macOS

| Requisito | Minimo |
|-----------|--------|
| Chip | Apple Silicon (M1/M2/M3/M4) |
| RAM | 8 GB+ (memoria unificata) |
| macOS | 14+ (Sonoma) |

### DGX Spark (Blackwell GB10)

| Requisito | Minimo |
|-----------|--------|
| GPU | NVIDIA GB10 (sm_121) |
| Driver | 580+ |
| CUDA | 13.0 |
| Spazio disco | 50 GB liberi |
| RAM | 8 GB+ durante il build |

---

## 2. Installazione da zero

### 2.1 Scaricare il progetto

```bash
git clone https://github.com/<tuo-utente>/vllm-manager.git
cd vllm-manager
```

Oppure copia manualmente il file `vllm_manager_web.py` sulla macchina target.

### 2.2 Installare Flask

```bash
pip install flask
```

Se sei su DGX Spark e non hai pip nel sistema:

```bash
sudo apt update && sudo apt install -y python3-pip
pip install flask
```

### 2.3 Avviare il manager

```bash
python vllm_manager_web.py --host 0.0.0.0 --port 5000
```

Opzioni:
- `--host 0.0.0.0` — rende accessibile da qualsiasi IP (necessario per accesso remoto)
- `--port 5000` — porta del web server (default: 5000)
- `--debug` — modalita' debug Flask (solo per sviluppo)

### 2.4 Accedere dal browser

Apri nel browser:
```
http://<ip-della-macchina>:5000
```

Per trovare l'IP della macchina:
```bash
# Linux / DGX
hostname -I | awk '{print $1}'

# macOS
ipconfig getifaddr en0
```

---

## 3. Prima configurazione

### 3.1 Installare vLLM (automatico)

Al primo avvio, se vLLM non e' rilevato, clicca **"Installa vLLM"** nel tab Server. Il manager gestisce tutto automaticamente in base alla piattaforma rilevata.

#### Cosa succede in background:

**Linux / WSL2:**
1. Crea virtualenv `~/vllm-env`
2. Installa vLLM via pip

**macOS Apple Silicon:**
1. Crea virtualenv `~/vllm-env`
2. Installa `vllm-mlx` (backend MLX per Apple Silicon)

**DGX Spark (Blackwell GB10):**

Processo completo in 9 step (~20-30 minuti):

1. **Controlli pre-installazione** — verifica GPU, CUDA toolkit (`nvcc`), spazio disco
2. **uv package manager** — installa `uv` per gestione dipendenze veloce
3. **Virtualenv** — crea `~/vllm-install/.vllm` con Python 3.12
4. **PyTorch** — installa PyTorch 2.9.0+cu130 (CUDA 13.0 per Blackwell)
5. **Triton** — compila Triton da source (branch main con supporto sm_121a)
6. **Dipendenze** — xgrammar, setuptools-scm, apache-tvm-ffi
7. **Clone vLLM** — scarica vLLM (commit testato per DGX Spark)
8. **Patch critiche:**
   - `pyproject.toml` — fix campo license per setuptools recenti
   - `CMakeLists.txt` — aggiunge SM100/SM120/SM121 ai kernel CUTLASS (`_C_stable_libtorch`)
   - flashinfer-python — fix license nella cache uv
   - `use_existing_torch.py` — configura vLLM per usare il PyTorch gia' installato
9. **Build vLLM** — compilazione con `TORCH_CUDA_ARCH_LIST=12.1a`

Puoi seguire tutto il processo in tempo reale nel tab **Logs**.

### 3.2 Verificare l'installazione

Dopo l'installazione, il manager rileva automaticamente vLLM. Nella barra di stato vedrai:

```
vLLM X.X.X detected (backend: cuda)
```

Se non compare, riavvia il manager (`Ctrl+C` e rilancia).

---

## 4. Utilizzo

### 4.1 Tab Server — Avviare un modello

1. **Scegli un modello** dal menu a tendina (preset per la tua piattaforma) oppure inserisci un ID HuggingFace custom
2. **Configura i parametri:**
   - **GPU Memory Utilization** — quanta VRAM usare (0.90 = 90%, consigliato)
   - **Max Model Len** — lunghezza massima del contesto (0 = auto)
   - **Dtype** — tipo dati (auto e' quasi sempre corretto)
   - **Host/Port** — dove esporre l'API vLLM (default: 0.0.0.0:8000)
   - **Chat Template** — necessario solo per modelli quantizzati vecchi (TheBloke)
   - **TP Size** — Tensor Parallel (>1 per multi-GPU)
   - **Extra Args** — argomenti aggiuntivi per vLLM
3. Clicca **Start Server**
4. Segui il caricamento nei log. Quando il server e' online, il pallino diventa verde

**Tempi di caricamento tipici:**
- Modelli 1-3B: 30-60 secondi
- Modelli 7-8B: 1-3 minuti
- Modelli 70B: 3-10 minuti (dipende dalla connessione per il download)

### 4.2 Tab GPU Monitor

Mostra in tempo reale:
- **VRAM** (o Memoria su macOS) — uso corrente e totale
- **GPU Utilization** — percentuale di utilizzo
- **Temperatura** — con colore (verde < 65C, giallo < 80C, rosso > 80C)
- **Potenza** — consumo in Watt
- **Grafico storico** — ultimi 2 minuti (VRAM, Util, Temp)

I dati si aggiornano automaticamente ogni 2 secondi.

### 4.3 Tab Logs

- Log del server in tempo reale via SSE (Server-Sent Events)
- Colorazione automatica: rosso = errori, giallo = warning, blu = info
- **Cerca** — cerca testo nei log con evidenziazione
- **Pulisci Log** — svuota la console

### 4.4 Tab Models

**Ricerca HuggingFace:**
1. Scrivi il nome del modello (es. "Qwen", "Llama", "Mistral")
2. Clicca Cerca
3. Seleziona un modello dalla lista
4. Clicca "Usa questo modello" — ti porta al tab Server con il modello impostato

**Modelli locali (cache):**
- Lista dei modelli gia' scaricati con dimensione su disco
- Puoi eliminare modelli per liberare spazio
- I modelli eliminati verranno riscaricati al prossimo utilizzo

### 4.5 Tab Benchmark

1. Scegli un prompt preset o scrivi il tuo
2. Configura Max Tokens e Temperatura
3. Clicca "Esegui Benchmark"
4. Risultati: tokens generati, tempo totale, **tokens/sec**, **TTFT** (Time to First Token)

Il server deve essere online per eseguire il benchmark.

### 4.6 Tab WebUI (Open WebUI)

Gestisce un container Docker di [Open WebUI](https://github.com/open-webui/open-webui) — un'interfaccia chat che si connette al server vLLM.

**Requisiti:** Docker installato sulla macchina.

1. Clicca **Start Open WebUI**
2. Accedi a `http://<ip>:3000` nel browser
3. Registra un account (locale, solo la prima volta)
4. Inizia a chattare con il modello

### 4.7 Tab Profili

- **Salva Profilo** — salva la configurazione corrente (modello + parametri)
- **Carica Profilo** — ripristina una configurazione salvata
- **Elimina** — rimuove un profilo
- I profili sono salvati in JSON nel config path della piattaforma

### 4.8 Tab DGX Spark (solo su DGX)

Il tab DGX permette di gestire un cluster multi-nodo **interamente dal nodo head**, senza dover accedere fisicamente ai worker. L'unica cosa che serve e' che i worker siano accesi, connessi alla rete, e abbiano SSH attivo.

#### Cluster Overview
- Hostname, IP, info GPU, stato Ray
- Clicca "Aggiorna Stato Cluster" per refresh

#### Gestione Worker Remoti

Sezione centrale per il multi-nodo. Qui aggiungi i nodi worker e li gestisci dal head.

**Aggiungere un worker:**
1. Inserisci IP, user SSH (default: root), e un alias opzionale
2. Clicca "Aggiungi Worker"
3. Il worker appare nella tabella con stato "Aggiunto"

**Ciclo di vita di ogni worker (tutto dal head):**

| Stato | Significato | Azioni disponibili |
|-------|-------------|-------------------|
| Aggiunto | Appena inserito | Setup SSH |
| SSH OK | Chiavi SSH copiate e verificate | Installa vLLM, Verifica |
| Installazione... | Installazione remota in corso (20-40 min) | Nessuna (attendi) |
| Installato | vLLM installato con successo | Verifica |
| Pronto | GPU, vLLM e Ray verificati | Pronto per Ray |
| Ray Connesso | Connesso al cluster Ray | Gestito dal cluster |
| Errore | Qualcosa e' fallito | Riprova SSH, Riprova Installa |

**Setup SSH:**
- Genera chiave RSA sul head (se non esiste)
- Copia la chiave pubblica sul worker con `ssh-copy-id`
- Verifica che la connessione SSH funzioni senza password
- Nota: la prima volta potrebbe chiedere la password del worker nel terminale

**Installazione remota vLLM:**
- Esegue gli stessi 9 step dell'installazione locale, ma via SSH
- Include: uv, PyTorch cu130, Triton da source, vLLM con patch sm_121
- Richiede 20-40 minuti per worker
- Una sola installazione alla volta (per evitare conflitti nei log)
- Progresso visibile nel tab Logs con prefisso `[WORKER <ip>]`

**Verifica:**
- Controlla via SSH: GPU (nvidia-smi), vLLM (import), Ray (import)
- Se tutto OK, lo stato diventa "Pronto"

#### Node Discovery & Network
- **Rileva Nodi** — scansiona interfacce di rete e tabella ARP per trovare altri DGX sulla rete
- **Test Connettivita'** — ping localhost
- **Test NCCL All-Reduce** — verifica NCCL e numero CUDA devices
- **Variabili NCCL** — modifica NCCL_SOCKET_IFNAME, UCX_NET_DEVICES, GLOO_SOCKET_IFNAME, NCCL_IB_HCA

#### Ray Cluster
- **Avvia Cluster Ray** — con un click: avvia Ray head su questo nodo, poi connette automaticamente tutti i worker con stato "pronto" o "installato"
- **Stop Cluster Ray** — ferma Ray su tutti i worker (via SSH) e poi sul head

#### Multi-Node Inference
- Configura **Tensor Parallel** e **Pipeline Parallel**
- Usa i preset multi-nodo per modelli grandi (405B, Mixtral 8x22B)
- Clicca "Start Multi-Node Server"

### 4.9 Guida Multi-Nodo passo-passo

Questa guida copre la procedura completa per avviare un modello distribuito su piu' DGX Spark. **Tutto viene fatto dal nodo head.**

#### Prerequisiti

- 2+ DGX Spark sulla stessa rete
- Tutti i nodi accesi con SSH attivo
- Il web manager avviato **solo sul nodo head**

#### Fase 1: Setup iniziale (una tantum)

1. Avvia il manager sul nodo head:
   ```bash
   python vllm_manager_web.py --host 0.0.0.0 --port 5000
   ```
2. Apri il browser: `http://<ip-head>:5000`
3. **Installa vLLM sul nodo head** dal tab Server (se non gia' fatto)

#### Fase 2: Aggiungere i worker

1. Vai al tab **DGX Spark**
2. (Opzionale) Clicca **"Rileva Nodi"** per vedere gli IP sulla rete
3. Per ogni worker:
   - Inserisci IP, user (es. `root`), alias (es. `dgx-2`)
   - Clicca **"+ Aggiungi Worker"**

#### Fase 3: Setup SSH (per ogni worker)

1. Nella tabella worker, clicca **"Setup SSH"** sul primo worker
2. Attendi che lo stato diventi **"SSH OK"**
3. Ripeti per ogni worker
4. Nota: se il worker non ha le chiavi SSH preconfigurate, potrebbe servire inserire la password la prima volta nel terminale dove gira il manager

#### Fase 4: Installare vLLM sui worker (per ogni worker)

1. Clicca **"Installa vLLM"** sul primo worker (stato SSH OK)
2. Lo stato diventa **"Installazione..."** — segui il progresso nel tab **Logs**
3. Dopo 20-40 minuti lo stato diventa **"Installato"**
4. **Una sola installazione alla volta** — attendi che finisca prima di iniziare la prossima
5. Ripeti per ogni worker

#### Fase 5: Verificare i worker

1. Clicca **"Verifica"** su ogni worker installato
2. Controlla che GPU, vLLM e Ray siano OK
3. Lo stato diventa **"Pronto"**

#### Fase 6: Configurare la rete (se necessario)

1. Nella sezione **"Variabili Ambiente NCCL"**:
   - Verifica che `NCCL_SOCKET_IFNAME` punti all'interfaccia ad alta velocita'
   - Imposta `NCCL_IB_HCA` se hai InfiniBand
2. Clicca **"Applica Env Vars"**
3. Esegui **"Test NCCL All-Reduce"** per conferma

#### Fase 7: Avviare il cluster Ray

1. Clicca **"Avvia Cluster Ray (Head + Worker)"**
2. Il sistema:
   - Avvia Ray head su questo nodo (porta 6379)
   - Si connette via SSH a ogni worker "pronto" e lancia `ray start`
   - Aggiorna lo stato dei worker a "Ray Connesso"
3. Verifica con **"Aggiorna Stato Cluster"** che tutti i nodi siano visibili

#### Fase 8: Avviare il server multi-nodo

1. Nella sezione **"Multi-Node Inference"**:
   - Clicca su un preset (es. Llama 405B FP8 → TP=4) oppure imposta manualmente TP e PP
   - TP = numero totale di GPU nel cluster (1 per nodo DGX Spark)
2. Clicca **"Start Multi-Node Server"**
3. Segui il caricamento nel tab **Logs**
4. Quando il pallino diventa verde, l'API e' attiva su `http://<ip-head>:8000`

#### Esempio: Llama 3.1 405B-FP8 su 4 nodi

```
Nodo head:  192.168.1.10 (qui gira il manager)
Worker 1:   192.168.1.11
Worker 2:   192.168.1.12
Worker 3:   192.168.1.13

TP = 4 (una GPU per nodo, 4 nodi)
PP = 1
Modello: meta-llama/Llama-3.1-405B-Instruct-FP8
VRAM richiesta: ~100 GB per nodo
```

Il modello viene diviso su 4 GPU (~100 GB ciascuna). L'endpoint API e' solo sull'head:
```bash
curl http://192.168.1.10:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"meta-llama/Llama-3.1-405B-Instruct-FP8","messages":[{"role":"user","content":"Ciao!"}]}'
```

---

## 5. Usare il server vLLM

Una volta che il server e' online (pallino verde), l'API e' compatibile con OpenAI.

### 5.1 Test rapido via curl

```bash
curl http://<ip>:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen/Qwen2.5-7B-Instruct",
    "messages": [{"role": "user", "content": "Ciao!"}],
    "temperature": 0.7,
    "max_tokens": 200
  }'
```

### 5.2 Python con OpenAI SDK

```python
from openai import OpenAI

client = OpenAI(base_url="http://<ip>:8000/v1", api_key="not-needed")
response = client.chat.completions.create(
    model="Qwen/Qwen2.5-7B-Instruct",
    messages=[{"role": "user", "content": "Ciao!"}],
    temperature=0.7,
    max_tokens=200,
)
print(response.choices[0].message.content)
```

### 5.3 Modelli disponibili

```bash
curl http://<ip>:8000/v1/models
```

---

## 6. Troubleshooting

### Il server non parte

**Errore: "vLLM non trovato"**
- Clicca "Installa vLLM" e attendi il completamento
- Controlla i log per errori durante l'installazione

**Errore: CUDA / libcudart**
- Su DGX Spark: usa l'installazione automatica integrata (gestisce le patch CUDA 13)
- Su Linux: verifica `nvidia-smi` e che il driver sia aggiornato

**Errore: Out of Memory**
- Riduci GPU Memory Utilization (es. 0.80)
- Riduci Max Model Len
- Usa un modello piu' piccolo o quantizzato

### GPU non rilevata

- Verifica che `nvidia-smi` funzioni
- Su macOS: la GPU e' sempre "Apple Silicon" (memoria unificata)

### Il tab DGX non mostra dati

- Clicca "Aggiorna Stato Cluster"
- Verifica che la macchina sia effettivamente un DGX Spark (`uname -m` deve dare `aarch64`)

### Open WebUI non parte

- Verifica che Docker sia installato: `docker --version`
- Verifica che il container non esista gia': `docker ps -a`

### Accesso remoto non funziona

- Assicurati di aver usato `--host 0.0.0.0`
- Verifica che la porta 5000 non sia bloccata dal firewall:
  ```bash
  sudo ufw allow 5000    # Ubuntu
  ```
- Verifica anche la porta 8000 (API vLLM) se vuoi accedere direttamente all'API

---

## 7. API Reference

Il manager espone le seguenti API REST:

### Server
| Metodo | Endpoint | Descrizione |
|--------|----------|-------------|
| POST | `/api/server/start` | Avvia server vLLM |
| POST | `/api/server/stop` | Ferma server |
| POST | `/api/server/stop-all` | Ferma tutto (vLLM + Docker + WSL) |
| GET | `/api/server/status` | Stato server (online/running/version) |
| POST | `/api/server/install` | Installa vLLM per la piattaforma |

### GPU
| Metodo | Endpoint | Descrizione |
|--------|----------|-------------|
| GET | `/api/gpu` | Metriche GPU + storico |

### Logs
| Metodo | Endpoint | Descrizione |
|--------|----------|-------------|
| GET | `/api/logs/stream` | SSE stream log in tempo reale |
| GET | `/api/logs/history` | Tutti i log correnti |
| POST | `/api/logs/clear` | Pulisci log |

### Models
| Metodo | Endpoint | Descrizione |
|--------|----------|-------------|
| GET | `/api/models/search?q=...` | Cerca su HuggingFace |
| GET | `/api/models/local` | Lista modelli in cache |
| POST | `/api/models/delete` | Elimina modello dalla cache |

### Benchmark
| Metodo | Endpoint | Descrizione |
|--------|----------|-------------|
| POST | `/api/benchmark` | Esegui benchmark |

### WebUI (Docker)
| Metodo | Endpoint | Descrizione |
|--------|----------|-------------|
| GET | `/api/webui/status` | Stato container |
| POST | `/api/webui/start` | Avvia Open WebUI |
| POST | `/api/webui/stop` | Ferma Open WebUI |

### Profili
| Metodo | Endpoint | Descrizione |
|--------|----------|-------------|
| GET | `/api/profiles` | Lista profili |
| POST | `/api/profiles/save` | Salva profilo |
| POST | `/api/profiles/delete` | Elimina profilo |

### DGX Spark — Network & Diagnostica
| Metodo | Endpoint | Descrizione |
|--------|----------|-------------|
| GET | `/api/dgx/cluster` | Overview cluster (host, GPU, Ray) |
| GET | `/api/dgx/discover` | Rileva nodi sulla rete (ARP) |
| GET | `/api/dgx/connectivity` | Test connettivita' (ping) |
| GET | `/api/dgx/nccl-test` | Test NCCL e GPU devices |
| POST | `/api/dgx/env` | Aggiorna env vars NCCL/UCX |
| GET | `/api/platform` | Info piattaforma |

### DGX Spark — Gestione Worker Remoti
| Metodo | Endpoint | Descrizione |
|--------|----------|-------------|
| GET | `/api/dgx/workers` | Lista worker con stato |
| POST | `/api/dgx/workers/add` | Aggiungi worker `{ip, user, alias}` |
| POST | `/api/dgx/workers/remove` | Rimuovi worker `{ip}` |
| POST | `/api/dgx/workers/ssh-setup` | Setup SSH keys su worker `{ip}` |
| POST | `/api/dgx/workers/ssh-test` | Test SSH verso worker `{ip}` |
| POST | `/api/dgx/workers/install` | Installa vLLM su worker remoto `{ip}` (20-40 min) |
| POST | `/api/dgx/workers/verify` | Verifica GPU/vLLM/Ray su worker `{ip}` |

### DGX Spark — Ray Cluster
| Metodo | Endpoint | Descrizione |
|--------|----------|-------------|
| POST | `/api/dgx/ray/start-head` | Avvia solo Ray head (uso avanzato) |
| POST | `/api/dgx/ray/connect-worker` | Connetti singolo worker (uso avanzato) |
| POST | `/api/dgx/ray/stop` | Stop solo Ray locale (uso avanzato) |
| POST | `/api/dgx/ray/start-cluster` | Avvia Ray head + connetti tutti i worker pronti |
| POST | `/api/dgx/ray/stop-cluster` | Stop Ray su tutti i worker e poi sull'head |

---

## 8. Struttura file

```
vllm-manager/
  vllm_manager_web.py   # Unico file dell'applicazione
  README.md             # Panoramica progetto
  GUIDA.md              # Questa guida
  .gitignore
```

I profili, le configurazioni e la lista worker sono salvati in:

| Piattaforma | Path |
|---|---|
| Windows | `%APPDATA%\vLLM Manager\config.json` |
| macOS | `~/Library/Application Support/vLLM Manager/config.json` |
| Linux / DGX | `~/.config/vllm-manager/config.json` |

L'installazione vLLM su DGX crea:

```
~/vllm-install/
  .vllm/              # Virtualenv Python 3.12
  triton/             # Triton compilato da source
  vllm/               # vLLM compilato con patch sm_121
  vllm_env.sh         # Script di attivazione ambiente
```
