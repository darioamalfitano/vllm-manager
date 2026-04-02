# Guida all'uso di vLLM Manager

## Indice

1. [Requisiti](#1-requisiti)
2. [Installazione](#2-installazione)
3. [Primo avvio](#3-primo-avvio)
4. [Tab Server](#4-tab-server)
5. [Tab GPU Monitor](#5-tab-gpu-monitor)
6. [Tab Logs](#6-tab-logs)
7. [Tab Models](#7-tab-models)
8. [Tab Benchmark](#8-tab-benchmark)
9. [Tab WebUI](#9-tab-webui)
10. [Tab Profili](#10-tab-profili)
11. [Tab DGX Spark](#11-tab-dgx-spark-solo-dgx)
12. [Compilare come eseguibile standalone](#12-compilare-come-eseguibile-standalone)
13. [Risoluzione problemi](#13-risoluzione-problemi)

---

## 1. Requisiti

### Tutte le piattaforme
- Python 3.10 o superiore
- tkinter (incluso nella maggior parte delle installazioni Python)
- Connessione internet (per scaricare modelli da HuggingFace)

### Windows
- Windows 10/11
- WSL2 con distribuzione Ubuntu installata
- NVIDIA GPU con driver aggiornati
- vLLM installato dentro WSL (virtualenv consigliato)
- Docker Desktop (opzionale, per Open WebUI)

### Linux
- Ubuntu/Debian (o derivati)
- NVIDIA GPU con driver CUDA
- vLLM installato (virtualenv consigliato)
- Docker (opzionale, per Open WebUI)

### macOS
- macOS 14 (Sonoma) o superiore
- Apple Silicon (M1/M2/M3/M4)
- vllm-metal o vllm-mlx installato
- Docker Desktop (opzionale, per Open WebUI)

### DGX Spark
- DGX OS (Ubuntu 24.04 ARM64)
- GPU GB10 Blackwell
- vLLM compilato da source con `TORCH_CUDA_ARCH_LIST="12.1"` (l'app puo' farlo per te)
- Ray (per multi-nodo)

---

## 2. Installazione

### Installazione rapida

Non serve installare nulla oltre a Python. L'app usa solo la libreria standard.

```bash
git clone <url-del-repo>
cd vllm-manager
python vllm_manager.py
```

### Installazione vLLM (se non presente)

#### Windows (dentro WSL)
```bash
wsl -d Ubuntu-22.04
python3 -m venv ~/vllm-env
source ~/vllm-env/bin/activate
pip install vllm
```

#### Linux
```bash
python3 -m venv ~/vllm-env
source ~/vllm-env/bin/activate
pip install vllm
```

#### macOS Apple Silicon
```bash
# Opzione 1: vllm-metal
pip install vllm-metal

# Opzione 2: vllm-mlx
pip install vllm-mlx
```

#### DGX Spark
L'app include un installatore automatico nel Tab DGX Spark (vedi sezione 11).

---

## 3. Primo avvio

```bash
python vllm_manager.py
```

All'avvio l'app:

1. **Rileva la piattaforma** automaticamente (Windows WSL2 / Linux / macOS / DGX Spark)
2. **Cerca vLLM** installato nel sistema (controlla virtualenv comuni e installazione di sistema)
3. **Carica i preset modelli** adatti alla tua piattaforma e quantita' di VRAM/memoria
4. **Avvia il monitoraggio GPU** in background

La barra di stato in basso mostra:
- Indicatore server (rosso = offline, verde = online)
- Informazioni GPU in tempo reale
- Piattaforma rilevata e versione app

---

## 4. Tab Server

Questo e' il tab principale per avviare e gestire il server vLLM.

### Selezione modello

- Il menu a tendina mostra i **preset** consigliati per la tua piattaforma
- Puoi anche scrivere manualmente qualsiasi ID modello HuggingFace (es. `mistralai/Mistral-7B-v0.3`)
- Sotto il nome modello viene mostrata una stima della VRAM/Memoria necessaria

### Parametri

| Parametro | Descrizione | Default |
|---|---|---|
| **GPU Memory Utilization** | Percentuale di VRAM da usare (solo NVIDIA) | 0.90 |
| **Max Model Len** | Lunghezza massima contesto in token (0 = auto) | 0 |
| **Dtype** | Tipo dati: auto, float16, bfloat16, float32 | auto |
| **Enable Prefix Caching** | Cache dei prefix comuni per velocizzare le risposte | attivo |
| **Host** | Indirizzo di ascolto del server | 0.0.0.0 |
| **Port** | Porta del server | 8000 |
| **Chat Template** | Template per la formattazione chat (auto per modelli recenti) | (auto) |
| **Attention Backend** | Backend attention (solo DGX): TRITON_ATTN, FLASHINFER, XFORMERS | (auto) |
| **Backend MLX** | Backend macOS: auto, vllm-metal, vllm-mlx | auto |
| **Extra Args** | Argomenti aggiuntivi da passare a vLLM | vuoto |

### Bottoni

- **Start Server**: avvia il server vLLM con i parametri configurati
- **Stop Server**: ferma il server vLLM
- **Stoppa Tutto**: ferma vLLM + container Docker. Su Windows termina anche WSL
- **Salva Profilo**: salva la configurazione corrente come profilo riutilizzabile

### Log preview

In basso viene mostrata un'anteprima dei log del server. I log completi sono nel Tab Logs.

### Consigli

- Se il modello non entra in memoria, riduci `GPU Memory Utilization` o `Max Model Len`
- Per modelli quantizzati vecchi (TheBloke), seleziona il chat template corretto (llama-2, mistral)
- I modelli recenti (Qwen2.5, Llama-3) gestiscono il template automaticamente

---

## 5. Tab GPU Monitor

Monitoraggio in tempo reale delle risorse GPU.

### Metriche mostrate

| Metrica | NVIDIA | Apple Silicon |
|---|---|---|
| VRAM/Memoria usata | Si | Si (memoria unificata) |
| Utilizzo GPU % | Si | N/A |
| Temperatura | Si | N/A |
| Potenza (Watt) | Si | N/A |

### Grafico storico

Il grafico mostra gli ultimi 2 minuti di:
- **Blu**: utilizzo VRAM/Memoria (%)
- **Verde**: utilizzo GPU (%)
- **Rosso**: temperatura

I dati vengono aggiornati ogni 2 secondi.

---

## 6. Tab Logs

Log completi del server vLLM con colorazione automatica:
- **Rosso**: errori
- **Giallo**: warning
- **Blu**: messaggi informativi

### Funzionalita'

- **Cerca**: usa la barra di ricerca o premi `Ctrl+F` (Windows/Linux) / `Cmd+F` (macOS)
- **Pulisci Log**: svuota l'area log

I log mostrano anche i comandi eseguiti (`[CMD]`), utili per debug.

---

## 7. Tab Models

### Ricerca su HuggingFace

1. Scrivi il nome del modello nella barra di ricerca (es. "qwen2.5", "llama instruct")
2. Clicca **Cerca** o premi Invio
3. I risultati mostrano: nome modello, numero download, likes
4. Seleziona un modello e clicca **Usa questo modello** per caricarlo nel Tab Server

### Modelli scaricati localmente

La sezione in basso mostra i modelli gia' presenti nella cache HuggingFace del sistema:
- **Aggiorna Lista**: riscansiona la cache
- **Elimina Selezionato**: rimuove un modello dalla cache per liberare spazio disco

I modelli eliminati verranno riscaricati automaticamente al prossimo utilizzo.

---

## 8. Tab Benchmark

Testa le performance del server vLLM attivo.

### Come usarlo

1. Assicurati che il server sia **online** (indicatore verde)
2. Scegli un prompt preset o scrivi il tuo (seleziona "Custom")
3. Regola **Max Tokens** e **Temperatura**
4. Clicca **Esegui Benchmark**

### Metriche misurate

| Metrica | Descrizione |
|---|---|
| **Tokens generati** | Numero di token nella risposta |
| **Tempo totale** | Tempo dall'invio alla fine della generazione |
| **Tokens/sec** | Velocita' di generazione |
| **TTFT** | Time to First Token — tempo prima del primo token generato |

In basso viene mostrata un'anteprima della risposta generata.

### Consigli per benchmark affidabili

- Esegui il benchmark piu' volte (la prima esecuzione puo' essere lenta per warm-up)
- Usa prompt di lunghezze diverse per valutare le performance
- Il conteggio token e' basato sui chunk SSE, potrebbe differire leggermente dal conteggio reale

---

## 9. Tab WebUI

Gestisci Open WebUI, un'interfaccia web per chattare con i modelli, via Docker.

### Prerequisiti

- Docker installato e avviato (Docker Desktop su Windows/macOS, Docker Engine su Linux)

### Configurazione

| Campo | Default | Descrizione |
|---|---|---|
| **Container name** | open-webui | Nome del container Docker |
| **Docker image** | ghcr.io/open-webui/open-webui:main | Immagine Docker da usare |
| **Port** | 3000 | Porta locale per accedere alla WebUI |

### Bottoni

- **Start Open WebUI**: avvia il container (lo crea al primo avvio)
- **Stop Open WebUI**: ferma il container
- **Apri nel Browser**: apre `http://localhost:3000` nel browser predefinito
- **Aggiorna Status**: controlla lo stato del container

### Primo utilizzo

1. Assicurati che Docker sia avviato
2. Clicca **Start Open WebUI** (il primo avvio scarica l'immagine, puo' richiedere qualche minuto)
3. Clicca **Apri nel Browser**
4. Nella WebUI, configura la connessione al server vLLM:
   - URL: `http://host.docker.internal:8000/v1` (Windows/macOS) oppure `http://localhost:8000/v1` (Linux)
   - API Key: qualsiasi valore (es. "not-needed")

---

## 10. Tab Profili

Salva e gestisci configurazioni server riutilizzabili.

### Cosa viene salvato in un profilo

- Modello selezionato
- Tutti i parametri (GPU mem util, max model len, dtype, prefix caching, host, port, extra args, chat template)

### Operazioni

- **Carica Profilo**: applica un profilo salvato e passa al Tab Server
- **Salva Corrente come Nuovo**: salva la configurazione attuale
- **Elimina Profilo**: rimuove un profilo
- **Esporta Profili (JSON)**: salva tutti i profili in un file JSON (utile per backup o condivisione)
- **Importa Profili (JSON)**: carica profili da un file JSON

### Dove vengono salvati

I profili sono memorizzati nel file `config.json` nella directory di configurazione della piattaforma:

| Piattaforma | Path |
|---|---|
| Windows | `%APPDATA%\vLLM Manager\config.json` |
| macOS | `~/Library/Application Support/vLLM Manager/config.json` |
| Linux / DGX | `~/.config/vllm-manager/config.json` |

---

## 11. Tab DGX Spark (solo DGX)

Questo tab appare solo quando l'app rileva un sistema DGX Spark. Gestisce l'intero cluster multi-nodo.

### Cluster Overview

Mostra in tempo reale:
- Hostname e IP del nodo locale
- Stato GPU: nome, VRAM usata/totale, temperatura, utilizzo
- Stato Ray cluster: nodi connessi, risorse disponibili

Clicca **Aggiorna Stato Cluster** per rinfrescare le informazioni.

### Node Discovery & Network

**Rileva Nodi**: scansiona la rete per trovare altri DGX Spark collegati. Mostra le interfacce di rete (ConnectX-7) e i nodi raggiungibili.

**Test Connettivita'**: verifica la comunicazione tra i nodi.

**Test NCCL All-Reduce**: esegue un test NCCL per verificare che la comunicazione GPU inter-nodo funzioni.

**Variabili Ambiente NCCL**: editor per configurare le variabili ambiente critiche per la comunicazione inter-nodo:

| Variabile | Descrizione |
|---|---|
| `NCCL_SOCKET_IFNAME` | Interfaccia di rete per NCCL |
| `UCX_NET_DEVICES` | Device di rete per UCX |
| `GLOO_SOCKET_IFNAME` | Interfaccia per Gloo |
| `NCCL_IB_HCA` | HCA InfiniBand/RDMA per NCCL |

Clicca **Applica Env Vars** dopo aver modificato i valori.

### Ray Cluster

Per l'inferenza multi-nodo serve un cluster Ray.

**Setup passo-passo:**

1. Sul nodo principale: clicca **Avvia Ray Head**
2. Per ogni nodo aggiuntivo: inserisci IP e utente, poi clicca **Connetti Worker**
3. Verifica lo stato nel **Ray Dashboard** (si apre nel browser)

**Bottoni:**
- **Avvia Ray Head**: avvia il nodo principale Ray (`ray start --head`)
- **Connetti Worker**: connette un nodo remoto al cluster via SSH
- **Stop Ray**: ferma Ray su tutti i nodi
- **Apri Ray Dashboard**: apre il dashboard web di Ray

### Multi-Node Inference

Configura l'inferenza distribuita su piu' nodi:

| Parametro | Descrizione |
|---|---|
| **Tensor Parallel Size** | Numero di GPU su cui dividere il modello (1 = singolo nodo) |
| **Pipeline Parallel Size** | Parallelismo pipeline (per modelli molto grandi) |
| **Attention Backend** | TRITON_ATTN, FLASHINFER, o XFORMERS |

**Preset Multi-Nodo**: tabella con configurazioni testate per modelli grandi. Seleziona un preset e clicca **Usa Preset Selezionato**.

**Start Multi-Node Server**: verifica che Ray sia attivo, applica le env vars NCCL, e lancia vLLM distribuito.

### vLLM Installation

Se vLLM non e' rilevato sul DGX Spark, questa sezione permette di installarlo:

**Build from source** (consigliato):
1. Crea un virtualenv Python
2. Installa PyTorch nightly ARM64 con supporto sm_121
3. Clona e compila vLLM da source
4. Il progresso viene mostrato in tempo reale

**Docker NVIDIA**:
- Scarica l'immagine ufficiale NVIDIA (`nvcr.io/nvidia/vllm:latest`)
- In alternativa, l'immagine community `eugr/spark-vllm`

### SSH Configuration

Per gestire nodi remoti del cluster:

1. Inserisci **IP** e **utente** del nodo target
2. **Setup SSH Keys**: genera chiavi SSH e le copia sul nodo remoto
3. **Test Connessione**: verifica che la connessione SSH funzioni

---

## 12. Compilare come eseguibile standalone

### Windows
```
build_exe.bat
```
Produce `dist\vLLM Manager.exe` (standalone, ~15-25 MB).

### Linux
```bash
./build_linux.sh
```
Produce `dist/vllm-manager`. Se `appimagetool` e' installato, crea anche un `.AppImage`.

### macOS
```bash
./build_macos.sh
```
Produce `dist/vLLM Manager.app`. Copia nella cartella Applicazioni per installare.

### DGX Spark
```bash
./build_dgx.sh
```
Produce `dist/vllm-manager-dgx`. Esegui direttamente sul DGX Spark.

> Nota: i build script creano un virtualenv temporaneo (`.venv-build`) e installano PyInstaller automaticamente.

---

## 13. Risoluzione problemi

### Il server non parte

- **"Could not start WSL"** (Windows): verifica che WSL2 sia installato (`wsl --install`) e che la distribuzione Ubuntu sia presente (`wsl -l -v`)
- **Out of memory**: il modello e' troppo grande per la VRAM disponibile. Prova a ridurre `GPU Memory Utilization` (es. 0.80) o `Max Model Len` (es. 2048), oppure scegli un modello piu' piccolo
- **Porta occupata**: cambia la porta nel Tab Server (es. 8001) se la 8000 e' gia' in uso

### GPU non rilevata

- **Windows**: verifica che i driver NVIDIA siano aggiornati e che `nvidia-smi` funzioni dentro WSL
- **Linux**: installa i driver NVIDIA e verifica con `nvidia-smi`
- **macOS**: il monitoraggio mostra la memoria unificata; temperatura e potenza non sono disponibili

### vLLM non trovato

L'app cerca vLLM in queste posizioni:
1. `~/vllm-env/bin/activate` (virtualenv nella home)
2. `~/.venv/vllm/bin/activate`
3. `/opt/vllm/venv/bin/activate`
4. Installazione di sistema (`python3 -c 'import vllm'`)

Su macOS cerca anche `vllm_metal` e `vllm_mlx`.

Se vLLM e' installato in un percorso diverso, usa il campo **Extra Args** per specificare il percorso, oppure attiva il virtualenv prima di lanciare l'app.

### Open WebUI non si avvia

- Verifica che Docker sia installato e avviato
- Su Windows: Docker Desktop deve essere in esecuzione
- Controlla i log nel Tab WebUI per dettagli sull'errore
- Prova a rimuovere il container e ricrearlo: `docker rm open-webui` poi clicca Start

### Modello lento o TTFT alto

- Attiva **Enable Prefix Caching** per velocizzare richieste ripetute
- Usa `bfloat16` come dtype se la GPU lo supporta
- Riduci `Max Model Len` se non serve un contesto lungo
- Su DGX: verifica che il backend attention sia ottimale (prova TRITON_ATTN o FLASHINFER)

### Problemi multi-nodo (DGX)

- Verifica che Ray sia attivo su tutti i nodi (`ray status`)
- Controlla che le interfacce NCCL siano corrette (devono corrispondere alle interfacce ConnectX-7 reali)
- Testa la connettivita' SSH prima di connettere i worker
- Verifica con il test NCCL All-Reduce che la comunicazione inter-nodo funzioni
