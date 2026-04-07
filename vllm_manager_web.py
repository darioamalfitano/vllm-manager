#!/usr/bin/env python3
"""vLLM Manager — Web interface for remote management of vLLM servers.

Run with: python vllm_manager_web.py [--host 0.0.0.0] [--port 5000]
Then open http://<server-ip>:5000 in your browser.

Requires: pip install flask
"""

import abc
import argparse
import base64
import json
import os
import platform
import queue
import shutil
import subprocess
import sys
import threading
import time
from collections import deque
from pathlib import Path
from urllib import request, parse

from flask import Flask, render_template_string, jsonify, request as flask_request, Response

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

APP_NAME = "vLLM Manager"
APP_VERSION = "2.1.0-web"

CHAT_TEMPLATES = {
    "": "",
    "llama-2": (
        "{% if messages[0]['role'] == 'system' %}"
        "{% set loop_messages = messages[1:] %}"
        "{% set system_message = messages[0]['content'] %}"
        "{% else %}"
        "{% set loop_messages = messages %}"
        "{% set system_message = false %}"
        "{% endif %}"
        "{% for message in loop_messages %}"
        "{% if (message['role'] == 'user') != (loop.index0 % 2 == 0) %}"
        "{{ raise_exception('Conversation roles must alternate user/assistant/user/assistant/...') }}"
        "{% endif %}"
        "{% if loop.index0 == 0 and system_message != false %}"
        "{% set content = '<<SYS>>\\n' + system_message + '\\n<</SYS>>\\n\\n' + message['content'] %}"
        "{% else %}"
        "{% set content = message['content'] %}"
        "{% endif %}"
        "{% if message['role'] == 'user' %}"
        "{{ bos_token + '[INST] ' + content.strip() + ' [/INST]' }}"
        "{% elif message['role'] == 'assistant' %}"
        "{{ ' ' + content.strip() + ' ' + eos_token }}"
        "{% endif %}"
        "{% endfor %}"
    ),
    "mistral": (
        "{{ bos_token }}"
        "{% for message in messages %}"
        "{% if message['role'] == 'user' %}"
        "{{ '[INST] ' + message['content'] + ' [/INST]' }}"
        "{% elif message['role'] == 'assistant' %}"
        "{{ message['content'] + eos_token }}"
        "{% endif %}"
        "{% endfor %}"
    ),
    "chatml": (
        "{% for message in messages %}"
        "{{'<|im_start|>' + message['role'] + '\\n' + message['content'] + '<|im_end|>' + '\\n'}}"
        "{% endfor %}"
        "{% if add_generation_prompt %}"
        "{{ '<|im_start|>assistant\\n' }}"
        "{% endif %}"
    ),
}


# ---------------------------------------------------------------------------
# 1. PlatformInfo
# ---------------------------------------------------------------------------

class PlatformInfo:
    def __init__(self):
        self.os = self._detect_os()
        self.arch = platform.machine().lower()
        if self.arch in ("x86_64", "amd64"):
            self.arch = "x86_64"
        elif self.arch in ("aarch64", "arm64"):
            self.arch = "aarch64"
        self.gpu_type = self._detect_gpu_type()
        self.is_dgx = self._detect_dgx()
        self.vram_total = 0

    def _detect_os(self) -> str:
        if sys.platform == "win32":
            return "windows-wsl"
        elif sys.platform == "darwin":
            return "macos"
        else:
            try:
                with open("/proc/version", "r") as f:
                    if "microsoft" in f.read().lower():
                        return "windows-wsl"
            except Exception:
                pass
            return "linux"

    def _detect_gpu_type(self) -> str:
        if self.os == "macos":
            return "apple-silicon" if self.arch == "aarch64" else "none"
        if shutil.which("nvidia-smi"):
            return "nvidia"
        if self.os == "windows-wsl":
            return "nvidia"
        return "none"

    def _detect_dgx(self) -> bool:
        if self.os != "linux" or self.arch != "aarch64":
            return False
        if os.path.exists("/etc/dgx-release"):
            return True
        try:
            proc = subprocess.run(
                ["nvidia-smi", "--query-gpu=compute_cap", "--format=csv,noheader,nounits"],
                capture_output=True, text=True, timeout=5,
            )
            if proc.returncode == 0 and proc.stdout.strip().startswith("12."):
                return True
        except Exception:
            pass
        return False

    @property
    def display_name(self) -> str:
        names = {
            "windows-wsl": "Windows WSL2",
            "linux": "DGX Spark" if self.is_dgx else "Linux",
            "macos": "macOS Apple Silicon" if self.arch == "aarch64" else "macOS",
        }
        return names.get(self.os, self.os)

    @property
    def vram_label(self) -> str:
        return "Memoria" if self.os == "macos" else "VRAM"


# ---------------------------------------------------------------------------
# 2. CommandRunner
# ---------------------------------------------------------------------------

class CommandRunner(abc.ABC):
    @abc.abstractmethod
    def run(self, cmd: str, timeout: int = 30) -> tuple:
        pass

    @abc.abstractmethod
    def popen(self, cmd: str, env: dict | None = None) -> subprocess.Popen:
        pass

    @abc.abstractmethod
    def run_raw(self, cmd: str, timeout: int = 30) -> tuple:
        pass

    @abc.abstractmethod
    def shutdown(self):
        pass

    @abc.abstractmethod
    def is_available(self) -> bool:
        pass


class WSLCommandRunner(CommandRunner):
    def __init__(self, distro: str = "Ubuntu-22.04"):
        self.distro = distro
        self._no_window = 0x08000000 if sys.platform == "win32" else 0

    def run(self, cmd: str, timeout: int = 30) -> tuple:
        try:
            proc = subprocess.run(
                ["wsl", "-d", self.distro, "--", "bash", "-lc", cmd],
                capture_output=True, text=True, timeout=timeout,
                creationflags=self._no_window,
            )
            return proc.returncode, proc.stdout, proc.stderr
        except subprocess.TimeoutExpired:
            return -1, "", "Command timed out"
        except Exception as e:
            return -1, "", str(e)

    def popen(self, cmd: str, env: dict | None = None) -> subprocess.Popen:
        return subprocess.Popen(
            ["wsl", "-d", self.distro, "--", "bash", "-lc", cmd],
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, bufsize=1, creationflags=self._no_window,
        )

    def run_raw(self, cmd: str, timeout: int = 30) -> tuple:
        try:
            proc = subprocess.run(
                ["wsl", "-d", self.distro, "--", "bash", "-c", cmd],
                capture_output=True, text=True, timeout=timeout,
                creationflags=self._no_window,
            )
            return proc.returncode, proc.stdout, proc.stderr
        except subprocess.TimeoutExpired:
            return -1, "", "Command timed out"
        except Exception as e:
            return -1, "", str(e)

    def shutdown(self):
        try:
            subprocess.run(["wsl", "--terminate", self.distro],
                           timeout=15, creationflags=self._no_window)
        except Exception:
            pass

    def is_available(self) -> bool:
        rc, out, _ = self.run("echo ok", timeout=10)
        return rc == 0 and "ok" in out


class NativeCommandRunner(CommandRunner):
    def run(self, cmd: str, timeout: int = 30) -> tuple:
        try:
            proc = subprocess.run(["bash", "-lc", cmd],
                                  capture_output=True, text=True, timeout=timeout)
            return proc.returncode, proc.stdout, proc.stderr
        except subprocess.TimeoutExpired:
            return -1, "", "Command timed out"
        except Exception as e:
            return -1, "", str(e)

    def popen(self, cmd: str, env: dict | None = None) -> subprocess.Popen:
        run_env = os.environ.copy()
        if env:
            run_env.update(env)
        return subprocess.Popen(["bash", "-lc", cmd],
                                stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                text=True, bufsize=1, env=run_env)

    def run_raw(self, cmd: str, timeout: int = 30) -> tuple:
        try:
            proc = subprocess.run(["bash", "-c", cmd],
                                  capture_output=True, text=True, timeout=timeout)
            return proc.returncode, proc.stdout, proc.stderr
        except subprocess.TimeoutExpired:
            return -1, "", "Command timed out"
        except Exception as e:
            return -1, "", str(e)

    def shutdown(self):
        pass

    def is_available(self) -> bool:
        rc, out, _ = self.run("echo ok", timeout=5)
        return rc == 0 and "ok" in out


class DGXCommandRunner(NativeCommandRunner):
    DGX_ENV = {
        "VLLM_ATTENTION_BACKEND": "TRITON_ATTN",
        "NCCL_SOCKET_IFNAME": "enP2p1s0f1np1",
        "UCX_NET_DEVICES": "enP2p1s0f1np1",
        "GLOO_SOCKET_IFNAME": "enp1s0f1np1",
    }

    def __init__(self, extra_env: dict | None = None):
        super().__init__()
        self.dgx_env = dict(self.DGX_ENV)
        if extra_env:
            self.dgx_env.update(extra_env)

    def popen(self, cmd: str, env: dict | None = None) -> subprocess.Popen:
        merged = dict(self.dgx_env)
        if env:
            merged.update(env)
        return super().popen(cmd, env=merged)

    def update_env(self, key: str, value: str):
        self.dgx_env[key] = value


def create_command_runner(platform_info: PlatformInfo) -> CommandRunner:
    if platform_info.os == "windows-wsl":
        return WSLCommandRunner()
    elif platform_info.is_dgx:
        return DGXCommandRunner()
    else:
        return NativeCommandRunner()


# ---------------------------------------------------------------------------
# 3. GpuMonitor
# ---------------------------------------------------------------------------

class BaseGpuMonitor(abc.ABC):
    def __init__(self, interval: float = 2.0):
        self.interval = interval
        self._stop = threading.Event()
        self._thread = None
        self.latest = {}
        self.history_vram = deque(maxlen=60)
        self.history_util = deque(maxlen=60)
        self.history_temp = deque(maxlen=60)

    def start(self):
        self._stop.clear()
        self._thread = threading.Thread(target=self._poll, daemon=True)
        self._thread.start()

    def stop(self):
        self._stop.set()

    @abc.abstractmethod
    def _poll(self):
        pass


class NvidiaGpuMonitor(BaseGpuMonitor):
    def __init__(self, runner: CommandRunner, interval: float = 2.0):
        super().__init__(interval)
        self.runner = runner

    def _poll(self):
        while not self._stop.is_set():
            try:
                rc, out, _ = self.runner.run(
                    "nvidia-smi --query-gpu=memory.used,memory.total,"
                    "utilization.gpu,temperature.gpu,power.draw,name "
                    "--format=csv,noheader,nounits", timeout=5)
                if rc == 0 and out.strip():
                    parts = [p.strip() for p in out.strip().split(",")]
                    if len(parts) >= 6:
                        self.latest = {
                            "vram_used": int(parts[0]),
                            "vram_total": int(parts[1]),
                            "gpu_util": int(parts[2]),
                            "temp": int(parts[3]),
                            "power": float(parts[4]) if parts[4].replace('.', '', 1).isdigit() else 0.0,
                            "name": parts[5],
                        }
                        self.history_vram.append(self.latest["vram_used"])
                        self.history_util.append(self.latest["gpu_util"])
                        self.history_temp.append(self.latest["temp"])
            except Exception:
                pass
            self._stop.wait(self.interval)


class AppleSiliconGpuMonitor(BaseGpuMonitor):
    def _poll(self):
        while not self._stop.is_set():
            try:
                proc = subprocess.run(["sysctl", "-n", "hw.memsize"],
                                      capture_output=True, text=True, timeout=5)
                total_bytes = int(proc.stdout.strip()) if proc.returncode == 0 else 0
                total_mb = total_bytes // (1024 * 1024)
                proc2 = subprocess.run(["vm_stat"], capture_output=True, text=True, timeout=5)
                used_mb = 0
                if proc2.returncode == 0:
                    pages = {}
                    for line in proc2.stdout.splitlines():
                        for key in ("Pages active", "Pages wired down", "Pages speculative"):
                            if line.startswith(key):
                                val = line.split(":")[1].strip().rstrip(".")
                                pages[key] = int(val)
                    active = sum(pages.values())
                    used_mb = (active * 16384) // (1024 * 1024)
                self.latest = {
                    "vram_used": used_mb, "vram_total": total_mb,
                    "gpu_util": 0, "temp": 0, "power": 0.0,
                    "name": "Apple Silicon",
                }
                self.history_vram.append(used_mb)
                self.history_util.append(0)
                self.history_temp.append(0)
            except Exception:
                pass
            self._stop.wait(self.interval)


def create_gpu_monitor(platform_info, runner):
    if platform_info.gpu_type == "apple-silicon":
        return AppleSiliconGpuMonitor()
    return NvidiaGpuMonitor(runner)


# ---------------------------------------------------------------------------
# 4. VLLMDetector
# ---------------------------------------------------------------------------

class VLLMDetector:
    def __init__(self, runner, platform_info):
        self.runner = runner
        self.platform = platform_info
        self.venv_path = None
        self.backend = None
        self.version = None

    def detect(self):
        if self.platform.os == "macos":
            self._detect_macos()
        else:
            self._detect_cuda()

    def _detect_cuda(self):
        candidates = [
            os.path.expanduser("~/vllm-install/.vllm"),
            os.path.expanduser("~/local-venv"),
            "/root/vllm-env",
            os.path.expanduser("~/vllm-env"),
            os.path.expanduser("~/.venv/vllm"),
            "/opt/vllm/venv",
        ]
        for venv in candidates:
            rc, out, _ = self.runner.run(
                "test -f %s/bin/activate && source %s/bin/activate && "
                "python -c 'import vllm; print(vllm.__version__)'" % (venv, venv),
                timeout=10)
            if rc == 0 and out.strip():
                self.venv_path = venv
                self.version = out.strip()
                self.backend = "cuda"
                return
        rc, out, _ = self.runner.run(
            "python3 -c 'import vllm; print(vllm.__version__)'", timeout=10)
        if rc == 0 and out.strip():
            self.venv_path = None
            self.version = out.strip()
            self.backend = "cuda"

    def _detect_macos(self):
        candidates = [os.path.expanduser("~/vllm-env")]
        for venv in candidates:
            for mod, backend in [("vllm_metal", "mlx-metal"), ("vllm_mlx", "mlx"), ("vllm", "cuda")]:
                rc, out, _ = self.runner.run(
                    "test -f %s/bin/activate && source %s/bin/activate && "
                    "python -c 'import %s; print(%s.__version__)'" % (venv, venv, mod, mod),
                    timeout=10)
                if rc == 0 and out.strip():
                    self.venv_path = venv
                    self.version = out.strip()
                    self.backend = backend
                    return
        for mod, backend in [("vllm_metal", "mlx-metal"), ("vllm_mlx", "mlx"), ("vllm", "cuda")]:
            rc, out, _ = self.runner.run(
                "python3 -c 'import %s; print(%s.__version__)'" % (mod, mod), timeout=10)
            if rc == 0 and out.strip():
                self.venv_path = None
                self.version = out.strip()
                self.backend = backend
                return

    @property
    def is_installed(self) -> bool:
        return self.backend is not None

    @property
    def activation_cmd(self) -> str:
        return "source %s/bin/activate && " % self.venv_path if self.venv_path else ""

    @property
    def install_dir(self) -> str:
        """Return the parent install directory (containing the venv).
        For DGX: ~/vllm-install  (venv is ~/vllm-install/.vllm)
        For others: ~/vllm-env   (venv IS the install dir)
        """
        if self.venv_path:
            # DGX pattern: ~/vllm-install/.vllm -> return ~/vllm-install
            if self.venv_path.endswith("/.vllm"):
                return self.venv_path[:-6]
            return self.venv_path
        return ""

    @property
    def server_module(self) -> str:
        if self.backend == "mlx-metal":
            return "vllm_metal.server"
        elif self.backend == "mlx":
            return "vllm_mlx.server"
        return "vllm.entrypoints.openai.api_server"


# ---------------------------------------------------------------------------
# 5. Preset models
# ---------------------------------------------------------------------------

def get_preset_models(platform_info):
    if platform_info.is_dgx:
        return [
            {"name": "meta-llama/Llama-3.3-70B-Instruct", "vram": "~70 GB",
             "note": "Llama 3.3 70B (1 nodo DGX)", "chat_template": ""},
            {"name": "Qwen/QwQ-32B", "vram": "~32 GB",
             "note": "QwQ 32B ragionamento", "chat_template": ""},
            {"name": "deepseek-ai/DeepSeek-R1-Distill-Qwen-32B", "vram": "~32 GB",
             "note": "DeepSeek R1 32B", "chat_template": ""},
            {"name": "mistralai/Mixtral-8x22B-v0.1", "vram": "~90 GB",
             "note": "Mixtral 8x22B MoE", "chat_template": ""},
            {"name": "Qwen/Qwen2.5-7B-Instruct", "vram": "~14 GB",
             "note": "Test veloce", "chat_template": ""},
        ]
    elif platform_info.os == "macos":
        return [
            {"name": "mlx-community/Qwen2.5-1.5B-Instruct-4bit", "vram": "~2 GB",
             "note": "Test veloce (MLX 4-bit)", "chat_template": ""},
            {"name": "mlx-community/Qwen2.5-7B-Instruct-4bit", "vram": "~4 GB",
             "note": "Buon bilanciamento (MLX 4-bit)", "chat_template": ""},
            {"name": "mlx-community/Llama-3.2-3B-Instruct-4bit", "vram": "~2 GB",
             "note": "Llama 3.2 compatto (MLX 4-bit)", "chat_template": ""},
        ]
    else:
        return [
            {"name": "Qwen/Qwen2.5-1.5B-Instruct", "vram": "~3 GB",
             "note": "Test veloce", "chat_template": ""},
            {"name": "Qwen/Qwen2.5-7B-Instruct", "vram": "~14 GB",
             "note": "Buon bilanciamento", "chat_template": ""},
            {"name": "meta-llama/Llama-3.1-8B-Instruct", "vram": "~16 GB",
             "note": "Al limite 16 GB", "chat_template": ""},
            {"name": "TheBloke/Mistral-7B-Instruct-v0.2-AWQ", "vram": "~4 GB",
             "note": "Quantizzato AWQ", "chat_template": "mistral"},
            {"name": "TheBloke/Llama-2-13B-GPTQ", "vram": "~8 GB",
             "note": "13B quantizzato GPTQ", "chat_template": "llama-2"},
        ]


DGX_MULTINODE_PRESETS = [
    {"name": "meta-llama/Llama-3.3-70B-Instruct", "nodes": 1, "tp": 1, "vram": "~70 GB"},
    {"name": "meta-llama/Llama-3.1-405B-Instruct-FP8", "nodes": 4, "tp": 4, "vram": "~400 GB"},
    {"name": "Qwen/Qwen2.5-72B-Instruct", "nodes": 1, "tp": 1, "vram": "~72 GB"},
    {"name": "mistralai/Mixtral-8x22B-v0.1", "nodes": 2, "tp": 2, "vram": "~180 GB"},
]


# ---------------------------------------------------------------------------
# 5b. Wizard Parameter Schema
# ---------------------------------------------------------------------------

DGX_WIZARD_PARAMS_SCHEMA = [
    {
        "key": "model_config", "label": "Configurazione Modello",
        "params": [
            {
                "key": "model", "type": "text", "default": "",
                "label": "Modello",
                "tooltip": "ID del modello HuggingFace o percorso locale al modello. "
                           "Esempio: meta-llama/Llama-3.3-70B-Instruct. "
                           "Recuperabile da https://huggingface.co/models filtrando per 'text-generation'.",
                "dgx_tip": "Su DGX Spark (128 GB memoria unificata LPDDR5x CPU+GPU) si possono caricare modelli fino a ~70B in full precision "
                           "o ~140B con quantizzazione FP4. Per multi-nodo (2 Spark) fino a ~140B full o 405B quantizzato.",
                "required": True,
            },
            {
                "key": "tokenizer", "type": "text", "default": "",
                "label": "Tokenizer",
                "tooltip": "Nome o percorso del tokenizer HuggingFace. Se vuoto, usa quello incluso nel modello. "
                           "Specificare solo se il tokenizer e' diverso dal modello (raro).",
                "dgx_tip": "Lasciare vuoto nella quasi totalita' dei casi.",
            },
            {
                "key": "dtype", "type": "select", "default": "auto",
                "options": ["auto", "float16", "bfloat16", "float32"],
                "label": "Tipo Dati (dtype)",
                "tooltip": "Tipo dati per i pesi del modello. 'auto' usa il tipo originale definito nel config del modello. "
                           "float16: 16-bit floating point. bfloat16: Brain Float 16 (migliore range). "
                           "float32: 32-bit, doppia memoria ma massima precisione.",
                "dgx_tip": "Il GPU Blackwell GB10 supporta bfloat16 nativamente. 'auto' e' la scelta consigliata. "
                           "Usare float16 solo se il modello lo richiede esplicitamente.",
            },
            {
                "key": "max_model_len", "type": "number", "default": 0,
                "label": "Lunghezza Max Contesto",
                "tooltip": "Lunghezza massima della sequenza (context window) in token. 0 = automatico dal config del modello. "
                           "Valori comuni: 2048, 4096, 8192, 32768, 131072. "
                           "Ridurre questo valore riduce significativamente l'uso di VRAM.",
                "dgx_tip": "Su DGX Spark con modelli grandi (70B), ridurre a 4096-8192 per risparmiare VRAM. "
                           "Per Llama-3.3-70B il default e' 131072 ma richiede molta memoria. Iniziare con 8192.",
            },
            {
                "key": "trust_remote_code", "type": "checkbox", "default": False,
                "label": "Trust Remote Code",
                "tooltip": "Permette l'esecuzione di codice Python personalizzato contenuto nel repository del modello. "
                           "Necessario per alcuni modelli (Qwen, Phi, DeepSeek) che definiscono architetture custom.",
                "dgx_tip": "Attivare per modelli Qwen, Phi-3, DeepSeek-V2 e altri con architetture custom. "
                           "Attenzione: esegue codice non verificato. Usare solo con modelli fidati.",
            },
            {
                "key": "revision", "type": "text", "default": "",
                "label": "Revisione Modello",
                "tooltip": "Commit hash, branch o tag specifico del modello su HuggingFace. "
                           "Se vuoto, usa la revisione 'main' (ultima disponibile). "
                           "Utile per fissare una versione specifica del modello.",
                "dgx_tip": "Lasciare vuoto per usare l'ultima versione. Specificare un hash solo per riproducibilita'.",
            },
            {
                "key": "quantization", "type": "select", "default": "",
                "options": ["", "awq", "gptq", "fp8", "compressed-tensors", "modelopt_fp4", "marlin", "bitsandbytes", "aqlm", "squeezellm"],
                "label": "Quantizzazione",
                "tooltip": "Metodo di quantizzazione dei pesi. Riduce VRAM a scapito di qualita'. "
                           "AWQ/GPTQ: quantizzazione 4-bit pre-calcolata (il modello deve essere gia' quantizzato). "
                           "FP8: 8-bit floating point. modelopt_fp4: NVIDIA FP4 nativo via ModelOpt (richiede checkpoint FP4). "
                           "compressed-tensors: formato generico per modelli compressi. "
                           "Se vuoto, nessuna quantizzazione (pesi originali).",
                "dgx_tip": "modelopt_fp4 e' supportato nativamente su Blackwell (sm_121) ed offre il miglior rapporto qualita'/compressione. "
                           "FP8 e' un buon compromesso. Per modelli >70B su singolo Spark, considerare fp8 o modelopt_fp4.",
            },
            {
                "key": "enforce_eager", "type": "checkbox", "default": False,
                "label": "Forza Eager Mode",
                "tooltip": "Disabilita CUDA graph capturing e forza l'esecuzione eager. "
                           "CUDA graphs accelerano l'inferenza ma usano piu' memoria e possono causare problemi di compatibilita'. "
                           "Utile per debugging o quando si verificano errori CUDA.",
                "dgx_tip": "Lasciare disattivato in produzione. Attivare temporaneamente se si verificano "
                           "errori di CUDA graph o comportamenti anomali.",
            },
            {
                "key": "seed", "type": "number", "default": 0,
                "label": "Seed Random",
                "tooltip": "Seed per il generatore di numeri random. 0 = seed casuale ad ogni avvio. "
                           "Impostare un valore fisso per ottenere risultati riproducibili.",
                "dgx_tip": "Lasciare 0 per uso normale. Impostare un valore fisso solo per test di riproducibilita'.",
            },
        ],
    },
    {
        "key": "memory_perf", "label": "Memoria & Performance",
        "params": [
            {
                "key": "gpu_memory_utilization", "type": "range", "default": 0.90,
                "min": 0.5, "max": 0.99, "step": 0.01,
                "label": "Utilizzo Memoria GPU",
                "tooltip": "Frazione della memoria GPU da dedicare alla KV cache (0.5 - 0.99). "
                           "Il resto e' usato per i pesi del modello e overhead. "
                           "Valori alti = piu' sequenze in parallelo. Troppo alto = rischio OOM.",
                "dgx_tip": "Su DGX Spark: 0.90 e' bilanciato. Aumentare a 0.95 per modelli grandi che servono piu' cache. "
                           "Ridurre a 0.80 se si verificano errori Out-Of-Memory.",
            },
            {
                "key": "cpu_offload_gb", "type": "number", "default": 0,
                "label": "CPU Offload (GB)",
                "tooltip": "Gigabyte di pesi del modello da scaricare sulla RAM CPU. 0 = tutto su GPU. "
                           "Permette di caricare modelli piu' grandi della VRAM a scapito della velocita'. "
                           "Richiede RAM sufficiente.",
                "dgx_tip": "DGX Spark ha 128 GB di memoria unificata (UMA). Il CPU offload puo' essere utile "
                           "per modelli molto grandi, ma su architettura UMA l'impatto e' minore rispetto a sistemi discreti.",
            },
            {
                "key": "kv_cache_dtype", "type": "select", "default": "auto",
                "options": ["auto", "fp8", "fp8_e5m2", "fp8_e4m3"],
                "label": "Tipo Dati KV Cache",
                "tooltip": "Tipo dati per la Key-Value cache. 'auto' usa lo stesso tipo dei pesi del modello. "
                           "FP8 dimezza la memoria della KV cache, permettendo piu' sequenze in parallelo. "
                           "fp8_e4m3: 4 bit mantissa (piu' preciso). fp8_e5m2: 5 bit esponente (piu' range).",
                "dgx_tip": "Su Blackwell, fp8_e4m3 e' supportato nativamente e offre un buon compromesso. "
                           "Usare fp8 per raddoppiare il throughput con modelli grandi.",
            },
            {
                "key": "block_size", "type": "select", "default": 16,
                "options": [8, 16, 32],
                "label": "Dimensione Blocco KV Cache",
                "tooltip": "Numero di token per blocco nella KV cache (PagedAttention). "
                           "Blocchi piu' grandi = meno overhead di gestione, ma piu' spreco di memoria interna. "
                           "Valori: 8, 16, 32.",
                "dgx_tip": "16 e' il default ottimale per la maggior parte dei casi su DGX Spark.",
            },
            {
                "key": "enable_prefix_caching", "type": "checkbox", "default": False,
                "label": "Prefix Caching",
                "tooltip": "Riutilizza la KV cache per prefissi di prompt identici tra richieste diverse. "
                           "Migliora significativamente il throughput quando piu' richieste condividono lo stesso system prompt "
                           "o contesto iniziale. In vLLM il default e' disattivato (auto-detect).",
                "dgx_tip": "Consigliato attivo per ambienti con system prompt condivisi. Particolarmente utile con RAG.",
            },
            {
                "key": "calculate_kv_scales", "type": "checkbox", "default": False,
                "label": "Calcola Scale KV",
                "tooltip": "Calcola automaticamente i fattori di scala per la KV cache quantizzata. "
                           "Necessario quando kv_cache_dtype e' impostato su fp8. "
                           "Aggiunge un piccolo overhead al primo avvio.",
                "dgx_tip": "Attivare se si usa kv_cache_dtype=fp8. Non necessario con 'auto'.",
                "advanced": True,
            },
        ],
    },
    {
        "key": "parallelism", "label": "Parallelismo",
        "params": [
            {
                "key": "tensor_parallel_size", "type": "number", "default": 1,
                "min": 1, "max": 8,
                "label": "Tensor Parallel Size",
                "tooltip": "Numero di GPU su cui suddividere i pesi del modello (tensor parallelism). "
                           "Ogni GPU elabora una porzione dei tensori. Richiede Ray per valori > 1 su multi-nodo. "
                           "Deve dividere equamente il numero di attention head del modello.",
                "dgx_tip": "DGX Spark ha 1 GPU per nodo. Usare 1 per singolo nodo, 2 per due Spark collegati via QSFP. "
                           "Per 4 nodi: tp_size=4. Richiede Ray cluster configurato.",
            },
            {
                "key": "pipeline_parallel_size", "type": "number", "default": 1,
                "min": 1, "max": 8,
                "label": "Pipeline Parallel Size",
                "tooltip": "Numero di stadi di pipeline parallelism. Distribuisce i layer del modello su piu' GPU in sequenza. "
                           "Complementare al tensor parallelism. tp_size * pp_size = totale GPU. "
                           "Pipeline parallelism e' meno efficiente ma supporta piu' GPU.",
                "dgx_tip": "Usare per modelli enormi (405B+) quando il tensor parallelism da solo non basta. "
                           "Preferire tensor parallelism quando possibile.",
            },
            {
                "key": "data_parallel_size", "type": "number", "default": 1,
                "min": 1, "max": 8,
                "label": "Data Parallel Size",
                "tooltip": "Numero di repliche del modello per data parallelism. "
                           "Ogni replica gestisce richieste indipendentemente, moltiplicando il throughput. "
                           "Richiede dp_size * tp_size * pp_size GPU totali.",
                "dgx_tip": "Utile solo con molti nodi. Su 2 Spark collegati, preferire tp_size=2 piuttosto che dp_size=2.",
                "advanced": True,
            },
            {
                "key": "distributed_executor_backend", "type": "select", "default": "",
                "options": ["", "ray", "mp"],
                "label": "Backend Esecuzione Distribuita",
                "tooltip": "Backend per l'esecuzione distribuita. Vuoto = automatico (vLLM sceglie in base alla configurazione). "
                           "'ray': usa Ray per orchestrazione multi-nodo (richiede Ray installato e cluster attivo). "
                           "'mp': usa multiprocessing Python (solo singolo nodo, piu' GPU).",
                "dgx_tip": "Per multi-nodo DGX Spark: usare 'ray' (richiede cluster Ray configurato nel tab DGX Spark). "
                           "Per singolo nodo: lasciare vuoto o 'mp'.",
            },
            {
                "key": "enable_expert_parallel", "type": "checkbox", "default": False,
                "label": "Expert Parallelism",
                "tooltip": "Abilita parallelismo degli esperti per modelli Mixture-of-Experts (MoE). "
                           "Distribuisce i diversi 'esperti' del modello su GPU separate. "
                           "Applicabile solo a modelli MoE come Mixtral, DeepSeek-V2, DBRX.",
                "dgx_tip": "Attivare per Mixtral-8x22B, DeepSeek-V2-MoE e simili. "
                           "Non ha effetto su modelli densi (Llama, Qwen standard).",
                "advanced": True,
            },
            {
                "key": "max_parallel_loading_workers", "type": "number", "default": 0,
                "min": 0, "max": 16,
                "label": "Worker Caricamento Parallelo",
                "tooltip": "Numero di worker paralleli per caricare i pesi del modello. 0 = automatico. "
                           "Valori piu' alti velocizzano il caricamento su storage veloci (NVMe). "
                           "Usa piu' RAM durante il caricamento.",
                "dgx_tip": "Lasciare 0 per la maggior parte dei casi. Aumentare a 4-8 se il caricamento e' lento.",
                "advanced": True,
            },
            {
                "key": "nnodes", "type": "number", "default": 1,
                "min": 1, "max": 8,
                "label": "Numero Nodi",
                "tooltip": "Numero totale di nodi nel cluster distribuito. "
                           "Deve corrispondere al numero di macchine DGX Spark collegate. "
                           "Richiede Ray cluster attivo con tutti i nodi connessi.",
                "dgx_tip": "Impostare al numero di DGX Spark collegati. 1 = singolo nodo. "
                           "2 = due Spark con QSFP. Verificare che tutti i worker siano in stato 'Ray Connesso'.",
            },
        ],
    },
    {
        "key": "serving", "label": "Serving & Scheduling",
        "params": [
            {
                "key": "host", "type": "text", "default": "0.0.0.0",
                "label": "Host",
                "tooltip": "Indirizzo IP su cui il server ascolta. '0.0.0.0' accetta connessioni da qualsiasi IP. "
                           "'127.0.0.1' solo connessioni locali. Specificare un IP per limitare l'accesso.",
                "dgx_tip": "Usare '0.0.0.0' per rendere il server accessibile dalla rete locale.",
            },
            {
                "key": "port", "type": "number", "default": 8000,
                "min": 1024, "max": 65535,
                "label": "Porta",
                "tooltip": "Porta HTTP su cui il server espone l'API compatibile OpenAI. "
                           "L'API sara' disponibile su http://<host>:<port>/v1/chat/completions. "
                           "Evitare porte gia' in uso (5000 = manager, 8265 = Ray dashboard).",
                "dgx_tip": "8000 e' il default standard. Cambiare se si avviano piu' istanze vLLM sullo stesso nodo.",
            },
            {
                "key": "max_num_seqs", "type": "number", "default": 0,
                "label": "Max Sequenze in Batch",
                "tooltip": "Numero massimo di sequenze elaborate contemporaneamente in un batch. "
                           "0 = automatico (vLLM decide in base al modello e alla memoria disponibile). "
                           "Valori alti = piu' throughput ma piu' latenza per singola richiesta. "
                           "Valori bassi = meno throughput ma latenza piu' bassa.",
                "dgx_tip": "Per modelli grandi (70B): ridurre a 32-64. Per modelli piccoli (7B): 128-256. "
                           "Lasciare 0 per calcolo automatico.",
            },
            {
                "key": "max_num_batched_tokens", "type": "number", "default": 0,
                "label": "Max Token per Batch",
                "tooltip": "Numero massimo di token elaborati in una singola iterazione del motore. "
                           "0 = automatico (calcolato da max_model_len e max_num_seqs). "
                           "Controlla il bilanciamento throughput vs latenza.",
                "dgx_tip": "Lasciare 0 per calcolo automatico. Ridurre se si verificano spike di latenza.",
                "advanced": True,
            },
            {
                "key": "enable_chunked_prefill", "type": "checkbox", "default": False,
                "label": "Chunked Prefill",
                "tooltip": "Divide il prefill di prompt lunghi in chunk piu' piccoli. "
                           "Riduce la latenza del primo token (TTFT) per prompt molto lunghi "
                           "permettendo ad altre richieste di essere servite durante il prefill.",
                "dgx_tip": "Consigliato per servire prompt lunghi (>4K token) con bassa latenza. "
                           "Leggero overhead sul throughput totale.",
            },
            {
                "key": "chat_template", "type": "text", "default": "",
                "label": "Template Chat",
                "tooltip": "Percorso a un file template Jinja2 per formattare i messaggi di chat, oppure il nome "
                           "di un template registrato nel tokenizer. Vuoto = auto-detect dal tokenizer del modello. "
                           "vLLM accetta: percorso file .jinja, stringa template inline, o nome template dal tokenizer. "
                           "La maggior parte dei modelli moderni include il template nel tokenizer.",
                "dgx_tip": "Lasciare vuoto (auto) per la maggior parte dei modelli recenti. "
                           "Specificare solo se il modello non include un template o se si vuole forzarne uno diverso.",
            },
            {
                "key": "scheduling_policy", "type": "select", "default": "fcfs",
                "options": ["fcfs", "priority"],
                "label": "Politica Scheduling",
                "tooltip": "Politica di scheduling delle richieste. "
                           "'fcfs' (First Come First Served): le richieste sono elaborate nell'ordine di arrivo. "
                           "'priority': le richieste possono avere priorita' diverse (richiede API priority).",
                "dgx_tip": "FCFS e' adatto per la maggior parte degli scenari. "
                           "Priority utile per ambienti multi-utente con SLA diversi.",
                "advanced": True,
            },
        ],
    },
    {
        "key": "attention", "label": "Attention & Backend",
        "params": [
            {
                "key": "attention_backend", "type": "select", "default": "",
                "options": ["", "FLASH_ATTN", "FLASHINFER", "TORCH_SDPA", "XFORMERS"],
                "label": "Backend Attention",
                "tooltip": "Implementazione del meccanismo di attenzione. Vuoto = auto-select. "
                           "FLASH_ATTN: Flash Attention 2/3. FLASHINFER: ottimizzato per throughput. "
                           "TORCH_SDPA: PyTorch scaled dot-product attention. XFORMERS: libreria xFormers di Meta. "
                           "Configurato via env var VLLM_ATTENTION_BACKEND o --attention-config.",
                "dgx_tip": "Su DGX Spark (Blackwell sm_121): FLASH_ATTN o FLASHINFER sono consigliati. "
                           "Se non funzionano, provare TORCH_SDPA come fallback. "
                           "Nota: impostato via variabile d'ambiente VLLM_ATTENTION_BACKEND, non come flag CLI diretto.",
            },
            {
                "key": "disable_cascade_attn", "type": "checkbox", "default": True,
                "label": "Disabilita Cascade Attention",
                "tooltip": "Disabilita l'ottimizzazione cascade attention che separa il computo tra prefix cache "
                           "e token nuovi. In vLLM il default e' disabilitato (True). "
                           "Abilitare cascade attention puo' migliorare le performance con prefix caching attivo.",
                "dgx_tip": "Il default vLLM e' disabilitato. Provare a disattivare (cascade attivo) se si usa prefix caching.",
                "advanced": True,
            },
            {
                "key": "disable_sliding_window", "type": "checkbox", "default": False,
                "label": "Disabilita Sliding Window",
                "tooltip": "Disabilita l'attenzione a finestra scorrevole (sliding window attention). "
                           "Alcuni modelli (Mistral, Phi) usano SWA per ridurre la complessita'. "
                           "Disabilitare forza full attention su tutta la sequenza.",
                "dgx_tip": "Lasciare disattivato. Attivare solo se richiesto per compatibilita' specifica.",
                "advanced": True,
            },
        ],
    },
    {
        "key": "lora", "label": "LoRA",
        "params": [
            {
                "key": "enable_lora", "type": "checkbox", "default": False,
                "label": "Abilita LoRA",
                "tooltip": "Abilita il caricamento di adattatori LoRA (Low-Rank Adaptation) a runtime. "
                           "Permette di servire modelli fine-tuned senza duplicare i pesi base. "
                           "Richiede adattatori LoRA compatibili con il modello base.",
                "dgx_tip": "Utile per servire varianti fine-tuned dello stesso modello base. "
                           "Aggiunge un piccolo overhead di memoria per ogni adattatore caricato.",
                "advanced": True,
            },
            {
                "key": "max_loras", "type": "number", "default": 1,
                "min": 1, "max": 64,
                "label": "Max Adattatori LoRA",
                "tooltip": "Numero massimo di adattatori LoRA che possono essere caricati simultaneamente. "
                           "Ogni adattatore aggiunge memoria proporzionale al suo rank.",
                "dgx_tip": "Aumentare se si servono molti utenti con adattatori diversi.",
                "advanced": True,
            },
            {
                "key": "max_lora_rank", "type": "number", "default": 16,
                "min": 1, "max": 256,
                "label": "Max LoRA Rank",
                "tooltip": "Rank massimo supportato per gli adattatori LoRA. Valori comuni: 8, 16, 32, 64. "
                           "Rank piu' alto = adattatore piu' espressivo ma piu' memoria.",
                "dgx_tip": "16 e' sufficiente per la maggior parte dei casi. Aumentare a 32-64 per adattatori complessi.",
                "advanced": True,
            },
        ],
    },
    {
        "key": "speculative", "label": "Speculative Decoding",
        "params": [
            {
                "key": "speculative_model", "type": "text", "default": "",
                "label": "Modello Draft",
                "tooltip": "ID HuggingFace del modello 'draft' per speculative decoding. "
                           "Deve essere un modello piu' piccolo e veloce dello stesso tipo (es. 1B per un modello 70B). "
                           "Se vuoto, speculative decoding e' disabilitato. "
                           "Passato a vLLM via --speculative-config JSON (i flag separati sono deprecati).",
                "dgx_tip": "Esempio: per Llama-3.3-70B usare meta-llama/Llama-3.2-1B-Instruct come draft. "
                           "Puo' aumentare il throughput del 30-50% per generazione lunga.",
                "advanced": True,
            },
            {
                "key": "num_speculative_tokens", "type": "number", "default": 0,
                "min": 0, "max": 20,
                "label": "Token Speculativi",
                "tooltip": "Numero di token generati speculativamente dal modello draft per ogni iterazione. "
                           "0 = disabilitato. Valori tipici: 3-5. "
                           "Piu' alto = piu' speedup potenziale ma piu' token da verificare. "
                           "Passato a vLLM via --speculative-config JSON.",
                "dgx_tip": "3-5 e' un buon compromesso. Valori troppo alti riducono il tasso di accettazione.",
                "advanced": True,
            },
        ],
    },
    {
        "key": "observability", "label": "Osservabilita'",
        "params": [
            {
                "key": "disable_log_stats", "type": "checkbox", "default": False,
                "label": "Disabilita Statistiche Log",
                "tooltip": "Disabilita la stampa di statistiche periodiche nei log (throughput, latenza, code). "
                           "Riduce il volume dei log in produzione. Flag CLI: --disable-log-stats.",
                "dgx_tip": "Lasciare disattivato durante il setup per monitorare le performance. "
                           "Attivare in produzione se i log sono troppo verbosi.",
                "advanced": True,
            },
        ],
    },
]


# ---------------------------------------------------------------------------
# 6. Config
# ---------------------------------------------------------------------------

def get_config_dir() -> Path:
    if sys.platform == "win32":
        base = Path(os.environ.get("APPDATA", Path.home()))
        return base / "vLLM Manager"
    elif sys.platform == "darwin":
        return Path.home() / "Library" / "Application Support" / "vLLM Manager"
    else:
        return Path.home() / ".config" / "vllm-manager"


CONFIG_DIR = get_config_dir()
CONFIG_FILE = CONFIG_DIR / "config.json"


class ConfigManager:
    def __init__(self):
        CONFIG_DIR.mkdir(parents=True, exist_ok=True)
        self.data = self._load()

    def _load(self):
        if CONFIG_FILE.exists():
            try:
                return json.loads(CONFIG_FILE.read_text("utf-8"))
            except Exception:
                pass
        return {"profiles": {}, "settings": {}, "workers": {}}

    def save(self):
        CONFIG_FILE.write_text(json.dumps(self.data, indent=2), "utf-8")

    def get_profiles(self):
        return self.data.setdefault("profiles", {})

    def save_profile(self, name, profile):
        self.data.setdefault("profiles", {})[name] = profile
        self.save()

    def delete_profile(self, name):
        self.data.get("profiles", {}).pop(name, None)
        self.save()

    # --- Worker management ---

    def get_workers(self):
        return self.data.setdefault("workers", {})

    def save_worker(self, ip, worker):
        self.data.setdefault("workers", {})[ip] = worker
        self.save()

    def delete_worker(self, ip):
        self.data.get("workers", {}).pop(ip, None)
        self.save()

    def update_worker_status(self, ip, status):
        w = self.data.get("workers", {}).get(ip)
        if w:
            w["status"] = status
            self.save()

    # --- Wizard state ---

    def get_wizard(self):
        return self.data.setdefault("wizard", {})

    def save_wizard_step(self, step_key, step_data):
        wiz = self.data.setdefault("wizard", {})
        wiz[step_key] = step_data
        self.save()

    def reset_wizard(self):
        self.data["wizard"] = {}
        self.save()


# ---------------------------------------------------------------------------
# 7. Docker
# ---------------------------------------------------------------------------

def get_docker_cmd() -> str:
    return "docker.exe" if sys.platform == "win32" else "docker"


def docker_run(args, timeout=30):
    docker_cmd = get_docker_cmd()
    try:
        flags = {}
        if sys.platform == "win32":
            flags["creationflags"] = 0x08000000
        proc = subprocess.run([docker_cmd] + args,
                              capture_output=True, text=True, timeout=timeout, **flags)
        return proc.returncode, proc.stdout, proc.stderr
    except Exception as e:
        return -1, "", str(e)


# ---------------------------------------------------------------------------
# Backend: VLLMProcess
# ---------------------------------------------------------------------------

class VLLMProcess:
    def __init__(self, runner, platform_info, detector, log_queue):
        self.runner = runner
        self.platform = platform_info
        self.detector = detector
        self.log_queue = log_queue
        self.proc = None
        self._reader_thread = None
        self.running = False

    def start(self, model, gpu_mem_util=0.90, max_model_len=None,
              dtype="auto", enable_prefix_caching=True, extra_args="",
              host="0.0.0.0", port=8000, chat_template="",
              tp_size=1, attention_backend="",
              # Wizard params
              pp_size=1, dp_size=None, max_num_seqs=None,
              max_num_batched_tokens=None, enable_chunked_prefill=False,
              kv_cache_dtype=None, block_size=None, cpu_offload_gb=None,
              trust_remote_code=False, enforce_eager=False,
              quantization=None, seed=None, tokenizer=None,
              enable_lora=False, max_loras=None, max_lora_rank=None,
              speculative_model=None, num_speculative_tokens=None,
              distributed_executor_backend=None, nnodes=1,
              scheduling_policy=None, disable_log_stats=False,
              disable_cascade_attn=True, disable_sliding_window=False,
              enable_expert_parallel=False):
        if self.running:
            return

        tpl_content = CHAT_TEMPLATES.get(chat_template, "")
        if tpl_content:
            self.runner.run(
                "mkdir -p /tmp/vllm-manager && cat > /tmp/vllm-manager/chat_template.jinja << 'ENDTPL'\n"
                "%s\nENDTPL" % tpl_content, timeout=5)

        parts = []
        if self.detector.activation_cmd:
            parts.append(self.detector.activation_cmd)
        parts.append("python -m %s" % self.detector.server_module)
        parts.append("--model '%s'" % model)
        parts.append("--host %s" % host)
        parts.append("--port %s" % port)

        if self.detector.backend == "cuda":
            parts.append("--gpu-memory-utilization %s" % gpu_mem_util)
            parts.append("--dtype %s" % dtype)
            if tpl_content:
                parts.append("--chat-template /tmp/vllm-manager/chat_template.jinja")
            if max_model_len:
                parts.append("--max-model-len %s" % max_model_len)
            if enable_prefix_caching:
                parts.append("--enable-prefix-caching")
            if tp_size > 1:
                parts.append("--tensor-parallel-size %d" % tp_size)
            if pp_size and pp_size > 1:
                parts.append("--pipeline-parallel-size %d" % pp_size)
            if dp_size and dp_size > 1:
                parts.append("--data-parallel-size %d" % dp_size)
            if max_num_seqs:
                parts.append("--max-num-seqs %d" % max_num_seqs)
            if max_num_batched_tokens:
                parts.append("--max-num-batched-tokens %d" % max_num_batched_tokens)
            if enable_chunked_prefill:
                parts.append("--enable-chunked-prefill")
            if kv_cache_dtype and kv_cache_dtype != "auto":
                parts.append("--kv-cache-dtype %s" % kv_cache_dtype)
            if block_size and block_size != 16:
                parts.append("--block-size %d" % block_size)
            if cpu_offload_gb and cpu_offload_gb > 0:
                parts.append("--cpu-offload-gb %.1f" % cpu_offload_gb)
            if trust_remote_code:
                parts.append("--trust-remote-code")
            if enforce_eager:
                parts.append("--enforce-eager")
            if quantization:
                parts.append("--quantization %s" % quantization)
            if seed and seed > 0:
                parts.append("--seed %d" % seed)
            if tokenizer:
                parts.append("--tokenizer '%s'" % tokenizer)
            if enable_lora:
                parts.append("--enable-lora")
                if max_loras and max_loras > 1:
                    parts.append("--max-loras %d" % max_loras)
                if max_lora_rank and max_lora_rank != 16:
                    parts.append("--max-lora-rank %d" % max_lora_rank)
            if speculative_model:
                import json as _json
                spec_cfg = {"model": speculative_model, "method": "draft_model"}
                if num_speculative_tokens and num_speculative_tokens > 0:
                    spec_cfg["num_speculative_tokens"] = num_speculative_tokens
                parts.append("--speculative-config '%s'" % _json.dumps(spec_cfg))
            if distributed_executor_backend and distributed_executor_backend not in ("", "auto"):
                parts.append("--distributed-executor-backend %s" % distributed_executor_backend)
            if nnodes and nnodes > 1:
                parts.append("--nnodes %d" % nnodes)
            if scheduling_policy and scheduling_policy != "fcfs":
                parts.append("--scheduling-policy %s" % scheduling_policy)
            if disable_log_stats:
                parts.append("--disable-log-stats")
            if disable_cascade_attn:
                parts.append("--disable-cascade-attn")
            if disable_sliding_window:
                parts.append("--disable-sliding-window")
            if enable_expert_parallel:
                parts.append("--enable-expert-parallel")

        if extra_args.strip():
            parts.append(extra_args.strip())

        cmd = " ".join(parts)
        self.log_queue.put("[CMD] %s\n" % cmd)

        env = None
        if self.platform.is_dgx:
            env = {}
            if isinstance(self.runner, DGXCommandRunner):
                env.update(self.runner.dgx_env)
            # Blackwell sm_121 env vars
            env["TORCH_CUDA_ARCH_LIST"] = "12.1a"
            env["VLLM_USE_FLASHINFER_MXFP4_MOE"] = "1"
            env["TRITON_PTXAS_PATH"] = "/usr/local/cuda/bin/ptxas"
            if attention_backend:
                env["VLLM_ATTENTION_BACKEND"] = attention_backend

        self.proc = self.runner.popen(cmd, env=env)
        self.running = True
        self._reader_thread = threading.Thread(target=self._read_output, daemon=True)
        self._reader_thread.start()

    def stop(self):
        if not self.running or self.proc is None:
            return
        self.log_queue.put("[INFO] Stopping vLLM server...\n")
        kill_cmd = (
            "pkill -9 -f 'vllm.entrypoints.openai.api_server'; "
            "pkill -9 -f 'vllm_metal.server'; "
            "pkill -9 -f 'vllm_mlx.server'; "
            "pkill -9 -f 'from multiprocessing.resource_tracker'; "
            "sleep 0.5"
        )
        self.runner.run_raw(kill_cmd, timeout=10)
        try:
            self.proc.kill()
            self.proc.wait(timeout=5)
        except Exception:
            pass
        self.running = False
        self.proc = None
        self.log_queue.put("[INFO] vLLM server stopped.\n")

    def _read_output(self):
        try:
            for line in self.proc.stdout:
                self.log_queue.put(line)
            self.proc.stdout.close()
        except Exception:
            pass
        finally:
            self.running = False


# ---------------------------------------------------------------------------
# Backend: VLLMApi
# ---------------------------------------------------------------------------

class VLLMApi:
    def __init__(self, base_url="http://localhost:8000"):
        self.base_url = base_url

    def health(self) -> bool:
        try:
            req = request.Request(self.base_url + "/health", method="GET")
            with request.urlopen(req, timeout=3) as resp:
                return resp.status == 200
        except Exception:
            return False

    def list_models(self) -> list:
        try:
            req = request.Request(self.base_url + "/v1/models", method="GET")
            with request.urlopen(req, timeout=5) as resp:
                data = json.loads(resp.read())
                return [m["id"] for m in data.get("data", [])]
        except Exception:
            return []

    def chat(self, model, messages, max_tokens=256, temperature=0.7, stream=False):
        body = json.dumps({
            "model": model, "messages": messages,
            "max_tokens": max_tokens, "temperature": temperature,
            "stream": stream,
        }).encode()
        req = request.Request(
            self.base_url + "/v1/chat/completions",
            data=body, method="POST",
            headers={"Content-Type": "application/json"},
        )
        try:
            with request.urlopen(req, timeout=120) as resp:
                if stream:
                    return self._read_stream(resp)
                return json.loads(resp.read())
        except Exception as e:
            return {"error": str(e)}

    def _read_stream(self, resp):
        tokens = []
        ttft = None
        t0 = time.perf_counter()
        for raw_line in resp:
            line = raw_line.decode("utf-8", errors="replace").strip()
            if not line.startswith("data: "):
                continue
            payload = line[6:]
            if payload == "[DONE]":
                break
            try:
                chunk = json.loads(payload)
                delta = chunk["choices"][0].get("delta", {})
                content = delta.get("content", "")
                if content:
                    if ttft is None:
                        ttft = time.perf_counter() - t0
                    tokens.append(content)
            except Exception:
                continue
        elapsed = time.perf_counter() - t0
        return {
            "text": "".join(tokens),
            "tokens_count": len(tokens),
            "elapsed": elapsed,
            "ttft": ttft,
        }


# ---------------------------------------------------------------------------
# Global state
# ---------------------------------------------------------------------------

platform_info = PlatformInfo()
runner = create_command_runner(platform_info)
log_queue = queue.Queue()
detector = VLLMDetector(runner, platform_info)
vllm_proc = VLLMProcess(runner, platform_info, detector, log_queue)
api = VLLMApi()
config = ConfigManager()
gpu_mon = create_gpu_monitor(platform_info, runner)
preset_models = get_preset_models(platform_info)

# Log buffer for SSE streaming
log_buffer = deque(maxlen=2000)
log_buffer_lock = threading.Lock()


def drain_log_queue():
    """Background thread: move log_queue items to log_buffer."""
    while True:
        try:
            line = log_queue.get(timeout=0.5)
            with log_buffer_lock:
                log_buffer.append(line)
        except queue.Empty:
            pass


# ---------------------------------------------------------------------------
# Flask app
# ---------------------------------------------------------------------------

app = Flask(__name__)


@app.route("/")
def index():
    return render_template_string(HTML_TEMPLATE,
                                  platform=platform_info,
                                  presets=preset_models,
                                  multinode_presets=DGX_MULTINODE_PRESETS,
                                  is_dgx=platform_info.is_dgx,
                                  is_macos=platform_info.os == "macos",
                                  app_version=APP_VERSION,
                                  chat_templates=["(auto)"] + [k for k in CHAT_TEMPLATES if k],
                                  dgx_vllm_commit=DGX_VLLM_COMMIT,
                                  dgx_triton_commit=DGX_TRITON_COMMIT,
                                  dgx_pytorch_index=DGX_PYTORCH_INDEX,
                                  dgx_install_dir=DGX_INSTALL_DIR_DEFAULT)


# --- API: Server control ---

@app.route("/api/server/start", methods=["POST"])
def api_start_server():
    data = flask_request.json or {}
    model = data.get("model", "").strip()
    if not model:
        return jsonify({"error": "Modello mancante"}), 400

    def _run():
        if not detector.is_installed:
            log_queue.put("[INFO] Searching for vLLM installation...\n")
            detector.detect()
        if not detector.is_installed:
            log_queue.put("[ERROR] vLLM non trovato. Installalo prima.\n")
            return
        log_queue.put("[INFO] vLLM %s detected (backend: %s)\n" % (
            detector.version, detector.backend))

        chat_tpl = data.get("chat_template", "")
        if chat_tpl == "(auto)":
            chat_tpl = ""
        attn_backend = data.get("attention_backend", "")
        if attn_backend == "(auto)":
            attn_backend = ""
        max_len = int(data.get("max_model_len", 0))

        vllm_proc.start(
            model=model,
            gpu_mem_util=float(data.get("gpu_mem_util", 0.90)),
            max_model_len=max_len if max_len > 0 else None,
            dtype=data.get("dtype", "auto"),
            enable_prefix_caching=data.get("prefix_caching", True),
            extra_args=data.get("extra_args", ""),
            host=data.get("host", "0.0.0.0"),
            port=int(data.get("port", 8000)),
            chat_template=chat_tpl,
            tp_size=int(data.get("tp_size", 1)),
            attention_backend=attn_backend,
        )

    threading.Thread(target=_run, daemon=True).start()
    return jsonify({"status": "starting"})


@app.route("/api/server/stop", methods=["POST"])
def api_stop_server():
    def _run():
        vllm_proc.stop()
    threading.Thread(target=_run, daemon=True).start()
    return jsonify({"status": "stopping"})


@app.route("/api/server/stop-all", methods=["POST"])
def api_stop_all():
    def _run():
        if vllm_proc.running:
            vllm_proc.stop()
        else:
            runner.run_raw(
                "pkill -9 -f 'vllm.entrypoints.openai.api_server'; "
                "pkill -9 -f 'vllm_metal.server'; "
                "pkill -9 -f 'vllm_mlx.server'; "
                "pkill -9 -f 'from multiprocessing.resource_tracker'",
                timeout=5)
        log_queue.put("[INFO] vLLM killed.\n")
        # Stop Docker
        docker_run(["stop", "open-webui"], timeout=15)
        log_queue.put("[INFO] Docker container stopped.\n")
        if platform_info.os == "windows-wsl":
            runner.shutdown()
            log_queue.put("[INFO] WSL terminated.\n")
    threading.Thread(target=_run, daemon=True).start()
    return jsonify({"status": "stopping_all"})


@app.route("/api/server/status")
def api_server_status():
    online = api.health()
    return jsonify({
        "online": online,
        "running": vllm_proc.running,
        "vllm_installed": detector.is_installed,
        "vllm_version": detector.version,
        "vllm_backend": detector.backend,
    })


@app.route("/api/server/install", methods=["POST"])
def api_install_vllm():
    data = flask_request.json or {}
    force = data.get("force", False)
    def _install():
        if platform_info.is_dgx:
            _install_vllm_dgx(force=force)
        elif platform_info.os == "macos":
            _install_vllm_simple([
                ("Creazione virtualenv...",
                 "%s -m venv %s" % (sys.executable, os.path.expanduser("~/vllm-env"))),
                ("Installazione vllm-mlx...",
                 "source ~/vllm-env/bin/activate && pip install --upgrade pip && pip install vllm-mlx"),
            ])
        else:
            _install_vllm_simple([
                ("Creazione virtualenv...",
                 "%s -m venv %s" % (sys.executable, os.path.expanduser("~/vllm-env"))),
                ("Installazione vLLM...",
                 "source ~/vllm-env/bin/activate && pip install --upgrade pip && pip install vllm"),
            ])

    threading.Thread(target=_install, daemon=True).start()
    return jsonify({"status": "installing"})


def _install_vllm_simple(steps):
    """Simple pip-based install for Linux/macOS/WSL."""
    for msg, cmd in steps:
        log_queue.put("[INSTALL] %s\n" % msg)
        rc, out, err = runner.run(cmd, timeout=600)
        if rc != 0:
            log_queue.put("[ERROR] %s\n" % (err.strip() or out.strip() or "exit code %d" % rc))
            return
        if out.strip():
            for line in out.strip().splitlines()[-3:]:
                log_queue.put("[INSTALL]   %s\n" % line)
    log_queue.put("[INSTALL] Installazione completata!\n")
    detector.detect()


# ---------------------------------------------------------------------------
# DGX Spark full installation (based on eelbaz/dgx-spark-vllm-setup)
# Handles: uv, PyTorch cu130, Triton from source, vLLM patched for sm_121
# ---------------------------------------------------------------------------

# Pinned versions tested on DGX Spark GB10
DGX_VLLM_COMMIT = "66a168a197ba214a5b70a74fa2e713c9eeb3251a"
DGX_TRITON_COMMIT = "4caa0328bf8df64896dd5f6fb9df41b0eb2e750a"
DGX_PYTORCH_INDEX = "https://download.pytorch.org/whl/cu130"
DGX_INSTALL_DIR_DEFAULT = os.path.expanduser("~/vllm-install")


def _dgx_run_step(msg, cmd, timeout=1800):
    """Run a single install step, log output, return True on success."""
    log_queue.put("[INSTALL] %s\n" % msg)
    rc, out, err = runner.run(cmd, timeout=timeout)
    if out.strip():
        for line in out.strip().splitlines()[-5:]:
            log_queue.put("[INSTALL]   %s\n" % line)
    if rc != 0:
        error = err.strip() or out.strip() or "exit code %d" % rc
        log_queue.put("[ERROR] %s\n" % error)
        return False
    return True


def _install_vllm_dgx(vllm_commit=None, triton_commit=None, pytorch_index=None, install_dir=None, force=False):
    """Full DGX Spark installation: uv, PyTorch, Triton, patched vLLM.
    If force=False and a working installation is found, skip."""
    vllm_commit = vllm_commit or DGX_VLLM_COMMIT
    triton_commit = triton_commit or DGX_TRITON_COMMIT
    pytorch_index = pytorch_index or DGX_PYTORCH_INDEX
    install_dir = install_dir or DGX_INSTALL_DIR_DEFAULT
    venv_dir = install_dir + "/.vllm"
    activate = "source %s/bin/activate" % venv_dir

    # Check for existing working installation
    if not force:
        rc, out, _ = runner.run(
            "test -f %s/bin/activate && %s && "
            "python -c \"import vllm; print(vllm.__version__)\" && "
            "python -c \"import torch; print(torch.cuda.is_available())\"" % (venv_dir, activate),
            timeout=20)
        if rc == 0 and "True" in out:
            lines = out.strip().split("\n")
            vllm_ver = lines[0].strip() if lines else "?"
            log_queue.put("[INSTALL] Installazione esistente trovata: vLLM %s con PyTorch CUDA.\n" % vllm_ver)
            log_queue.put("[INSTALL] Saltata installazione. Usa 'Forza Reinstallazione' per reinstallare.\n")
            detector.detect()
            return

    log_queue.put("[INSTALL] === Installazione vLLM per DGX Spark (Blackwell GB10) ===\n")
    log_queue.put("[INSTALL] Directory: %s\n" % install_dir)
    log_queue.put("[INSTALL] Questa operazione richiede 20-30 minuti.\n")

    # Step 1: Pre-flight checks
    log_queue.put("[INSTALL] Step 1/9: Controlli pre-installazione...\n")
    rc, out, _ = runner.run("nvidia-smi --query-gpu=name --format=csv,noheader | head -1", timeout=10)
    gpu_name = out.strip() if rc == 0 else "unknown"
    log_queue.put("[INSTALL]   GPU: %s\n" % gpu_name)
    rc, out, _ = runner.run("nvcc --version 2>/dev/null || /usr/local/cuda/bin/nvcc --version 2>/dev/null", timeout=10)
    if rc != 0:
        log_queue.put("[ERROR] CUDA toolkit (nvcc) non trovato. Installa CUDA 13.0+\n")
        return
    log_queue.put("[INSTALL]   CUDA toolkit trovato\n")
    rc, out, _ = runner.run("df -BG $HOME | tail -1 | awk '{print $4}' | sed 's/G//'", timeout=5)
    free_gb = int(out.strip()) if rc == 0 and out.strip().isdigit() else 0
    if free_gb < 50:
        log_queue.put("[WARNING] Spazio disco basso: %d GB (consigliati 50 GB)\n" % free_gb)

    # Step 2: Install uv package manager
    if not _dgx_run_step(
        "Step 2/9: Installazione uv package manager...",
        "command -v uv >/dev/null 2>&1 || (curl -LsSf https://astral.sh/uv/install.sh | sh) && "
        "export PATH=\"$HOME/.local/bin:$PATH\" && uv --version"
    ):
        return

    # Step 3: Create Python venv
    if not _dgx_run_step(
        "Step 3/9: Creazione virtualenv Python 3.12...",
        "export PATH=\"$HOME/.local/bin:$PATH\" && "
        "mkdir -p %s && cd %s && "
        "uv venv .vllm --python 3.12" % (install_dir, install_dir)
    ):
        return

    # Step 4: Install PyTorch with CUDA 13.0
    if not _dgx_run_step(
        "Step 4/9: Installazione PyTorch 2.9.0+cu130...",
        "export PATH=\"$HOME/.local/bin:$PATH\" && "
        "%s && "
        "uv pip install torch torchvision torchaudio --index-url %s && "
        "python -c \"import torch; print('PyTorch:', torch.__version__, 'CUDA:', torch.version.cuda, 'GPU:', torch.cuda.is_available())\"" % (
            activate, pytorch_index),
        timeout=600,
    ):
        return

    # Step 5: Build Triton from source (sm_121a support)
    triton_dir = install_dir + "/triton"
    if not _dgx_run_step(
        "Step 5/9: Clone e build Triton da source (5-10 min)...",
        "export PATH=\"$HOME/.local/bin:$PATH\" && "
        "%s && "
        "cd %s && "
        "(test -d triton && cd triton && git fetch || git clone https://github.com/triton-lang/triton.git && cd triton) && "
        "cd %s && "
        "git checkout %s && "
        "git submodule update --init --recursive && "
        "uv pip install pip cmake ninja pybind11 && "
        "export TRITON_PTXAS_PATH=/usr/local/cuda/bin/ptxas && "
        "export CMAKE_BUILD_PARALLEL_LEVEL=$(nproc) && "
        "python -m pip install --no-build-isolation -v . && "
        "python -c \"import triton; print('Triton:', triton.__version__)\"" % (
            activate, install_dir, triton_dir, triton_commit),
        timeout=1800,
    ):
        return

    # Step 6: Install additional dependencies
    if not _dgx_run_step(
        "Step 6/9: Installazione dipendenze (xgrammar, setuptools-scm, tvm)...",
        "export PATH=\"$HOME/.local/bin:$PATH\" && "
        "%s && "
        "uv pip install xgrammar setuptools-scm apache-tvm-ffi==0.1.0b15 --prerelease=allow" % activate,
        timeout=300,
    ):
        return

    # Step 7: Clone vLLM
    vllm_dir = install_dir + "/vllm"
    if not _dgx_run_step(
        "Step 7/9: Clone vLLM repository...",
        "cd %s && "
        "(test -d vllm || git clone --recursive https://github.com/vllm-project/vllm.git) && "
        "cd vllm && "
        "git checkout %s && "
        "git submodule update --init --recursive" % (install_dir, vllm_commit),
        timeout=300,
    ):
        return

    # Step 8: Apply critical patches for Blackwell sm_121
    log_queue.put("[INSTALL] Step 8/9: Applicazione patch per Blackwell sm_121...\n")

    # 8a. Fix pyproject.toml license field
    _dgx_run_step(
        "  Patch: pyproject.toml license field...",
        "cd %s && "
        "sed -i 's/^license = \"Apache-2.0\"$/license = {text = \"Apache-2.0\"}/' pyproject.toml && "
        "sed -i '/^license-files = /d' pyproject.toml" % vllm_dir,
        timeout=10,
    )

    # 8b. CMakeLists.txt — add SM100/SM120/SM121 to CUTLASS kernel archs
    _dgx_run_step(
        "  Patch: CMakeLists.txt SM100/SM120 CUTLASS fix...",
        "cd %s && "
        "sed -i 's/cuda_archs_loose_intersection(SCALED_MM_ARCHS \"10.0f;11.0f\"/cuda_archs_loose_intersection(SCALED_MM_ARCHS \"10.0f;11.0f;12.0f\"/' CMakeLists.txt && "
        "sed -i 's/cuda_archs_loose_intersection(SCALED_MM_ARCHS \"10.0a\"/cuda_archs_loose_intersection(SCALED_MM_ARCHS \"10.0a;12.1a\"/' CMakeLists.txt" % vllm_dir,
        timeout=10,
    )

    # 8c. Fix flashinfer-python license in uv cache
    _dgx_run_step(
        "  Patch: flashinfer-python license cache...",
        "find $HOME/.cache/uv/sdists-v9/pypi/flashinfer-python -name 'pyproject.toml' "
        "-exec sed -i 's/^license = \"Apache-2.0\"$/license = {text = \"Apache-2.0\"}/' {} \\; "
        "-exec sed -i '/^license-files = /d' {} \\; 2>/dev/null; true",
        timeout=10,
    )

    # 8d. Configure vLLM to use existing PyTorch
    _dgx_run_step(
        "  Patch: use_existing_torch.py...",
        "%s && cd %s && "
        "python use_existing_torch.py" % (activate, vllm_dir),
        timeout=30,
    )

    # 8e. Record PyTorch version before build to detect overwrites
    rc_pt, out_pt, _ = runner.run(
        "%s && python -c \"import torch; print(torch.__version__)\"" % activate, timeout=10)
    torch_ver_before = out_pt.strip() if rc_pt == 0 else ""
    if torch_ver_before:
        log_queue.put("[INSTALL]   PyTorch pre-build: %s\n" % torch_ver_before)
    else:
        log_queue.put("[WARNING]   PyTorch NON trovato prima del build! Reinstallazione forzata dopo il build.\n")

    log_queue.put("[INSTALL]   Patch applicate.\n")

    # Step 9: Build vLLM (the big one, 15-20 min)
    if not _dgx_run_step(
        "Step 9/9: Build vLLM con TORCH_CUDA_ARCH_LIST=12.1a (15-20 min)...",
        "export PATH=\"$HOME/.local/bin:$PATH\" && "
        "%s && "
        "cd %s && "
        "export TORCH_CUDA_ARCH_LIST=12.1a && "
        "export VLLM_USE_FLASHINFER_MXFP4_MOE=1 && "
        "export TRITON_PTXAS_PATH=/usr/local/cuda/bin/ptxas && "
        "uv pip install --no-build-isolation --prerelease=allow -e ." % (activate, vllm_dir),
        timeout=2400,
    ):
        # Retry with flashinfer fix
        log_queue.put("[INSTALL] Retry: fix flashinfer e rebuild...\n")
        _dgx_run_step(
            "  Fix flashinfer + retry build...",
            "find $HOME/.cache/uv/sdists-v9/pypi/flashinfer-python -name 'pyproject.toml' "
            "-exec sed -i 's/^license = \"Apache-2.0\"$/license = {text = \"Apache-2.0\"}/' {} \\; "
            "-exec sed -i '/^license-files = /d' {} \\; 2>/dev/null; "
            "export PATH=\"$HOME/.local/bin:$PATH\" && "
            "%s && cd %s && "
            "export TORCH_CUDA_ARCH_LIST=12.1a && "
            "export VLLM_USE_FLASHINFER_MXFP4_MOE=1 && "
            "export TRITON_PTXAS_PATH=/usr/local/cuda/bin/ptxas && "
            "uv pip install --no-build-isolation --prerelease=allow -e ." % (activate, vllm_dir),
            timeout=2400,
        )

    # Step 9b: Verify PyTorch is still the CUDA version (uv may have overwritten it)
    rc_pt2, out_pt2, _ = runner.run(
        "%s && python -c \"import torch; print(torch.__version__); print(torch.cuda.is_available())\"" % activate,
        timeout=15)
    torch_ok = rc_pt2 == 0 and "True" in out_pt2
    if not torch_ok:
        log_queue.put("[WARNING] PyTorch CUDA non trovato dopo il build! Reinstallazione PyTorch cu130...\n")
        _dgx_run_step(
            "  Reinstallazione PyTorch 2.9.0+cu130 (sovrascritto durante il build)...",
            "export PATH=\"$HOME/.local/bin:$PATH\" && "
            "%s && "
            "uv pip install torch torchvision torchaudio --index-url %s --force-reinstall && "
            "python -c \"import torch; print('PyTorch:', torch.__version__, 'CUDA:', torch.version.cuda)\"" % (
                activate, pytorch_index),
            timeout=600,
        )
    else:
        torch_ver_after = out_pt2.strip().split("\n")[0] if out_pt2.strip() else ""
        log_queue.put("[INSTALL]   PyTorch post-build: %s (CUDA OK)\n" % torch_ver_after)

    # Create env activation script
    _dgx_run_step(
        "Creazione script ambiente vllm_env.sh...",
        "cat > %s/vllm_env.sh << 'ENVEOF'\n"
        "#!/bin/bash\n"
        "SCRIPT_DIR=\"$(cd \"$(dirname \"${BASH_SOURCE[0]}\")\" && pwd)\"\n"
        "source \"$SCRIPT_DIR/.vllm/bin/activate\"\n"
        "export TORCH_CUDA_ARCH_LIST=12.1a\n"
        "export VLLM_USE_FLASHINFER_MXFP4_MOE=1\n"
        "CUDA_PATH=$(ls -d /usr/local/cuda* 2>/dev/null | head -1)\n"
        "export TRITON_PTXAS_PATH=\"$CUDA_PATH/bin/ptxas\"\n"
        "export PATH=\"$CUDA_PATH/bin:$PATH\"\n"
        "export LD_LIBRARY_PATH=\"$CUDA_PATH/lib64:$LD_LIBRARY_PATH\"\n"
        "export TIKTOKEN_CACHE_DIR=\"$SCRIPT_DIR/.tiktoken_cache\"\n"
        "mkdir -p \"$TIKTOKEN_CACHE_DIR\"\n"
        "echo \"=== vLLM DGX Spark Environment Active ===\"\n"
        "ENVEOF\n"
        "chmod +x %s/vllm_env.sh" % (install_dir, install_dir),
        timeout=10,
    )

    # Verification
    log_queue.put("[INSTALL] Verifica installazione...\n")
    rc, out, err = runner.run(
        "%s && python -c \"import vllm; print('vLLM', vllm.__version__)\" && "
        "python -c \"import torch; print('PyTorch', torch.__version__, 'CUDA', torch.version.cuda, 'GPU', torch.cuda.is_available())\"" % activate,
        timeout=30,
    )
    if rc == 0:
        for line in out.strip().splitlines():
            log_queue.put("[INSTALL]   %s\n" % line)
        log_queue.put("[INSTALL] === Installazione DGX Spark completata con successo! ===\n")
    else:
        log_queue.put("[ERROR] Verifica fallita: %s\n" % (err.strip() or out.strip()))

    detector.detect()


# --- API: GPU ---

@app.route("/api/gpu")
def api_gpu():
    d = gpu_mon.latest
    if not d:
        return jsonify({"available": False})
    total = d.get("vram_total", 1)
    history_vram_pct = [v / total * 100 for v in gpu_mon.history_vram] if total else []
    return jsonify({
        "available": True,
        **d,
        "history_vram_pct": list(history_vram_pct),
        "history_util": list(gpu_mon.history_util),
        "history_temp": list(gpu_mon.history_temp),
        "vram_label": platform_info.vram_label,
        "is_macos": platform_info.os == "macos",
    })


# --- API: Logs (SSE) ---

@app.route("/api/logs/stream")
def api_logs_stream():
    def generate():
        idx = 0
        while True:
            with log_buffer_lock:
                buf_len = len(log_buffer)
            if idx < buf_len:
                with log_buffer_lock:
                    new_lines = list(log_buffer)[idx:buf_len]
                idx = buf_len
                for line in new_lines:
                    escaped = json.dumps(line)
                    yield "data: %s\n\n" % escaped
            else:
                time.sleep(0.3)
    return Response(generate(), mimetype="text/event-stream",
                    headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


@app.route("/api/logs/history")
def api_logs_history():
    with log_buffer_lock:
        lines = list(log_buffer)
    return jsonify({"lines": lines})


@app.route("/api/logs/clear", methods=["POST"])
def api_logs_clear():
    with log_buffer_lock:
        log_buffer.clear()
    return jsonify({"status": "cleared"})


# --- API: Models ---

@app.route("/api/models/search")
def api_models_search():
    q = flask_request.args.get("q", "").strip()
    if not q:
        return jsonify({"models": []})
    try:
        url = ("https://huggingface.co/api/models?"
               "search=%s&limit=20&sort=downloads&direction=-1" % parse.quote(q))
        req_obj = request.Request(url, headers={"User-Agent": "vLLM-Manager/2.0"})
        with request.urlopen(req_obj, timeout=15) as resp:
            models = json.loads(resp.read())
        results = []
        for m in models:
            results.append({
                "id": m.get("modelId", m.get("id", "?")),
                "downloads": m.get("downloads", 0),
                "likes": m.get("likes", 0),
            })
        return jsonify({"models": results})
    except Exception as e:
        return jsonify({"error": str(e), "models": []})


@app.route("/api/models/local")
def api_models_local():
    if platform_info.os == "macos":
        cache_path = "~/Library/Caches/huggingface/hub/models--*/"
    else:
        cache_path = "~/.cache/huggingface/hub/models--*/"
    rc, out, _ = runner.run_raw("du -sh %s 2>/dev/null" % cache_path, timeout=30)
    models = []
    if rc == 0 and out.strip():
        for line in out.strip().splitlines():
            parts = line.split("\t", 1)
            if len(parts) == 2:
                size = parts[0].strip()
                path = parts[1].strip().rstrip("/")
                dirname = path.rsplit("/", 1)[-1]
                name = dirname.replace("models--", "", 1).replace("--", "/", 1)
                models.append({"name": name, "size": size})
    return jsonify({"models": models})


@app.route("/api/models/delete", methods=["POST"])
def api_models_delete():
    data = flask_request.json or {}
    name = data.get("name", "")
    if not name:
        return jsonify({"error": "Nome modello mancante"}), 400
    dir_name = "models--" + name.replace("/", "--")
    if platform_info.os == "macos":
        cache_base = "~/Library/Caches/huggingface/hub"
    else:
        cache_base = "~/.cache/huggingface/hub"
    rc, out, err = runner.run_raw("rm -rf %s/%s" % (cache_base, dir_name), timeout=60)
    if rc == 0:
        return jsonify({"status": "deleted"})
    return jsonify({"error": err.strip() or "errore sconosciuto"}), 500


# --- API: Benchmark ---

@app.route("/api/benchmark", methods=["POST"])
def api_benchmark():
    data = flask_request.json or {}
    prompt = data.get("prompt", "").strip()
    max_tokens = int(data.get("max_tokens", 256))
    temperature = float(data.get("temperature", 0.7))

    if not prompt:
        return jsonify({"error": "Prompt vuoto"}), 400

    models = api.list_models()
    model = models[0] if models else data.get("model", "")
    messages = [{"role": "user", "content": prompt}]

    t0 = time.perf_counter()
    result = api.chat(model, messages, max_tokens=max_tokens,
                      temperature=temperature, stream=True)
    total_time = time.perf_counter() - t0

    if "error" in result:
        return jsonify({"error": result["error"]})

    n_tokens = result.get("tokens_count", 0)
    elapsed = result.get("elapsed", total_time)
    ttft = result.get("ttft")
    tps = n_tokens / elapsed if elapsed > 0 else 0

    return jsonify({
        "model": model,
        "tokens": n_tokens,
        "elapsed": round(elapsed, 2),
        "tps": round(tps, 1),
        "ttft_ms": round(ttft * 1000) if ttft else None,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "text": result.get("text", "")[:2000],
    })


# --- API: WebUI (Docker) ---

@app.route("/api/webui/status")
def api_webui_status():
    rc, out, _ = docker_run(
        ["ps", "--filter", "name=open-webui", "--format", "{{.Status}}"], timeout=10)
    return jsonify({"status": out.strip() if rc == 0 and out.strip() else "non in esecuzione"})


@app.route("/api/webui/start", methods=["POST"])
def api_webui_start():
    data = flask_request.json or {}
    container = data.get("container", "open-webui")
    image = data.get("image", "ghcr.io/open-webui/open-webui:main")
    port = int(data.get("port", 3000))

    def _run():
        rc, out, err = docker_run(["start", container], timeout=15)
        if rc == 0:
            log_queue.put("[WEBUI] Container '%s' avviato.\n" % container)
        else:
            rc2, out2, err2 = docker_run([
                "run", "-d", "--name", container,
                "-p", "%d:8080" % port,
                "--add-host=host.docker.internal:host-gateway",
                "-v", "open-webui:/app/backend/data",
                "--restart", "always", image,
            ], timeout=60)
            msg = out2.strip() or err2.strip() or "done"
            log_queue.put("[WEBUI] docker run: %s\n" % msg)
    threading.Thread(target=_run, daemon=True).start()
    return jsonify({"status": "starting"})


@app.route("/api/webui/stop", methods=["POST"])
def api_webui_stop():
    data = flask_request.json or {}
    container = data.get("container", "open-webui")

    def _run():
        rc, out, err = docker_run(["stop", container], timeout=20)
        msg = "Container fermato." if rc == 0 else (err.strip() or "errore")
        log_queue.put("[WEBUI] %s\n" % msg)
    threading.Thread(target=_run, daemon=True).start()
    return jsonify({"status": "stopping"})


# --- API: Profiles ---

@app.route("/api/profiles")
def api_profiles_list():
    return jsonify({"profiles": config.get_profiles()})


@app.route("/api/profiles/save", methods=["POST"])
def api_profiles_save():
    data = flask_request.json or {}
    name = data.get("name", "").strip()
    profile = data.get("profile", {})
    if not name:
        return jsonify({"error": "Nome mancante"}), 400
    config.save_profile(name, profile)
    return jsonify({"status": "saved"})


@app.route("/api/profiles/delete", methods=["POST"])
def api_profiles_delete():
    data = flask_request.json or {}
    name = data.get("name", "").strip()
    if not name:
        return jsonify({"error": "Nome mancante"}), 400
    config.delete_profile(name)
    return jsonify({"status": "deleted"})


# --- API: Platform info ---

@app.route("/api/platform")
def api_platform():
    return jsonify({
        "os": platform_info.os,
        "arch": platform_info.arch,
        "gpu_type": platform_info.gpu_type,
        "is_dgx": platform_info.is_dgx,
        "display_name": platform_info.display_name,
        "vram_label": platform_info.vram_label,
        "presets": preset_models,
    })


# --- API: DGX Spark ---

@app.route("/api/dgx/cluster")
def api_dgx_cluster():
    lines = []
    rc, out, _ = runner.run("hostname", timeout=5)
    hostname = out.strip() if rc == 0 else "unknown"
    rc, out, _ = runner.run("hostname -I", timeout=5)
    ips = out.strip() if rc == 0 else "?"
    lines.append("Nodo locale: %s  IP: %s" % (hostname, ips))

    rc, out, _ = runner.run(
        "nvidia-smi --query-gpu=name,memory.used,memory.total,temperature.gpu,utilization.gpu "
        "--format=csv,noheader,nounits", timeout=5)
    if rc == 0 and out.strip():
        for gpu_line in out.strip().splitlines():
            parts = [p.strip() for p in gpu_line.split(",")]
            if len(parts) >= 5:
                lines.append("  GPU: %s  %s/%s MB  Temp: %s C  Util: %s%%" % (
                    parts[0], parts[1], parts[2], parts[3], parts[4]))

    rc, out, _ = runner.run("ray status 2>/dev/null", timeout=10)
    if rc == 0 and out.strip():
        lines.append("\nRay Cluster:")
        for ray_line in out.strip().splitlines()[:10]:
            lines.append("  " + ray_line)
    else:
        lines.append("\nRay: non attivo")

    return jsonify({"text": "\n".join(lines)})


@app.route("/api/dgx/discover")
def api_dgx_discover():
    lines = []
    rc, out, _ = runner.run(
        "ip -br addr show | grep -E '(enP|enp|mlx)'", timeout=10)
    if rc == 0 and out.strip():
        lines.append("Interfacce di rete:")
        for line in out.strip().splitlines():
            lines.append("  " + line.strip())

    rc, out, _ = runner.run("arp -a 2>/dev/null | head -20", timeout=10)
    if rc == 0 and out.strip():
        lines.append("\nNodi rilevati (ARP):")
        for line in out.strip().splitlines():
            lines.append("  " + line.strip())

    return jsonify({"text": "\n".join(lines) if lines else "Nessun nodo aggiuntivo rilevato."})


@app.route("/api/dgx/connectivity")
def api_dgx_connectivity():
    rc, out, _ = runner.run("ping -c 2 -W 2 localhost", timeout=10)
    status = "OK" if rc == 0 else "FAIL"
    log_queue.put("[NETWORK] localhost: %s\n" % status)
    return jsonify({"status": status})


@app.route("/api/dgx/nccl-test")
def api_dgx_nccl_test():
    rc, out, err = runner.run(
        "python3 -c \""
        "import torch; import torch.distributed as dist; "
        "print('NCCL available:', torch.cuda.nccl.is_available(torch.randn(1).cuda())); "
        "print('CUDA devices:', torch.cuda.device_count())\"",
        timeout=30)
    if rc == 0:
        log_queue.put("[NCCL] %s\n" % out.strip())
        return jsonify({"text": out.strip()})
    else:
        msg = err.strip() or "test failed"
        log_queue.put("[NCCL ERROR] %s\n" % msg)
        return jsonify({"text": "ERROR: " + msg})


@app.route("/api/dgx/ray/start-head", methods=["POST"])
def api_dgx_ray_start():
    def _run():
        rc, out, err = runner.run(
            "ray start --head --port=6379 --dashboard-host=0.0.0.0", timeout=30)
        if rc == 0:
            log_queue.put("[RAY] Head node started.\n%s\n" % out.strip())
        else:
            log_queue.put("[RAY ERROR] %s\n" % (err.strip() or out.strip()))
    threading.Thread(target=_run, daemon=True).start()
    return jsonify({"status": "starting"})


@app.route("/api/dgx/ray/connect-worker", methods=["POST"])
def api_dgx_ray_connect():
    data = flask_request.json or {}
    worker_ip = data.get("ip", "").strip()
    worker_user = data.get("user", "root").strip()
    if not worker_ip:
        return jsonify({"error": "IP mancante"}), 400

    def _run():
        rc, head_ip, _ = runner.run("hostname -I | awk '{print $1}'", timeout=5)
        head_ip = head_ip.strip()
        if not head_ip:
            log_queue.put("[ERROR] Cannot determine head IP.\n")
            return
        rc, out, err = runner.run(
            "ssh %s@%s 'ray start --address=%s:6379'" % (worker_user, worker_ip, head_ip),
            timeout=30)
        if rc == 0:
            log_queue.put("[RAY] Worker %s connected.\n" % worker_ip)
        else:
            log_queue.put("[RAY ERROR] %s\n" % (err.strip() or out.strip()))
    threading.Thread(target=_run, daemon=True).start()
    return jsonify({"status": "connecting"})


@app.route("/api/dgx/ray/stop", methods=["POST"])
def api_dgx_ray_stop():
    def _run():
        runner.run("ray stop", timeout=15)
        log_queue.put("[RAY] Ray stopped.\n")
    threading.Thread(target=_run, daemon=True).start()
    return jsonify({"status": "stopping"})


@app.route("/api/dgx/env", methods=["POST"])
def api_dgx_env():
    data = flask_request.json or {}
    if isinstance(runner, DGXCommandRunner):
        for key, value in data.items():
            runner.update_env(key, value)
        log_queue.put("[INFO] DGX env vars updated.\n")
    return jsonify({"status": "updated"})


@app.route("/api/dgx/ssh/setup", methods=["POST"])
def api_dgx_ssh_setup():
    data = flask_request.json or {}
    ip = data.get("ip", "").strip()
    user = data.get("user", "root").strip()
    if not ip:
        return jsonify({"error": "IP mancante"}), 400

    def _run():
        runner.run("test -f ~/.ssh/id_rsa || ssh-keygen -t rsa -b 4096 -N '' -f ~/.ssh/id_rsa", timeout=10)
        rc, out, err = runner.run(
            "ssh-copy-id -o StrictHostKeyChecking=no %s@%s" % (user, ip), timeout=30)
        if rc == 0:
            log_queue.put("[SSH] Chiavi copiate su %s@%s\n" % (user, ip))
        else:
            log_queue.put("[SSH ERROR] %s\n" % (err.strip() or "errore"))
    threading.Thread(target=_run, daemon=True).start()
    return jsonify({"status": "setting_up"})


@app.route("/api/dgx/ssh/test", methods=["POST"])
def api_dgx_ssh_test():
    data = flask_request.json or {}
    ip = data.get("ip", "").strip()
    user = data.get("user", "root").strip()
    if not ip:
        return jsonify({"error": "IP mancante"}), 400
    rc, out, err = runner.run(
        "ssh -o ConnectTimeout=5 %s@%s 'hostname'" % (user, ip), timeout=15)
    if rc == 0:
        return jsonify({"status": "ok", "hostname": out.strip()})
    return jsonify({"status": "fail", "error": err.strip() or "connessione fallita"})


# ---------------------------------------------------------------------------
# DGX Worker Management — remote operations from head node
# ---------------------------------------------------------------------------

# Runtime status for worker installs (not persisted — only while running)
_worker_install_lock = threading.Lock()
_worker_installing = {}  # ip -> True while install is running


def _ssh_cmd(user, ip, cmd, timeout=30):
    """Run a command on a remote node via SSH."""
    return runner.run(
        "ssh -o ConnectTimeout=10 -o StrictHostKeyChecking=no %s@%s 'bash -lc \"%s\"'" % (
            user, ip, cmd.replace('"', '\\"')),
        timeout=timeout)


def _ssh_cmd_b64(user, ip, cmd, timeout=30):
    """Run a complex command on a remote node via SSH using base64 encoding
    to avoid shell escaping issues."""
    encoded = base64.b64encode(cmd.encode()).decode()
    return runner.run(
        "ssh -o ConnectTimeout=10 -o StrictHostKeyChecking=no %s@%s 'echo %s | base64 -d | bash -l'" % (
            user, ip, encoded),
        timeout=timeout)


def _dgx_remote_run_step(msg, cmd, ip, user, timeout=1800):
    """Run a single install step on remote worker via SSH, log output."""
    log_queue.put("[WORKER %s] %s\n" % (ip, msg))
    rc, out, err = _ssh_cmd_b64(user, ip, cmd, timeout=timeout)
    if out.strip():
        for line in out.strip().splitlines()[-5:]:
            log_queue.put("[WORKER %s]   %s\n" % (ip, line))
    if rc != 0:
        error = err.strip() or out.strip() or "exit code %d" % rc
        log_queue.put("[WORKER %s ERROR] %s\n" % (ip, error))
        return False
    return True


def _install_vllm_dgx_remote(ip, user):
    """Full DGX Spark installation on a remote worker node via SSH."""
    install_dir = DGX_INSTALL_DIR_DEFAULT
    vllm_commit = DGX_VLLM_COMMIT
    triton_commit = DGX_TRITON_COMMIT
    pytorch_index = DGX_PYTORCH_INDEX
    venv_dir = install_dir + "/.vllm"
    activate = "source %s/bin/activate" % venv_dir

    log_queue.put("[WORKER %s] === Installazione remota vLLM per DGX Spark ===\n" % ip)
    log_queue.put("[WORKER %s] Questa operazione richiede 20-40 minuti.\n" % ip)

    # Step 1: Pre-flight checks
    log_queue.put("[WORKER %s] Step 1/9: Controlli pre-installazione...\n" % ip)
    rc, out, _ = _ssh_cmd(user, ip, "nvidia-smi --query-gpu=name --format=csv,noheader | head -1", timeout=15)
    gpu_name = out.strip() if rc == 0 else "unknown"
    log_queue.put("[WORKER %s]   GPU: %s\n" % (ip, gpu_name))
    rc, out, _ = _ssh_cmd(user, ip, "nvcc --version 2>/dev/null || /usr/local/cuda/bin/nvcc --version 2>/dev/null", timeout=15)
    if rc != 0:
        log_queue.put("[WORKER %s ERROR] CUDA toolkit (nvcc) non trovato.\n" % ip)
        return False
    log_queue.put("[WORKER %s]   CUDA toolkit trovato\n" % ip)

    # Step 2: Install uv package manager
    if not _dgx_remote_run_step(
        "Step 2/9: Installazione uv package manager...",
        "command -v uv >/dev/null 2>&1 || (curl -LsSf https://astral.sh/uv/install.sh | sh) && "
        "export PATH=\"$HOME/.local/bin:$PATH\" && uv --version",
        ip, user,
    ):
        return False

    # Step 3: Create Python venv
    if not _dgx_remote_run_step(
        "Step 3/9: Creazione virtualenv Python 3.12...",
        "export PATH=\"$HOME/.local/bin:$PATH\" && "
        "mkdir -p %s && cd %s && "
        "uv venv .vllm --python 3.12" % (install_dir, install_dir),
        ip, user,
    ):
        return False

    # Step 4: Install PyTorch with CUDA 13.0
    if not _dgx_remote_run_step(
        "Step 4/9: Installazione PyTorch 2.9.0+cu130...",
        "export PATH=\"$HOME/.local/bin:$PATH\" && "
        "%s && "
        "uv pip install torch torchvision torchaudio --index-url %s && "
        "python -c \"import torch; print('PyTorch:', torch.__version__, 'CUDA:', torch.version.cuda, 'GPU:', torch.cuda.is_available())\"" % (
            activate, pytorch_index),
        ip, user, timeout=600,
    ):
        return False

    # Step 5: Build Triton from source (sm_121a support)
    triton_dir = install_dir + "/triton"
    if not _dgx_remote_run_step(
        "Step 5/9: Clone e build Triton da source (5-10 min)...",
        "export PATH=\"$HOME/.local/bin:$PATH\" && "
        "%s && "
        "cd %s && "
        "(test -d triton && cd triton && git fetch || git clone https://github.com/triton-lang/triton.git && cd triton) && "
        "cd %s && "
        "git checkout %s && "
        "git submodule update --init --recursive && "
        "uv pip install pip cmake ninja pybind11 && "
        "export TRITON_PTXAS_PATH=/usr/local/cuda/bin/ptxas && "
        "export CMAKE_BUILD_PARALLEL_LEVEL=$(nproc) && "
        "python -m pip install --no-build-isolation -v . && "
        "python -c \"import triton; print('Triton:', triton.__version__)\"" % (
            activate, install_dir, triton_dir, triton_commit),
        ip, user, timeout=1800,
    ):
        return False

    # Step 6: Install additional dependencies
    if not _dgx_remote_run_step(
        "Step 6/9: Installazione dipendenze (xgrammar, setuptools-scm, tvm)...",
        "export PATH=\"$HOME/.local/bin:$PATH\" && "
        "%s && "
        "uv pip install xgrammar setuptools-scm apache-tvm-ffi==0.1.0b15 --prerelease=allow" % activate,
        ip, user, timeout=300,
    ):
        return False

    # Step 7: Clone vLLM
    vllm_dir = install_dir + "/vllm"
    if not _dgx_remote_run_step(
        "Step 7/9: Clone vLLM repository...",
        "cd %s && "
        "(test -d vllm || git clone --recursive https://github.com/vllm-project/vllm.git) && "
        "cd vllm && "
        "git checkout %s && "
        "git submodule update --init --recursive" % (install_dir, vllm_commit),
        ip, user, timeout=300,
    ):
        return False

    # Step 8: Apply critical patches for Blackwell sm_121
    log_queue.put("[WORKER %s] Step 8/9: Applicazione patch per Blackwell sm_121...\n" % ip)

    _dgx_remote_run_step(
        "  Patch: pyproject.toml license field...",
        "cd %s && "
        "sed -i 's/^license = \"Apache-2.0\"$/license = {text = \"Apache-2.0\"}/' pyproject.toml && "
        "sed -i '/^license-files = /d' pyproject.toml" % vllm_dir,
        ip, user, timeout=10,
    )

    _dgx_remote_run_step(
        "  Patch: CMakeLists.txt SM100/SM120 CUTLASS fix...",
        "cd %s && "
        "sed -i 's/cuda_archs_loose_intersection(SCALED_MM_ARCHS \"10.0f;11.0f\"/cuda_archs_loose_intersection(SCALED_MM_ARCHS \"10.0f;11.0f;12.0f\"/' CMakeLists.txt && "
        "sed -i 's/cuda_archs_loose_intersection(SCALED_MM_ARCHS \"10.0a\"/cuda_archs_loose_intersection(SCALED_MM_ARCHS \"10.0a;12.1a\"/' CMakeLists.txt" % vllm_dir,
        ip, user, timeout=10,
    )

    _dgx_remote_run_step(
        "  Patch: flashinfer-python license cache...",
        "find $HOME/.cache/uv/sdists-v9/pypi/flashinfer-python -name 'pyproject.toml' "
        "-exec sed -i 's/^license = \"Apache-2.0\"$/license = {text = \"Apache-2.0\"}/' {} \\; "
        "-exec sed -i '/^license-files = /d' {} \\; 2>/dev/null; true",
        ip, user, timeout=10,
    )

    _dgx_remote_run_step(
        "  Patch: use_existing_torch.py...",
        "%s && cd %s && "
        "python use_existing_torch.py" % (activate, vllm_dir),
        ip, user, timeout=30,
    )

    # Record PyTorch version before build
    rc_pt, out_pt, _ = _ssh_cmd_b64(user, ip,
        "%s && python -c \"import torch; print(torch.__version__)\"" % activate, timeout=10)
    torch_ver_before = out_pt.strip() if rc_pt == 0 else ""
    if torch_ver_before:
        log_queue.put("[WORKER %s]   PyTorch pre-build: %s\n" % (ip, torch_ver_before))
    else:
        log_queue.put("[WORKER %s WARNING]   PyTorch NON trovato prima del build!\n" % ip)

    # Step 9: Build vLLM
    if not _dgx_remote_run_step(
        "Step 9/9: Build vLLM con TORCH_CUDA_ARCH_LIST=12.1a (15-20 min)...",
        "export PATH=\"$HOME/.local/bin:$PATH\" && "
        "%s && "
        "cd %s && "
        "export TORCH_CUDA_ARCH_LIST=12.1a && "
        "export VLLM_USE_FLASHINFER_MXFP4_MOE=1 && "
        "export TRITON_PTXAS_PATH=/usr/local/cuda/bin/ptxas && "
        "uv pip install --no-build-isolation --prerelease=allow -e ." % (activate, vllm_dir),
        ip, user, timeout=2400,
    ):
        # Retry with flashinfer fix
        log_queue.put("[WORKER %s] Retry: fix flashinfer e rebuild...\n" % ip)
        if not _dgx_remote_run_step(
            "  Fix flashinfer + retry build...",
            "find $HOME/.cache/uv/sdists-v9/pypi/flashinfer-python -name 'pyproject.toml' "
            "-exec sed -i 's/^license = \"Apache-2.0\"$/license = {text = \"Apache-2.0\"}/' {} \\; "
            "-exec sed -i '/^license-files = /d' {} \\; 2>/dev/null; "
            "export PATH=\"$HOME/.local/bin:$PATH\" && "
            "%s && cd %s && "
            "export TORCH_CUDA_ARCH_LIST=12.1a && "
            "export VLLM_USE_FLASHINFER_MXFP4_MOE=1 && "
            "export TRITON_PTXAS_PATH=/usr/local/cuda/bin/ptxas && "
            "uv pip install --no-build-isolation --prerelease=allow -e ." % (activate, vllm_dir),
            ip, user, timeout=2400,
        ):
            return False

    # Verify PyTorch is still the CUDA version after build
    rc_pt2, out_pt2, _ = _ssh_cmd_b64(user, ip,
        "%s && python -c \"import torch; print(torch.__version__); print(torch.cuda.is_available())\"" % activate,
        timeout=15)
    torch_ok = rc_pt2 == 0 and "True" in out_pt2
    if not torch_ok:
        log_queue.put("[WORKER %s WARNING] PyTorch CUDA sovrascritto! Reinstallazione...\n" % ip)
        _dgx_remote_run_step(
            "  Reinstallazione PyTorch cu130...",
            "export PATH=\"$HOME/.local/bin:$PATH\" && "
            "%s && "
            "uv pip install torch torchvision torchaudio --index-url %s --force-reinstall && "
            "python -c \"import torch; print('PyTorch:', torch.__version__, 'CUDA:', torch.version.cuda)\"" % (
                activate, pytorch_index),
            ip, user, timeout=600,
        )

    # Create env activation script
    _dgx_remote_run_step(
        "Creazione script ambiente vllm_env.sh...",
        "cat > %s/vllm_env.sh << 'ENVEOF'\n"
        "#!/bin/bash\n"
        "SCRIPT_DIR=\"$(cd \"$(dirname \"${BASH_SOURCE[0]}\")\" && pwd)\"\n"
        "source \"$SCRIPT_DIR/.vllm/bin/activate\"\n"
        "export TORCH_CUDA_ARCH_LIST=12.1a\n"
        "export VLLM_USE_FLASHINFER_MXFP4_MOE=1\n"
        "CUDA_PATH=$(ls -d /usr/local/cuda* 2>/dev/null | head -1)\n"
        "export TRITON_PTXAS_PATH=\"$CUDA_PATH/bin/ptxas\"\n"
        "export PATH=\"$CUDA_PATH/bin:$PATH\"\n"
        "export LD_LIBRARY_PATH=\"$CUDA_PATH/lib64:$LD_LIBRARY_PATH\"\n"
        "export TIKTOKEN_CACHE_DIR=\"$SCRIPT_DIR/.tiktoken_cache\"\n"
        "mkdir -p \"$TIKTOKEN_CACHE_DIR\"\n"
        "echo \"=== vLLM DGX Spark Environment Active ===\"\n"
        "ENVEOF\n"
        "chmod +x %s/vllm_env.sh" % (install_dir, install_dir),
        ip, user, timeout=10,
    )

    # Verification
    log_queue.put("[WORKER %s] Verifica installazione...\n" % ip)
    rc, out, err = _ssh_cmd_b64(user, ip,
        "%s && python -c \"import vllm; print('vLLM', vllm.__version__)\" && "
        "python -c \"import torch; print('PyTorch', torch.__version__, 'CUDA', torch.version.cuda, 'GPU', torch.cuda.is_available())\"" % activate,
        timeout=30)
    if rc == 0:
        for line in out.strip().splitlines():
            log_queue.put("[WORKER %s]   %s\n" % (ip, line))
        log_queue.put("[WORKER %s] === Installazione completata con successo! ===\n" % ip)
        return True
    else:
        log_queue.put("[WORKER %s ERROR] Verifica fallita: %s\n" % (ip, err.strip() or out.strip()))
        return False


def _verify_worker(ip, user):
    """Check GPU, vLLM, and Ray on a remote worker. Returns dict of results."""
    dgx_dir = detector.install_dir or DGX_INSTALL_DIR_DEFAULT
    activate = "source %s/.vllm/bin/activate" % dgx_dir
    results = {}

    # GPU check
    rc, out, _ = _ssh_cmd(user, ip,
        "nvidia-smi --query-gpu=name,memory.total --format=csv,noheader", timeout=15)
    results["gpu"] = {"ok": rc == 0 and bool(out.strip()), "info": out.strip() if rc == 0 else "non rilevata"}

    # vLLM check
    rc, out, _ = _ssh_cmd_b64(user, ip,
        "%s && python -c \"import vllm; print(vllm.__version__)\"" % activate, timeout=30)
    results["vllm"] = {"ok": rc == 0 and bool(out.strip()), "info": out.strip() if rc == 0 else "non installato"}

    # Ray check
    rc, out, _ = _ssh_cmd_b64(user, ip,
        "%s && python -c \"import ray; print(ray.__version__)\"" % activate, timeout=15)
    results["ray"] = {"ok": rc == 0 and bool(out.strip()), "info": out.strip() if rc == 0 else "non disponibile"}

    return results


# --- Worker API endpoints ---

@app.route("/api/dgx/workers")
def api_dgx_workers():
    workers = dict(config.get_workers())
    # Merge runtime install status
    for ip, w in workers.items():
        if _worker_installing.get(ip):
            w["status"] = "installing"
    return jsonify({"workers": workers})


@app.route("/api/dgx/workers/add", methods=["POST"])
def api_dgx_workers_add():
    data = flask_request.json or {}
    ip = data.get("ip", "").strip()
    user = data.get("user", "root").strip()
    alias = data.get("alias", "").strip()
    if not ip:
        return jsonify({"error": "IP mancante"}), 400
    worker = {"ip": ip, "user": user, "alias": alias or ip, "status": "added"}
    config.save_worker(ip, worker)
    log_queue.put("[WORKER %s] Aggiunto (%s@%s)\n" % (ip, user, ip))
    return jsonify({"status": "ok", "worker": worker})


@app.route("/api/dgx/workers/remove", methods=["POST"])
def api_dgx_workers_remove():
    data = flask_request.json or {}
    ip = data.get("ip", "").strip()
    if not ip:
        return jsonify({"error": "IP mancante"}), 400
    if _worker_installing.get(ip):
        return jsonify({"error": "Installazione in corso, impossibile rimuovere"}), 400
    config.delete_worker(ip)
    log_queue.put("[WORKER %s] Rimosso\n" % ip)
    return jsonify({"status": "ok"})


@app.route("/api/dgx/workers/ssh-setup", methods=["POST"])
def api_dgx_workers_ssh_setup():
    data = flask_request.json or {}
    ip = data.get("ip", "").strip()
    if not ip:
        return jsonify({"error": "IP mancante"}), 400
    w = config.get_workers().get(ip)
    if not w:
        return jsonify({"error": "Worker non trovato"}), 404
    user = w["user"]

    def _run():
        runner.run("test -f ~/.ssh/id_rsa || ssh-keygen -t rsa -b 4096 -N '' -f ~/.ssh/id_rsa", timeout=10)
        rc, out, err = runner.run(
            "ssh-copy-id -o StrictHostKeyChecking=no %s@%s" % (user, ip), timeout=30)
        if rc == 0:
            log_queue.put("[WORKER %s] Chiavi SSH copiate\n" % ip)
            # Verify SSH works
            rc2, out2, _ = runner.run(
                "ssh -o ConnectTimeout=5 %s@%s 'hostname'" % (user, ip), timeout=15)
            if rc2 == 0:
                config.update_worker_status(ip, "ssh_ok")
                log_queue.put("[WORKER %s] SSH verificato (host: %s)\n" % (ip, out2.strip()))
            else:
                config.update_worker_status(ip, "error")
        else:
            log_queue.put("[WORKER %s ERROR] SSH setup fallito: %s\n" % (ip, err.strip() or "errore"))
            config.update_worker_status(ip, "error")
    threading.Thread(target=_run, daemon=True).start()
    return jsonify({"status": "setting_up"})


@app.route("/api/dgx/workers/ssh-test", methods=["POST"])
def api_dgx_workers_ssh_test():
    data = flask_request.json or {}
    ip = data.get("ip", "").strip()
    if not ip:
        return jsonify({"error": "IP mancante"}), 400
    w = config.get_workers().get(ip)
    if not w:
        return jsonify({"error": "Worker non trovato"}), 404
    rc, out, err = runner.run(
        "ssh -o ConnectTimeout=5 %s@%s 'hostname'" % (w["user"], ip), timeout=15)
    if rc == 0:
        config.update_worker_status(ip, "ssh_ok")
        return jsonify({"status": "ok", "hostname": out.strip()})
    return jsonify({"status": "fail", "error": err.strip() or "connessione fallita"})


@app.route("/api/dgx/workers/install", methods=["POST"])
def api_dgx_workers_install():
    data = flask_request.json or {}
    ip = data.get("ip", "").strip()
    if not ip:
        return jsonify({"error": "IP mancante"}), 400
    w = config.get_workers().get(ip)
    if not w:
        return jsonify({"error": "Worker non trovato"}), 404

    with _worker_install_lock:
        if any(_worker_installing.values()):
            return jsonify({"error": "Un'altra installazione e' gia' in corso. Attendi il completamento."}), 409
        _worker_installing[ip] = True

    config.update_worker_status(ip, "installing")

    def _run():
        try:
            success = _install_vllm_dgx_remote(ip, w["user"])
            if success:
                config.update_worker_status(ip, "installed")
            else:
                config.update_worker_status(ip, "error")
        finally:
            _worker_installing[ip] = False
    threading.Thread(target=_run, daemon=True).start()
    return jsonify({"status": "installing"})


@app.route("/api/dgx/workers/verify", methods=["POST"])
def api_dgx_workers_verify():
    data = flask_request.json or {}
    ip = data.get("ip", "").strip()
    if not ip:
        return jsonify({"error": "IP mancante"}), 400
    w = config.get_workers().get(ip)
    if not w:
        return jsonify({"error": "Worker non trovato"}), 404

    results = _verify_worker(ip, w["user"])
    all_ok = all(r["ok"] for r in results.values())
    if all_ok:
        config.update_worker_status(ip, "ready")
        log_queue.put("[WORKER %s] Verifica OK: GPU=%s, vLLM=%s, Ray=%s\n" % (
            ip, results["gpu"]["info"], results["vllm"]["info"], results["ray"]["info"]))
    else:
        failed = [k for k, v in results.items() if not v["ok"]]
        log_queue.put("[WORKER %s] Verifica FALLITA: %s\n" % (ip, ", ".join(failed)))
    return jsonify({"status": "ok" if all_ok else "fail", "results": results})


@app.route("/api/dgx/ray/start-cluster", methods=["POST"])
def api_dgx_ray_start_cluster():
    """Start Ray head on this node, then connect all ready workers."""
    def _run():
        # Start head
        rc, out, err = runner.run(
            "ray start --head --port=6379 --dashboard-host=0.0.0.0", timeout=30)
        if rc != 0:
            log_queue.put("[RAY ERROR] Head start fallito: %s\n" % (err.strip() or out.strip()))
            return
        log_queue.put("[RAY] Head node avviato.\n")

        # Get head IP
        rc, head_ip, _ = runner.run("hostname -I | awk '{print $1}'", timeout=5)
        head_ip = head_ip.strip()
        if not head_ip:
            log_queue.put("[RAY ERROR] Impossibile determinare IP head.\n")
            return

        # Connect each ready/installed worker
        workers = config.get_workers()
        connected = 0
        for ip, w in workers.items():
            if w.get("status") not in ("ready", "installed"):
                log_queue.put("[RAY] Skip worker %s (stato: %s)\n" % (ip, w.get("status", "?")))
                continue
            log_queue.put("[RAY] Connessione worker %s...\n" % ip)
            dgx_dir = detector.install_dir or DGX_INSTALL_DIR_DEFAULT
            activate = "source %s/.vllm/bin/activate" % dgx_dir
            rc, out, err = _ssh_cmd_b64(w["user"], ip,
                "%s && ray start --address=%s:6379" % (activate, head_ip), timeout=30)
            if rc == 0:
                config.update_worker_status(ip, "ray_connected")
                log_queue.put("[RAY] Worker %s connesso.\n" % ip)
                connected += 1
            else:
                log_queue.put("[RAY ERROR] Worker %s: %s\n" % (ip, err.strip() or out.strip()))

        log_queue.put("[RAY] Cluster avviato: head + %d worker connessi.\n" % connected)

    threading.Thread(target=_run, daemon=True).start()
    return jsonify({"status": "starting"})


@app.route("/api/dgx/ray/stop-cluster", methods=["POST"])
def api_dgx_ray_stop_cluster():
    """Stop Ray on all workers (via SSH), then stop local head."""
    def _run():
        workers = config.get_workers()
        # Stop workers first
        for ip, w in workers.items():
            if w.get("status") == "ray_connected":
                log_queue.put("[RAY] Stop worker %s...\n" % ip)
                dgx_dir = detector.install_dir or DGX_INSTALL_DIR_DEFAULT
                activate = "source %s/.vllm/bin/activate" % dgx_dir
                _ssh_cmd_b64(w["user"], ip, "%s && ray stop" % activate, timeout=15)
                config.update_worker_status(ip, "ready")
        # Stop head
        runner.run("ray stop", timeout=15)
        log_queue.put("[RAY] Cluster fermato.\n")

    threading.Thread(target=_run, daemon=True).start()
    return jsonify({"status": "stopping"})


# ---------------------------------------------------------------------------
# DGX Wizard API
# ---------------------------------------------------------------------------

# Wizard install state tracking
_wizard_install_state = {"status": "idle", "step": 0, "error": ""}


@app.route("/api/dgx/wizard/state")
def api_dgx_wizard_state():
    return jsonify({"wizard": config.get_wizard()})


@app.route("/api/dgx/wizard/state", methods=["POST"])
def api_dgx_wizard_state_save():
    config.data["wizard"] = request.json.get("wizard", {})
    config.save()
    return jsonify({"ok": True})


@app.route("/api/dgx/wizard/reset", methods=["POST"])
def api_dgx_wizard_reset():
    config.reset_wizard()
    global _wizard_install_state
    _wizard_install_state = {"status": "idle", "step": 0, "error": ""}
    return jsonify({"ok": True})


@app.route("/api/dgx/wizard/hw-detect")
def api_dgx_wizard_hw_detect():
    """Comprehensive hardware detection for DGX wizard."""
    results = {}

    # GPU
    rc, out, _ = runner.run(
        "nvidia-smi --query-gpu=name,memory.total,compute_cap --format=csv,noheader", timeout=10)
    if rc == 0 and out.strip():
        parts = [x.strip() for x in out.strip().split(",")]
        results["gpu"] = {
            "ok": True, "name": parts[0] if len(parts) > 0 else "?",
            "vram": parts[1] if len(parts) > 1 else "?",
            "compute_cap": parts[2] if len(parts) > 2 else "?"
        }
    else:
        results["gpu"] = {"ok": False, "info": "GPU non rilevata"}

    # CUDA toolkit
    rc, out, _ = runner.run(
        "nvcc --version 2>/dev/null || /usr/local/cuda/bin/nvcc --version 2>/dev/null", timeout=10)
    if rc == 0:
        import re as _re
        m = _re.search(r"release (\S+)", out)
        results["cuda"] = {"ok": True, "version": m.group(1) if m else out.strip()[:40]}
    else:
        results["cuda"] = {"ok": False, "info": "CUDA toolkit (nvcc) non trovato"}

    # Disk space
    rc, out, _ = runner.run("df -BG $HOME | tail -1 | awk '{print $4}' | sed 's/G//'", timeout=5)
    free_gb = int(out.strip()) if rc == 0 and out.strip().isdigit() else 0
    results["disk"] = {"ok": free_gb >= 50, "free_gb": free_gb,
                        "info": "%d GB liberi" % free_gb + (" (minimo 50 GB)" if free_gb < 50 else "")}

    # RAM
    rc, out, _ = runner.run("free -g | awk '/Mem:/{print $2}'", timeout=5)
    ram_gb = int(out.strip()) if rc == 0 and out.strip().isdigit() else 0
    results["ram"] = {"ok": ram_gb > 0, "total_gb": ram_gb, "info": "%d GB" % ram_gb}

    # Network interfaces (ibdev2netdev)
    rc, out, _ = runner.run("ibdev2netdev 2>/dev/null || echo 'ibdev2netdev non disponibile'", timeout=5)
    interfaces = []
    if rc == 0:
        for line in out.strip().split("\n"):
            if line.strip():
                interfaces.append(line.strip())
    results["network"] = {"ok": len(interfaces) > 0, "interfaces": interfaces}

    # IP addresses of high-speed interfaces
    rc, out, _ = runner.run("ip -br addr show 2>/dev/null | grep -E '(enP|enp|mlx)'", timeout=5)
    ip_addrs = []
    if rc == 0:
        for line in out.strip().split("\n"):
            if line.strip():
                ip_addrs.append(line.strip())
    results["ip_addresses"] = ip_addrs

    # NCCL availability
    rc, out, _ = runner.run(
        'python3 -c "import torch; print(torch.cuda.nccl.is_available()); '
        'print(torch.cuda.device_count())" 2>/dev/null', timeout=15)
    if rc == 0 and "True" in out:
        lines = out.strip().split("\n")
        dev_count = lines[1].strip() if len(lines) > 1 else "?"
        results["nccl"] = {"ok": True, "devices": dev_count}
    else:
        results["nccl"] = {"ok": False, "info": "NCCL non disponibile o torch non installato"}

    # Existing installation — use detector's path if available, fallback to default
    detected_dir = detector.install_dir or DGX_INSTALL_DIR_DEFAULT
    rc, out, _ = runner.run(
        "test -d %s && echo 'exists' || echo 'not_found'" % detected_dir, timeout=5)
    has_install = "exists" in out
    vllm_ver = detector.version or ""
    if has_install and not vllm_ver:
        rc2, out2, _ = runner.run(
            "source %s/.vllm/bin/activate 2>/dev/null && python -c \"import vllm; print(vllm.__version__)\" 2>/dev/null"
            % detected_dir, timeout=10)
        if rc2 == 0:
            vllm_ver = out2.strip()
    results["existing_install"] = {"found": has_install or detector.is_installed,
                                    "vllm_version": vllm_ver,
                                    "path": detected_dir,
                                    "venv_path": detector.venv_path or ""}

    return jsonify(results)


@app.route("/api/dgx/wizard/params-schema")
def api_dgx_wizard_params_schema():
    return jsonify(DGX_WIZARD_PARAMS_SCHEMA)


@app.route("/api/dgx/wizard/params-save", methods=["POST"])
def api_dgx_wizard_params_save():
    config.save_wizard_step("engine_params", request.json)
    return jsonify({"ok": True})


@app.route("/api/dgx/wizard/install-config", methods=["POST"])
def api_dgx_wizard_install_config():
    config.save_wizard_step("install_config", request.json)
    return jsonify({"ok": True})


@app.route("/api/dgx/wizard/install-start", methods=["POST"])
def api_dgx_wizard_install_start():
    global _wizard_install_state
    if _wizard_install_state.get("status") == "running":
        return jsonify({"error": "Installazione gia' in corso"}), 409

    wiz = config.get_wizard()
    ic = wiz.get("install_config", {})
    force = request.json.get("force", False) if request.json else False

    def _run_install():
        global _wizard_install_state
        _wizard_install_state = {"status": "running", "step": 0, "error": ""}
        try:
            _install_vllm_dgx(
                vllm_commit=ic.get("vllm_commit") or None,
                triton_commit=ic.get("triton_commit") or None,
                pytorch_index=ic.get("pytorch_index") or None,
                install_dir=ic.get("install_dir") or None,
                force=force,
            )
            _wizard_install_state["status"] = "done"
            config.save_wizard_step("install_status", "done")
        except Exception as e:
            _wizard_install_state["status"] = "error"
            _wizard_install_state["error"] = str(e)
            config.save_wizard_step("install_status", "error")

    threading.Thread(target=_run_install, daemon=True).start()
    return jsonify({"status": "started"})


@app.route("/api/dgx/wizard/install-status")
def api_dgx_wizard_install_status():
    return jsonify(_wizard_install_state)


@app.route("/api/dgx/wizard/install-check")
def api_dgx_wizard_install_check():
    """Quick check if vLLM+PyTorch are installed and working."""
    dgx_dir = detector.install_dir or DGX_INSTALL_DIR_DEFAULT
    venv = dgx_dir + "/.vllm"
    activate = "source %s/bin/activate" % venv
    rc, out, _ = runner.run(
        "test -f %s/bin/activate && %s && "
        "python -c \"import vllm; print(vllm.__version__)\" && "
        "python -c \"import torch; print(torch.__version__); print(torch.cuda.is_available())\"" % (venv, activate),
        timeout=20)
    if rc == 0:
        lines = out.strip().split("\n")
        vllm_ver = lines[0].strip() if len(lines) > 0 else ""
        torch_ver = lines[1].strip() if len(lines) > 1 else ""
        cuda_ok = "True" in (lines[2] if len(lines) > 2 else "")
        return jsonify({
            "installed": True, "vllm_version": vllm_ver,
            "torch_version": torch_ver, "cuda_available": cuda_ok,
            "path": dgx_dir,
        })
    return jsonify({"installed": False, "path": dgx_dir})


@app.route("/api/dgx/wizard/model-save", methods=["POST"])
def api_dgx_wizard_model_save():
    config.save_wizard_step("model_config", request.json)
    return jsonify({"ok": True})


@app.route("/api/dgx/wizard/model-download", methods=["POST"])
def api_dgx_wizard_model_download():
    model_id = request.json.get("model", "")
    if not model_id:
        return jsonify({"error": "Nessun modello specificato"}), 400

    def _run():
        log_queue.put("[WIZARD] Download modello: %s\n" % model_id)
        activate = ""
        if vllm_detector.activation_cmd:
            activate = vllm_detector.activation_cmd + " && "
        rc, out, err = runner.run(
            "%shuggingface-cli download '%s' 2>&1 || "
            "python -c \"from huggingface_hub import snapshot_download; snapshot_download('%s')\" 2>&1"
            % (activate, model_id, model_id), timeout=3600)
        if rc == 0:
            log_queue.put("[WIZARD] Download completato: %s\n" % model_id)
        else:
            log_queue.put("[ERROR] Download fallito: %s\n" % (err or out)[:200])

    threading.Thread(target=_run, daemon=True).start()
    return jsonify({"status": "downloading"})


@app.route("/api/dgx/wizard/net-detect")
def api_dgx_wizard_net_detect():
    """Detect network interfaces and suggest NCCL configuration."""
    results = {"interfaces": [], "suggested": {}}

    # ibdev2netdev
    rc, out, _ = runner.run("ibdev2netdev 2>/dev/null", timeout=5)
    if rc == 0:
        for line in out.strip().split("\n"):
            if line.strip():
                results["interfaces"].append(line.strip())
                # Find the Up interface
                if "(Up)" in line:
                    parts = line.split()
                    for i, p in enumerate(parts):
                        if p.startswith("en") or p.startswith("mlx"):
                            if_name = p
                            results["suggested"]["NCCL_SOCKET_IFNAME"] = if_name
                            results["suggested"]["UCX_NET_DEVICES"] = if_name
                            results["suggested"]["GLOO_SOCKET_IFNAME"] = if_name
                            results["suggested"]["TP_SOCKET_IFNAME"] = if_name
                            break
                    # IB HCA device
                    if len(parts) > 0 and parts[0].startswith("mlx"):
                        results["suggested"]["NCCL_IB_HCA"] = parts[0]

    # IP of interfaces
    rc, out, _ = runner.run("ip -br addr show 2>/dev/null | grep -E '(enP|enp|mlx)'", timeout=5)
    if rc == 0:
        results["ip_info"] = [l.strip() for l in out.strip().split("\n") if l.strip()]

    return jsonify(results)


@app.route("/api/dgx/wizard/ray-config", methods=["POST"])
def api_dgx_wizard_ray_config():
    config.save_wizard_step("ray_config", request.json)
    return jsonify({"ok": True})


@app.route("/api/dgx/wizard/verify-all", methods=["POST"])
def api_dgx_wizard_verify_all():
    """Comprehensive verification: local + remote workers."""
    results = {"local": {}, "workers": {}}

    # Local GPU
    rc, out, _ = runner.run(
        "nvidia-smi --query-gpu=name,memory.total --format=csv,noheader", timeout=10)
    results["local"]["gpu"] = {"ok": rc == 0, "info": out.strip() if rc == 0 else "GPU non rilevata"}

    # Local vLLM
    activate = ""
    if vllm_detector.activation_cmd:
        activate = vllm_detector.activation_cmd + " && "
    rc, out, _ = runner.run(
        '%spython -c "import vllm; print(vllm.__version__)" 2>/dev/null' % activate, timeout=15)
    results["local"]["vllm"] = {"ok": rc == 0, "info": out.strip() if rc == 0 else "vLLM non installato"}

    # Local Ray
    rc, out, _ = runner.run(
        '%spython -c "import ray; print(ray.__version__)" 2>/dev/null' % activate, timeout=15)
    results["local"]["ray"] = {"ok": rc == 0, "info": out.strip() if rc == 0 else "Ray non installato"}

    # Local NCCL
    rc, out, _ = runner.run(
        '%spython -c "import torch; print(torch.cuda.nccl.is_available())" 2>/dev/null' % activate,
        timeout=15)
    results["local"]["nccl"] = {"ok": rc == 0 and "True" in out,
                                 "info": "Disponibile" if (rc == 0 and "True" in out) else "Non disponibile"}

    # Remote workers
    workers = config.get_workers()
    for ip, w in workers.items():
        if w.get("status") in ("installed", "ready", "ray_connected"):
            wr = _verify_worker(ip, w.get("user", "root"))
            results["workers"][ip] = wr

    all_local_ok = all(v.get("ok", False) for v in results["local"].values())
    all_workers_ok = all(
        all(c.get("ok", False) for c in wr.get("results", {}).values())
        for wr in results["workers"].values()
    ) if results["workers"] else True

    results["all_passed"] = all_local_ok and all_workers_ok
    return jsonify(results)


@app.route("/api/dgx/wizard/launch", methods=["POST"])
def api_dgx_wizard_launch():
    """Launch vLLM server with full wizard configuration."""
    wiz = config.get_wizard()
    params = wiz.get("engine_params", {})
    model_cfg = wiz.get("model_config", {})
    ray_cfg = wiz.get("ray_config", {})

    model = params.get("model") or model_cfg.get("model_id", "")
    if not model:
        return jsonify({"error": "Nessun modello configurato"}), 400

    # Apply NCCL env vars if configured
    if ray_cfg and isinstance(runner, DGXCommandRunner):
        for key in ("NCCL_SOCKET_IFNAME", "UCX_NET_DEVICES", "GLOO_SOCKET_IFNAME",
                     "NCCL_IB_HCA", "TP_SOCKET_IFNAME"):
            val = ray_cfg.get(key)
            if val:
                runner.dgx_env[key] = val
        if ray_cfg.get("VLLM_HOST_IP"):
            runner.dgx_env["VLLM_HOST_IP"] = ray_cfg["VLLM_HOST_IP"]
        if ray_cfg.get("MASTER_ADDR"):
            runner.dgx_env["MASTER_ADDR"] = ray_cfg["MASTER_ADDR"]

    try:
        vllm_proc.start(
            model=model,
            gpu_mem_util=float(params.get("gpu_memory_utilization", 0.90)),
            max_model_len=int(params["max_model_len"]) if params.get("max_model_len") else None,
            dtype=params.get("dtype", "auto"),
            enable_prefix_caching=params.get("enable_prefix_caching", True),
            extra_args="",
            host=params.get("host", "0.0.0.0"),
            port=int(params.get("port", 8000)),
            chat_template=params.get("chat_template", ""),
            tp_size=int(params.get("tensor_parallel_size", 1)),
            attention_backend=params.get("attention_backend", ""),
            pp_size=int(params.get("pipeline_parallel_size", 1)),
            dp_size=int(params.get("data_parallel_size", 1)) if params.get("data_parallel_size") else None,
            max_num_seqs=int(params["max_num_seqs"]) if params.get("max_num_seqs") else None,
            max_num_batched_tokens=int(params["max_num_batched_tokens"]) if params.get("max_num_batched_tokens") else None,
            enable_chunked_prefill=params.get("enable_chunked_prefill", False),
            kv_cache_dtype=params.get("kv_cache_dtype"),
            block_size=int(params["block_size"]) if params.get("block_size") else None,
            cpu_offload_gb=float(params["cpu_offload_gb"]) if params.get("cpu_offload_gb") else None,
            trust_remote_code=params.get("trust_remote_code", False),
            enforce_eager=params.get("enforce_eager", False),
            quantization=params.get("quantization") or None,
            seed=int(params["seed"]) if params.get("seed") else None,
            tokenizer=params.get("tokenizer") or None,
            enable_lora=params.get("enable_lora", False),
            max_loras=int(params["max_loras"]) if params.get("max_loras") else None,
            max_lora_rank=int(params["max_lora_rank"]) if params.get("max_lora_rank") else None,
            speculative_model=params.get("speculative_model") or None,
            num_speculative_tokens=int(params["num_speculative_tokens"]) if params.get("num_speculative_tokens") else None,
            distributed_executor_backend=params.get("distributed_executor_backend") or None,
            nnodes=int(params.get("nnodes", 1)),
            scheduling_policy=params.get("scheduling_policy"),
            disable_log_stats=params.get("disable_log_stats", False),
            disable_cascade_attn=params.get("disable_cascade_attn", True),
            disable_sliding_window=params.get("disable_sliding_window", False),
            enable_expert_parallel=params.get("enable_expert_parallel", False),
        )
        return jsonify({"status": "started"})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ---------------------------------------------------------------------------
# HTML Template
# ---------------------------------------------------------------------------

HTML_TEMPLATE = r"""<!DOCTYPE html>
<html lang="it">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>vLLM Manager Web</title>
<style>
:root {
  --bg: #0f1117; --bg2: #1a1b26; --bg3: #24253a; --bg4: #2f3146;
  --fg: #c0caf5; --fg2: #a9b1d6; --fg3: #565f89;
  --accent: #7aa2f7; --green: #9ece6a; --red: #f7768e;
  --orange: #e0af68; --cyan: #7dcfff; --purple: #bb9af7;
  --border: #3b3d57; --radius: 8px;
}
* { box-sizing: border-box; margin: 0; padding: 0; }
body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', system-ui, sans-serif;
       background: var(--bg); color: var(--fg); min-height: 100vh; }
a { color: var(--accent); }

/* Header */
.header { background: var(--bg2); border-bottom: 1px solid var(--border);
           padding: 12px 24px; display: flex; align-items: center; gap: 16px; }
.header h1 { font-size: 18px; font-weight: 600; }
.header .badge { background: var(--bg4); padding: 3px 10px; border-radius: 12px;
                  font-size: 12px; color: var(--fg3); }
.status-dot { width: 10px; height: 10px; border-radius: 50%; display: inline-block; }
.status-dot.online { background: var(--green); box-shadow: 0 0 8px var(--green); }
.status-dot.offline { background: var(--red); }
.status-dot.loading { background: var(--orange); animation: pulse 1s infinite; }
@keyframes pulse { 50% { opacity: 0.4; } }
#status-text { font-size: 13px; color: var(--fg2); }

/* Tabs */
.tabs { display: flex; background: var(--bg2); border-bottom: 1px solid var(--border);
         padding: 0 16px; overflow-x: auto; }
.tab-btn { padding: 10px 20px; background: none; border: none; color: var(--fg3);
            cursor: pointer; font-size: 13px; font-weight: 500; border-bottom: 2px solid transparent;
            white-space: nowrap; transition: all 0.2s; }
.tab-btn:hover { color: var(--fg); background: var(--bg3); }
.tab-btn.active { color: var(--accent); border-bottom-color: var(--accent); }
.tab-panel { display: none; padding: 20px 24px; max-width: 1200px; }
.tab-panel.active { display: block; }

/* Cards */
.card { background: var(--bg2); border: 1px solid var(--border); border-radius: var(--radius);
         padding: 16px; margin-bottom: 16px; }
.card h3 { font-size: 14px; color: var(--fg2); margin-bottom: 12px; font-weight: 600;
             text-transform: uppercase; letter-spacing: 0.5px; }

/* Form elements */
label { font-size: 13px; color: var(--fg2); display: block; margin-bottom: 4px; }
input, select, textarea { background: var(--bg3); border: 1px solid var(--border);
    color: var(--fg); padding: 8px 12px; border-radius: 6px; font-size: 13px;
    width: 100%; outline: none; transition: border-color 0.2s; font-family: inherit; }
input:focus, select:focus, textarea:focus { border-color: var(--accent); }
select { cursor: pointer; }
.form-row { display: grid; grid-template-columns: 1fr 1fr; gap: 12px; margin-bottom: 12px; }
.form-row.three { grid-template-columns: 1fr 1fr 1fr; }
.form-group { margin-bottom: 12px; }
.checkbox-row { display: flex; align-items: center; gap: 8px; margin-bottom: 12px; }
.checkbox-row input[type="checkbox"] { width: auto; accent-color: var(--accent); }

/* Buttons */
.btn { padding: 8px 18px; border: none; border-radius: 6px; cursor: pointer;
        font-size: 13px; font-weight: 500; transition: all 0.2s; display: inline-flex;
        align-items: center; gap: 6px; }
.btn-primary { background: var(--accent); color: #1a1b26; }
.btn-primary:hover { filter: brightness(1.1); }
.btn-danger { background: var(--red); color: #1a1b26; }
.btn-danger:hover { filter: brightness(1.1); }
.btn-secondary { background: var(--bg4); color: var(--fg); }
.btn-secondary:hover { background: var(--border); }
.btn-success { background: var(--green); color: #1a1b26; }
.btn-warning { background: var(--orange); color: #1a1b26; }
.btn:disabled { opacity: 0.5; cursor: not-allowed; }
.btn-row { display: flex; gap: 8px; flex-wrap: wrap; }

/* Log console */
.console { background: #0d1117; border: 1px solid var(--border); border-radius: var(--radius);
            padding: 12px; font-family: 'SF Mono', 'Fira Code', 'Consolas', monospace;
            font-size: 12px; line-height: 1.6; overflow-y: auto; max-height: 400px;
            white-space: pre-wrap; word-break: break-all; }
.console .error { color: var(--red); }
.console .warning { color: var(--orange); }
.console .info { color: var(--cyan); }
.console .cmd { color: var(--purple); }
.console .install { color: var(--green); }

/* GPU bars */
.gpu-metric { margin-bottom: 10px; }
.gpu-metric .label-row { display: flex; justify-content: space-between; font-size: 12px; margin-bottom: 4px; }
.bar-bg { background: var(--bg); border-radius: 4px; height: 20px; overflow: hidden; }
.bar-fill { height: 100%; border-radius: 4px; transition: width 0.5s; }
.bar-fill.vram { background: linear-gradient(90deg, var(--accent), var(--cyan)); }
.bar-fill.util { background: linear-gradient(90deg, var(--green), var(--cyan)); }

/* GPU graph */
.gpu-graph { background: #0d1117; border: 1px solid var(--border); border-radius: var(--radius);
              height: 200px; position: relative; }

/* Table */
.data-table { width: 100%; border-collapse: collapse; font-size: 13px; }
.data-table th { text-align: left; padding: 8px 12px; background: var(--bg3);
                   color: var(--fg2); font-weight: 500; border-bottom: 1px solid var(--border); }
.data-table td { padding: 8px 12px; border-bottom: 1px solid var(--border); }
.data-table tr:hover td { background: var(--bg3); }
.data-table tr.selected td { background: var(--bg4); }
.data-table tr { cursor: pointer; }

/* Toast */
.toast { position: fixed; bottom: 20px; right: 20px; background: var(--bg3);
          border: 1px solid var(--border); border-radius: var(--radius);
          padding: 12px 20px; font-size: 13px; z-index: 1000;
          transform: translateY(100px); opacity: 0; transition: all 0.3s; }
.toast.show { transform: translateY(0); opacity: 1; }
.toast.success { border-color: var(--green); }
.toast.error { border-color: var(--red); }

/* Responsive */
@media (max-width: 768px) {
  .form-row { grid-template-columns: 1fr; }
  .form-row.three { grid-template-columns: 1fr; }
  .tab-panel { padding: 12px; }
}

/* Wizard */
.wizard-stepper { display: flex; align-items: center; justify-content: center; gap: 0; padding: 16px 0 24px; }
.wizard-step-ind { display: flex; flex-direction: column; align-items: center; gap: 4px; cursor: pointer; min-width: 80px; }
.step-circle { width: 36px; height: 36px; border-radius: 50%; display: flex; align-items: center; justify-content: center;
  font-weight: 700; font-size: 14px; border: 2px solid var(--fg3); color: var(--fg3); background: var(--bg2); transition: all .3s; }
.step-circle.active { border-color: var(--accent); color: var(--accent); background: rgba(122,162,247,.15); }
.step-circle.completed { border-color: var(--green); color: var(--bg); background: var(--green); }
.step-circle.error { border-color: var(--red); color: var(--bg); background: var(--red); }
.step-label { font-size: 11px; color: var(--fg3); white-space: nowrap; }
.step-label.active { color: var(--accent); font-weight: 600; }
.step-label.completed { color: var(--green); }
.step-line { flex: 1; height: 2px; background: var(--fg3); min-width: 20px; max-width: 60px; margin: 0 2px; margin-bottom: 18px; }
.step-line.completed { background: var(--green); }
.wizard-panel { display: none; }
.wizard-panel.active { display: block; }
.wizard-nav { display: flex; justify-content: space-between; margin-top: 20px; padding-top: 16px; border-top: 1px solid var(--border); }

.param-category { margin-bottom: 16px; }
.param-category-header { display: flex; align-items: center; gap: 8px; cursor: pointer; padding: 8px 12px;
  background: var(--bg3); border-radius: var(--radius); font-weight: 600; font-size: 14px; }
.param-category-header:hover { background: var(--bg4); }
.param-category-body { padding: 12px 0; }
.param-row { display: grid; grid-template-columns: 200px 1fr 28px; gap: 8px; align-items: center; padding: 6px 12px; }
.param-row:hover { background: rgba(122,162,247,.05); border-radius: 4px; }
.param-row label { font-size: 13px; color: var(--fg2); }
.param-row input, .param-row select { background: var(--bg2); border: 1px solid var(--border); color: var(--fg);
  padding: 6px 10px; border-radius: 4px; font-size: 13px; }
.param-row input[type="range"] { padding: 0; }
.param-row input[type="checkbox"] { width: 18px; height: 18px; }

.param-tip { display: inline-flex; align-items: center; justify-content: center; width: 20px; height: 20px;
  border-radius: 50%; background: var(--bg3); color: var(--fg3); font-size: 11px; font-weight: 700;
  cursor: help; position: relative; flex-shrink: 0; }
.param-tip:hover { background: var(--accent); color: var(--bg); }
.param-tip:hover::after { content: attr(data-tip); position: absolute; left: 28px; top: 50%; transform: translateY(-50%);
  background: var(--bg4); color: var(--fg); border: 1px solid var(--border); border-radius: 6px; padding: 10px 14px;
  font-size: 12px; font-weight: 400; line-height: 1.5; width: 340px; z-index: 999; white-space: pre-wrap;
  box-shadow: 0 4px 12px rgba(0,0,0,.4); pointer-events: none; }
.param-tip:hover::before { content: ''; position: absolute; left: 22px; top: 50%; transform: translateY(-50%);
  border: 6px solid transparent; border-right-color: var(--border); z-index: 999; }

.advanced-toggle { font-size: 12px; color: var(--accent); cursor: pointer; padding: 4px 12px; }
.advanced-toggle:hover { text-decoration: underline; }
.advanced-section { display: none; }
.advanced-section.show { display: block; }

.hw-check-item { display: flex; align-items: center; gap: 10px; padding: 8px 12px; border-bottom: 1px solid var(--bg3); }
.hw-check-icon { width: 22px; height: 22px; border-radius: 50%; display: flex; align-items: center; justify-content: center;
  font-size: 12px; font-weight: 700; flex-shrink: 0; }
.hw-check-icon.pass { background: var(--green); color: var(--bg); }
.hw-check-icon.fail { background: var(--red); color: var(--bg); }
.hw-check-icon.warn { background: var(--orange); color: var(--bg); }
.hw-check-label { font-size: 13px; font-weight: 600; min-width: 120px; }
.hw-check-value { font-size: 13px; color: var(--fg2); }

.wizard-summary { width: 100%; font-size: 13px; }
.wizard-summary td { padding: 4px 12px; border-bottom: 1px solid var(--bg3); }
.wizard-summary td:first-child { color: var(--fg3); width: 200px; }

.range-val { font-size: 12px; color: var(--accent); font-weight: 600; min-width: 40px; text-align: center; }
</style>
</head>
<body>

<!-- Header -->
<div class="header">
  <h1>vLLM Manager</h1>
  <span class="badge">{{ app_version }}</span>
  <span class="badge">{{ platform.display_name }} | {{ platform.arch }}</span>
  <div style="margin-left:auto; display:flex; align-items:center; gap:8px;">
    <span class="status-dot offline" id="status-dot"></span>
    <span id="status-text">Server: offline</span>
    <span id="gpu-status" style="font-size:12px; color:var(--fg3); margin-left:12px;"></span>
  </div>
</div>

<!-- Tabs -->
<div class="tabs">
  <button class="tab-btn active" onclick="switchTab('server')">Server</button>
  <button class="tab-btn" onclick="switchTab('gpu')">GPU Monitor</button>
  <button class="tab-btn" onclick="switchTab('logs')">Logs</button>
  <button class="tab-btn" onclick="switchTab('models')">Models</button>
  <button class="tab-btn" onclick="switchTab('benchmark')">Benchmark</button>
  <button class="tab-btn" onclick="switchTab('webui')">WebUI</button>
  <button class="tab-btn" onclick="switchTab('profiles')">Profili</button>
  {% if is_dgx %}
  <button class="tab-btn" onclick="switchTab('dgx-wizard')">Wizard DGX</button>
  <button class="tab-btn" onclick="switchTab('dgx')">DGX Spark</button>
  {% endif %}
</div>

<!-- Tab 1: Server -->
<div class="tab-panel active" id="tab-server">
  <div class="card">
    <h3>Modello</h3>
    <div class="form-group">
      <label>Modello (preset o custom HuggingFace ID)</label>
      <select id="model-select" onchange="onModelSelect()">
        {% for m in presets %}
        <option value="{{ m.name }}" data-vram="{{ m.vram }}" data-note="{{ m.note }}"
                data-tpl="{{ m.chat_template }}">{{ m.name }}</option>
        {% endfor %}
        <option value="">-- Custom --</option>
      </select>
    </div>
    <div class="form-group">
      <input type="text" id="model-custom" placeholder="oppure inserisci nome modello custom..."
             style="display:none">
    </div>
    <div id="model-info" style="font-size:12px; color:var(--fg3);"></div>
  </div>

  <div class="card">
    <h3>Parametri</h3>
    {% if not is_macos %}
    <div class="form-group">
      <label>GPU Memory Utilization: <span id="gpu-mem-val">0.90</span></label>
      <input type="range" id="gpu-mem" min="0.5" max="0.99" step="0.01" value="0.90"
             oninput="document.getElementById('gpu-mem-val').textContent=parseFloat(this.value).toFixed(2)">
    </div>
    {% endif %}
    <div class="form-row">
      <div>
        <label>Max Model Len (0=auto)</label>
        <input type="number" id="max-model-len" value="0" min="0">
      </div>
      <div>
        <label>Dtype</label>
        <select id="dtype">
          <option value="auto">auto</option>
          <option value="float16">float16</option>
          <option value="bfloat16">bfloat16</option>
          <option value="float32">float32</option>
        </select>
      </div>
    </div>
    <div class="form-row">
      <div>
        <label>Host</label>
        <input type="text" id="host" value="0.0.0.0">
      </div>
      <div>
        <label>Port</label>
        <input type="number" id="port" value="8000">
      </div>
    </div>
    <div class="form-row">
      <div>
        <label>Chat Template</label>
        <select id="chat-template">
          {% for t in chat_templates %}
          <option value="{{ t }}">{{ t }}</option>
          {% endfor %}
        </select>
      </div>
      <div>
        <label>TP Size</label>
        <input type="number" id="tp-size" value="1" min="1" max="8">
      </div>
    </div>
    {% if is_dgx %}
    <div class="form-group">
      <label>Attention Backend</label>
      <select id="attention-backend">
        <option value="(auto)">(auto)</option>
        <option value="TRITON_ATTN">TRITON_ATTN</option>
        <option value="FLASHINFER">FLASHINFER</option>
        <option value="XFORMERS">XFORMERS</option>
      </select>
    </div>
    {% endif %}
    <div class="checkbox-row">
      <input type="checkbox" id="prefix-caching" checked>
      <label for="prefix-caching" style="margin:0">Enable Prefix Caching</label>
    </div>
    <div class="form-group">
      <label>Extra Args</label>
      <input type="text" id="extra-args" placeholder="--arg1 value1 --arg2 value2">
    </div>
  </div>

  <div class="btn-row" style="margin-bottom:16px">
    <button class="btn btn-success" id="btn-start" onclick="startServer()">Start Server</button>
    <button class="btn btn-danger" id="btn-stop" onclick="stopServer()" disabled>Stop Server</button>
    <button class="btn btn-warning" onclick="stopAll()">Stoppa Tutto</button>
    <button class="btn btn-secondary" onclick="quickSaveProfile()">Salva Profilo</button>
    <button class="btn btn-secondary" onclick="installVllm()">Installa vLLM</button>
  </div>

  <div class="card">
    <h3>Log Preview</h3>
    <div class="console" id="server-log-preview" style="max-height:250px;"></div>
  </div>
</div>

<!-- Tab 2: GPU Monitor -->
<div class="tab-panel" id="tab-gpu">
  <div class="card">
    <h3 id="gpu-name">GPU: rilevamento...</h3>
    <div class="gpu-metric">
      <div class="label-row">
        <span id="vram-label-text">{{ platform.vram_label }}</span>
        <span id="vram-text">-- / -- MB</span>
      </div>
      <div class="bar-bg"><div class="bar-fill vram" id="vram-bar" style="width:0%"></div></div>
    </div>
    <div class="gpu-metric">
      <div class="label-row">
        <span>GPU Util</span>
        <span id="util-text">-- %</span>
      </div>
      <div class="bar-bg"><div class="bar-fill util" id="util-bar" style="width:0%"></div></div>
    </div>
    <div style="display:flex; gap:32px; margin-top:12px; font-size:14px;">
      <div>Temp: <strong id="temp-text" style="color:var(--green)">-- C</strong></div>
      <div>Power: <strong id="power-text">-- W</strong></div>
    </div>
  </div>
  <div class="card">
    <h3>Storico (ultimi 2 minuti)</h3>
    <div class="gpu-graph">
      <canvas id="gpu-canvas" style="width:100%;height:100%"></canvas>
    </div>
    <div style="display:flex;gap:16px;margin-top:8px;font-size:11px;color:var(--fg3)">
      <span style="color:var(--accent)">&#9632; VRAM %</span>
      <span style="color:var(--green)">&#9632; Util %</span>
      <span style="color:var(--red)">&#9632; Temp</span>
    </div>
  </div>
</div>

<!-- Tab 3: Logs -->
<div class="tab-panel" id="tab-logs">
  <div style="display:flex;gap:8px;margin-bottom:12px;align-items:center;">
    <input type="text" id="log-search" placeholder="Cerca nei log..." style="max-width:300px">
    <button class="btn btn-secondary" onclick="searchLogs()">Trova</button>
    <button class="btn btn-secondary" onclick="clearLogs()">Pulisci Log</button>
  </div>
  <div class="console" id="log-console" style="max-height:calc(100vh - 200px);min-height:400px;"></div>
</div>

<!-- Tab 4: Models -->
<div class="tab-panel" id="tab-models">
  <div class="card">
    <h3>Cerca su HuggingFace</h3>
    <div style="display:flex;gap:8px;margin-bottom:12px">
      <input type="text" id="hf-search" placeholder="es. Qwen, Llama, Mistral..."
             style="max-width:400px" onkeydown="if(event.key==='Enter')searchHF()">
      <button class="btn btn-primary" onclick="searchHF()">Cerca</button>
      <span id="hf-status" style="font-size:12px;color:var(--fg3);align-self:center"></span>
    </div>
    <div style="overflow-x:auto">
      <table class="data-table" id="hf-table">
        <thead><tr><th>Model ID</th><th style="text-align:right">Downloads</th><th style="text-align:right">Likes</th></tr></thead>
        <tbody></tbody>
      </table>
    </div>
    <button class="btn btn-secondary" style="margin-top:8px" onclick="useSelectedHFModel()">Usa questo modello</button>
  </div>
  <div class="card">
    <h3>Modelli scaricati (cache locale)</h3>
    <div style="overflow-x:auto">
      <table class="data-table" id="local-table">
        <thead><tr><th>Modello</th><th style="text-align:right">Dimensione</th></tr></thead>
        <tbody></tbody>
      </table>
    </div>
    <div class="btn-row" style="margin-top:8px">
      <button class="btn btn-secondary" onclick="refreshLocalModels()">Aggiorna Lista</button>
      <button class="btn btn-danger" onclick="deleteLocalModel()">Elimina Selezionato</button>
      <span id="local-status" style="font-size:12px;color:var(--fg3);align-self:center"></span>
    </div>
  </div>
</div>

<!-- Tab 5: Benchmark -->
<div class="tab-panel" id="tab-benchmark">
  <div class="card">
    <h3>Prompt</h3>
    <div class="form-group">
      <select id="bench-preset" onchange="onBenchPreset()">
        <option value="Spiega cos'e' Python in 2 frasi.">Breve - Spiega cos'e' Python</option>
        <option value="Scrivi una funzione Python che calcola i numeri di Fibonacci fino a n.">Media - Scrivi una funzione</option>
        <option value="Scrivi un articolo dettagliato di 500 parole sui vantaggi del machine learning.">Lunga - Scrivi un articolo</option>
        <option value="">Custom</option>
      </select>
    </div>
    <div class="form-group">
      <textarea id="bench-prompt" rows="3" style="font-family:monospace">Spiega cos'e' Python in 2 frasi.</textarea>
    </div>
    <div class="form-row">
      <div>
        <label>Max Tokens: <span id="bench-tok-val">256</span></label>
        <input type="range" id="bench-tokens" min="32" max="2048" value="256"
               oninput="document.getElementById('bench-tok-val').textContent=this.value">
      </div>
      <div>
        <label>Temperatura</label>
        <input type="number" id="bench-temp" value="0.7" step="0.1" min="0" max="2">
      </div>
    </div>
    <div class="btn-row">
      <button class="btn btn-primary" id="btn-bench" onclick="runBenchmark()">Esegui Benchmark</button>
      <span id="bench-status" style="font-size:12px;color:var(--fg3);align-self:center"></span>
    </div>
  </div>
  <div class="card">
    <h3>Risultati</h3>
    <table class="data-table" id="bench-results">
      <thead><tr><th>Metrica</th><th>Valore</th></tr></thead>
      <tbody></tbody>
    </table>
    <div class="console" id="bench-response" style="margin-top:12px;max-height:200px;"></div>
  </div>
</div>

<!-- Tab 6: WebUI -->
<div class="tab-panel" id="tab-webui">
  <div class="card">
    <h3>Open WebUI (Docker)</h3>
    <div style="font-size:14px;margin-bottom:12px">Status: <strong id="webui-status">sconosciuto</strong></div>
    <div class="btn-row" style="margin-bottom:12px">
      <button class="btn btn-success" onclick="startWebUI()">Start Open WebUI</button>
      <button class="btn btn-danger" onclick="stopWebUI()">Stop Open WebUI</button>
      <button class="btn btn-secondary" onclick="checkWebUIStatus()">Aggiorna Status</button>
    </div>
  </div>
  <div class="card">
    <h3>Configurazione</h3>
    <div class="form-row three">
      <div>
        <label>Container name</label>
        <input type="text" id="webui-container" value="open-webui">
      </div>
      <div>
        <label>Docker image</label>
        <input type="text" id="webui-image" value="ghcr.io/open-webui/open-webui:main">
      </div>
      <div>
        <label>Port</label>
        <input type="number" id="webui-port" value="3000">
      </div>
    </div>
  </div>
</div>

<!-- Tab 7: Profiles -->
<div class="tab-panel" id="tab-profiles">
  <div style="display:grid;grid-template-columns:1fr 1fr;gap:16px;">
    <div class="card">
      <h3>Profili Salvati</h3>
      <div id="profiles-list" style="min-height:200px;"></div>
    </div>
    <div class="card">
      <h3>Dettagli Profilo</h3>
      <pre class="console" id="profile-detail" style="min-height:150px;max-height:300px;"></pre>
      <div class="btn-row" style="margin-top:12px">
        <button class="btn btn-primary" onclick="loadProfile()">Carica</button>
        <button class="btn btn-danger" onclick="deleteProfile()">Elimina</button>
      </div>
    </div>
  </div>
</div>

{% if is_dgx %}
<!-- Tab: Wizard DGX -->
<div class="tab-panel" id="tab-dgx-wizard">

  <!-- Stepper -->
  <div class="wizard-stepper" id="wiz-stepper">
    <div class="wizard-step-ind" onclick="wizGoToStep(0)">
      <div class="step-circle active" id="wiz-sc-0">1</div>
      <div class="step-label active" id="wiz-sl-0">Hardware</div>
    </div><div class="step-line" id="wiz-line-0"></div>
    <div class="wizard-step-ind" onclick="wizGoToStep(1)">
      <div class="step-circle" id="wiz-sc-1">2</div>
      <div class="step-label" id="wiz-sl-1">Installazione</div>
    </div><div class="step-line" id="wiz-line-1"></div>
    <div class="wizard-step-ind" onclick="wizGoToStep(2)">
      <div class="step-circle" id="wiz-sc-2">3</div>
      <div class="step-label" id="wiz-sl-2">Parametri</div>
    </div><div class="step-line" id="wiz-line-2"></div>
    <div class="wizard-step-ind" onclick="wizGoToStep(3)">
      <div class="step-circle" id="wiz-sc-3">4</div>
      <div class="step-label" id="wiz-sl-3">Modello</div>
    </div><div class="step-line" id="wiz-line-3"></div>
    <div class="wizard-step-ind" onclick="wizGoToStep(4)">
      <div class="step-circle" id="wiz-sc-4">5</div>
      <div class="step-label" id="wiz-sl-4">Ray Cluster</div>
    </div><div class="step-line" id="wiz-line-4"></div>
    <div class="wizard-step-ind" onclick="wizGoToStep(5)">
      <div class="step-circle" id="wiz-sc-5">6</div>
      <div class="step-label" id="wiz-sl-5">Verifica</div>
    </div>
  </div>

  <!-- Step 0: Hardware Detection -->
  <div class="wizard-panel active" id="wiz-step-0">
    <div class="card">
      <h3>Rilevamento Hardware</h3>
      <p style="color:var(--fg3);margin-bottom:12px">Verifica automatica di GPU, CUDA, disco, RAM, rete e NCCL.</p>
      <button class="btn btn-primary" onclick="wizDetectHW()">Avvia Rilevamento</button>
      <div id="wiz-hw-results" style="margin-top:16px"></div>
    </div>
  </div>

  <!-- Step 1: Installation -->
  <div class="wizard-panel" id="wiz-step-1">
    <div class="card">
      <h3>Configurazione Installazione vLLM</h3>
      <p style="color:var(--fg3);margin-bottom:12px">Scegli le versioni dei componenti. I valori default sono testati per DGX Spark.</p>
      <div class="form-group">
        <label>vLLM Commit</label>
        <div style="display:flex;gap:8px;align-items:center">
          <input type="text" id="wiz-vllm-commit" value="{{ dgx_vllm_commit }}" style="flex:1">
          <span class="param-tip" data-tip="Commit hash del repository vLLM da compilare. Il default e' testato per Blackwell sm_121. Recuperabile da https://github.com/vllm-project/vllm/commits/main">?</span>
        </div>
      </div>
      <div class="form-group">
        <label>Triton Commit</label>
        <div style="display:flex;gap:8px;align-items:center">
          <input type="text" id="wiz-triton-commit" value="{{ dgx_triton_commit }}" style="flex:1">
          <span class="param-tip" data-tip="Commit hash di Triton con supporto sm_121a. Il default include le patch necessarie per Blackwell. Recuperabile da https://github.com/triton-lang/triton/commits/main">?</span>
        </div>
      </div>
      <div class="form-group">
        <label>PyTorch Index URL</label>
        <div style="display:flex;gap:8px;align-items:center">
          <input type="text" id="wiz-pytorch-index" value="{{ dgx_pytorch_index }}" style="flex:1">
          <span class="param-tip" data-tip="URL dell'indice pip per PyTorch con CUDA 13.0. Il default punta all'indice ufficiale cu130. Recuperabile da https://pytorch.org/get-started/locally/">?</span>
        </div>
      </div>
      <div class="form-group">
        <label>Directory Installazione</label>
        <div style="display:flex;gap:8px;align-items:center">
          <input type="text" id="wiz-install-dir" value="{{ dgx_install_dir }}" style="flex:1">
          <span class="param-tip" data-tip="Percorso dove verra' installato vLLM. Il venv Python sara' creato in <dir>/.vllm. Richiede almeno 50 GB di spazio libero.">?</span>
        </div>
      </div>
      <div id="wiz-install-check-box" style="margin-top:16px;padding:12px;border-radius:8px;border:1px solid var(--bg3);display:none">
        <div style="display:flex;align-items:center;gap:8px;margin-bottom:6px">
          <span id="wiz-install-check-icon" style="font-size:18px"></span>
          <strong id="wiz-install-check-title"></strong>
        </div>
        <div id="wiz-install-check-details" style="font-size:12px;color:var(--fg3)"></div>
      </div>
      <div class="btn-row" style="margin-top:16px">
        <button class="btn btn-primary" id="wiz-btn-install" onclick="wizStartInstall(false)">Avvia Installazione</button>
        <button class="btn" id="wiz-btn-force-install" onclick="wizStartInstall(true)" style="display:none;background:var(--orange);color:#000">Forza Reinstallazione</button>
        <span id="wiz-install-status" style="font-size:12px;color:var(--fg3);align-self:center"></span>
      </div>
      <div id="wiz-install-info" style="margin-top:8px;font-size:12px;color:var(--fg3)"></div>
    </div>
  </div>

  <!-- Step 2: Engine Parameters -->
  <div class="wizard-panel" id="wiz-step-2">
    <div class="card">
      <h3>Parametri Engine vLLM</h3>
      <p style="color:var(--fg3);margin-bottom:12px">Configura i parametri del motore vLLM. Passa il mouse sulla (?) per dettagli e raccomandazioni DGX.</p>
      <div id="wiz-params-container">Caricamento schema parametri...</div>
    </div>
  </div>

  <!-- Step 3: Model Selection -->
  <div class="wizard-panel" id="wiz-step-3">
    <div class="card">
      <h3>Selezione Modello</h3>
      <p style="color:var(--fg3);margin-bottom:12px">Scegli un modello preset ottimizzato per DGX Spark o cerca su HuggingFace.</p>

      <h4 style="margin:12px 0 8px;font-size:14px">Preset DGX Spark (singolo nodo)</h4>
      <table class="data-table">
        <thead><tr><th>Modello</th><th>VRAM</th><th>Note</th><th></th></tr></thead>
        <tbody>
        {% for m in presets %}
        <tr><td>{{ m.name }}</td><td>{{ m.vram }}</td><td>{{ m.note }}</td>
        <td><button class="btn btn-sm" onclick="wizSelectPreset('{{ m.name }}')">Seleziona</button></td></tr>
        {% endfor %}
        </tbody>
      </table>

      <h4 style="margin:16px 0 8px;font-size:14px">Preset Multi-Nodo</h4>
      <table class="data-table">
        <thead><tr><th>Modello</th><th>Nodi</th><th>TP</th><th>VRAM</th><th></th></tr></thead>
        <tbody>
        {% for m in multinode_presets %}
        <tr><td>{{ m.name }}</td><td>{{ m.nodes }}</td><td>{{ m.tp }}</td><td>{{ m.vram }}</td>
        <td><button class="btn btn-sm" onclick="wizSelectPreset('{{ m.name }}',{{ m.tp }},{{ m.nodes }})">Seleziona</button></td></tr>
        {% endfor %}
        </tbody>
      </table>

      <h4 style="margin:16px 0 8px;font-size:14px">Ricerca HuggingFace</h4>
      <div class="form-row">
        <div class="form-group">
          <input type="text" id="wiz-model-search" placeholder="Cerca modello..." onkeydown="if(event.key==='Enter')wizSearchModel()">
        </div>
        <div class="form-group">
          <button class="btn btn-primary" onclick="wizSearchModel()">Cerca</button>
        </div>
      </div>
      <div id="wiz-model-results" style="max-height:200px;overflow-y:auto"></div>

      <div class="form-group" style="margin-top:16px">
        <label>Modello Selezionato</label>
        <div style="display:flex;gap:8px">
          <input type="text" id="wiz-selected-model" style="flex:1" placeholder="Nessun modello selezionato">
          <button class="btn btn-secondary" onclick="wizDownloadModel()">Download</button>
        </div>
      </div>
    </div>
  </div>

  <!-- Step 4: Ray Cluster & Network -->
  <div class="wizard-panel" id="wiz-step-4">
    <div class="card">
      <h3>Configurazione Rete & Ray Cluster</h3>
      <p style="color:var(--fg3);margin-bottom:12px">Configura le variabili di rete NCCL e gestisci i worker remoti.</p>

      <h4 style="margin:12px 0 8px;font-size:14px">Rilevamento Interfacce di Rete</h4>
      <button class="btn btn-secondary" onclick="wizDetectNetwork()">Rileva Interfacce</button>
      <div id="wiz-net-results" style="margin-top:8px"></div>

      <h4 style="margin:16px 0 8px;font-size:14px">Variabili NCCL</h4>
      <div class="param-row">
        <label>NCCL_SOCKET_IFNAME</label>
        <input type="text" id="wiz-nccl-socket" placeholder="es. enP2p1s0f1np1">
        <span class="param-tip" data-tip="Interfaccia di rete per le comunicazioni collettive GPU-to-GPU via NCCL. Deve puntare all'interfaccia 200GbE QSFP, non alla porta di gestione 1GbE. Recuperare con: ibdev2netdev (cercare l'interfaccia Up)">?</span>
      </div>
      <div class="param-row">
        <label>UCX_NET_DEVICES</label>
        <input type="text" id="wiz-ucx-dev" placeholder="es. enP2p1s0f1np1">
        <span class="param-tip" data-tip="Dispositivo UCX per il trasporto dati. Tipicamente uguale a NCCL_SOCKET_IFNAME. UCX e' il layer di comunicazione usato da NCCL per le operazioni collettive.">?</span>
      </div>
      <div class="param-row">
        <label>GLOO_SOCKET_IFNAME</label>
        <input type="text" id="wiz-gloo-socket" placeholder="es. enp1s0f1np1">
        <span class="param-tip" data-tip="Interfaccia per il backend Gloo (usato per operazioni collettive CPU-side). Puo' essere la stessa di NCCL_SOCKET_IFNAME o un'altra interfaccia di rete.">?</span>
      </div>
      <div class="param-row">
        <label>NCCL_IB_HCA</label>
        <input type="text" id="wiz-nccl-hca" placeholder="es. mlx5_0,mlx5_1">
        <span class="param-tip" data-tip="Dispositivi InfiniBand HCA per NCCL. Su DGX Spark con ConnectX-7: tipicamente mlx5_0 e/o mlx5_1. Recuperare con: ibdev2netdev (prima colonna).">?</span>
      </div>
      <div class="param-row">
        <label>VLLM_HOST_IP</label>
        <input type="text" id="wiz-host-ip" placeholder="es. 192.168.100.10">
        <span class="param-tip" data-tip="IP di questo nodo sull'interfaccia multi-nodo. Deve essere l'IP assegnato all'interfaccia QSFP. Per il nodo head: il suo IP. Recuperare con: ip addr show <interfaccia>">?</span>
      </div>
      <div class="param-row">
        <label>MASTER_ADDR</label>
        <input type="text" id="wiz-master-addr" placeholder="es. 192.168.100.10">
        <span class="param-tip" data-tip="IP del nodo Ray head. Tutti i worker devono poter raggiungere questo indirizzo. Deve essere l'IP del nodo dove gira il Ray head sull'interfaccia QSFP.">?</span>
      </div>

      <h4 style="margin:16px 0 8px;font-size:14px">Worker Remoti</h4>
      <p style="color:var(--fg3);font-size:12px;margin-bottom:8px">I worker si gestiscono dal tab DGX Spark. I worker con stato "Pronto" o "Ray Connesso" saranno usati per il cluster.</p>
      <div id="wiz-workers-list" style="margin-top:8px"></div>
      <button class="btn btn-secondary" onclick="switchTab('dgx')" style="margin-top:8px">Gestisci Worker nel tab DGX Spark</button>
    </div>
  </div>

  <!-- Step 5: Verify & Launch -->
  <div class="wizard-panel" id="wiz-step-5">
    <div class="card">
      <h3>Verifica & Avvio</h3>
      <p style="color:var(--fg3);margin-bottom:12px">Verifica che tutti i componenti siano pronti, poi avvia il server vLLM.</p>

      <button class="btn btn-primary" onclick="wizVerifyAll()">Verifica Completa</button>
      <div id="wiz-verify-results" style="margin-top:16px"></div>

      <h4 style="margin:20px 0 8px;font-size:14px">Riepilogo Configurazione</h4>
      <div id="wiz-summary"></div>

      <div class="btn-row" style="margin-top:20px">
        <button class="btn btn-success" onclick="wizLaunch()" id="wiz-launch-btn" disabled>Avvia Server vLLM</button>
      </div>
    </div>
  </div>

  <!-- Navigation -->
  <div class="wizard-nav">
    <button class="btn btn-secondary" id="wiz-btn-back" onclick="wizBack()" style="visibility:hidden">Indietro</button>
    <button class="btn btn-primary" id="wiz-btn-next" onclick="wizNext()">Avanti</button>
  </div>
</div>

<!-- Tab 8: DGX Spark -->
<div class="tab-panel" id="tab-dgx">
  <!-- Cluster Overview -->
  <div class="card">
    <h3>Cluster Overview</h3>
    <div class="console" id="dgx-cluster" style="max-height:200px">Caricamento...</div>
    <div class="btn-row" style="margin-top:8px">
      <button class="btn btn-primary" onclick="refreshDGXCluster()">Aggiorna Stato Cluster</button>
      <span id="dgx-cluster-status" style="font-size:12px;color:var(--fg3);align-self:center"></span>
    </div>
  </div>

  <!-- Worker Management -->
  <div class="card">
    <h3>Gestione Worker Remoti</h3>
    <p style="font-size:12px;color:var(--fg3);margin-bottom:8px">Aggiungi i nodi worker e gestiscili interamente da qui (SSH, installazione vLLM, Ray).</p>
    <div class="form-row three" style="margin-bottom:8px">
      <div>
        <label>IP Worker</label>
        <input type="text" id="dgx-add-ip" placeholder="192.168.1.x">
      </div>
      <div>
        <label>User SSH</label>
        <input type="text" id="dgx-add-user" value="root">
      </div>
      <div>
        <label>Alias (opzionale)</label>
        <input type="text" id="dgx-add-alias" placeholder="dgx-worker-1">
      </div>
    </div>
    <button class="btn btn-primary" onclick="dgxAddWorker()" style="margin-bottom:12px">+ Aggiungi Worker</button>

    <div id="dgx-workers-table-wrap">
      <table class="data-table" id="dgx-workers-table">
        <thead><tr><th>IP</th><th>User</th><th>Alias</th><th>Stato</th><th>Azioni</th></tr></thead>
        <tbody id="dgx-workers-body">
          <tr><td colspan="5" style="text-align:center;color:var(--fg3)">Nessun worker configurato</td></tr>
        </tbody>
      </table>
    </div>
  </div>

  <!-- Node Discovery & Network -->
  <div class="card">
    <h3>Node Discovery & Network</h3>
    <div class="btn-row" style="margin-bottom:8px">
      <button class="btn btn-secondary" onclick="dgxDiscover()">Rileva Nodi</button>
      <button class="btn btn-secondary" onclick="dgxTestConnectivity()">Test Connettivita'</button>
      <button class="btn btn-secondary" onclick="dgxTestNCCL()">Test NCCL All-Reduce</button>
    </div>
    <div class="console" id="dgx-nodes" style="max-height:150px"></div>

    <h3 style="margin-top:16px">Variabili Ambiente NCCL</h3>
    <div class="form-row">
      <div>
        <label>NCCL_SOCKET_IFNAME</label>
        <input type="text" id="dgx-env-nccl-socket" value="enP2p1s0f1np1">
      </div>
      <div>
        <label>UCX_NET_DEVICES</label>
        <input type="text" id="dgx-env-ucx" value="enP2p1s0f1np1">
      </div>
    </div>
    <div class="form-row">
      <div>
        <label>GLOO_SOCKET_IFNAME</label>
        <input type="text" id="dgx-env-gloo" value="enp1s0f1np1">
      </div>
      <div>
        <label>NCCL_IB_HCA</label>
        <input type="text" id="dgx-env-ib" value="mlx5_0,mlx5_1">
      </div>
    </div>
    <button class="btn btn-secondary" onclick="dgxApplyEnv()" style="margin-top:4px">Applica Env Vars</button>
  </div>

  <!-- Ray Cluster -->
  <div class="card">
    <h3>Ray Cluster</h3>
    <div id="dgx-ray-status" style="font-size:13px;margin-bottom:8px;color:var(--fg2)">Ray: sconosciuto</div>
    <div class="btn-row">
      <button class="btn btn-success" onclick="dgxStartCluster()">Avvia Cluster Ray (Head + Worker)</button>
      <button class="btn btn-danger" onclick="dgxStopCluster()">Stop Cluster Ray</button>
    </div>
    <p style="font-size:11px;color:var(--fg3);margin-top:6px">Avvia Ray head su questo nodo e connette automaticamente tutti i worker con stato "pronto" o "installato".</p>
  </div>

  <!-- Multi-Node Inference -->
  <div class="card">
    <h3>Multi-Node Inference</h3>
    <div class="form-row three">
      <div>
        <label>Tensor Parallel Size</label>
        <input type="number" id="dgx-tp" value="1" min="1" max="8">
      </div>
      <div>
        <label>Pipeline Parallel Size</label>
        <input type="number" id="dgx-pp" value="1" min="1" max="8">
      </div>
      <div>
        <label>Attention Backend</label>
        <select id="dgx-attn">
          <option value="TRITON_ATTN">TRITON_ATTN</option>
          <option value="FLASHINFER">FLASHINFER</option>
          <option value="XFORMERS">XFORMERS</option>
        </select>
      </div>
    </div>

    <h3 style="margin-top:12px">Preset Multi-Nodo</h3>
    <table class="data-table">
      <thead><tr><th>Modello</th><th>Nodi</th><th>TP</th><th>VRAM</th></tr></thead>
      <tbody>
      {% for p in [
        {"name":"meta-llama/Llama-3.3-70B-Instruct","nodes":1,"tp":1,"vram":"~70 GB"},
        {"name":"meta-llama/Llama-3.1-405B-Instruct-FP8","nodes":4,"tp":4,"vram":"~400 GB"},
        {"name":"Qwen/Qwen2.5-72B-Instruct","nodes":1,"tp":1,"vram":"~72 GB"},
        {"name":"mistralai/Mixtral-8x22B-v0.1","nodes":2,"tp":2,"vram":"~180 GB"}
      ] %}
      <tr onclick="document.getElementById('model-select').value='';document.getElementById('model-custom').style.display='block';document.getElementById('model-custom').value='{{p.name}}';document.getElementById('tp-size').value='{{p.tp}}';document.getElementById('dgx-tp').value='{{p.tp}}';switchTab('server');">
        <td>{{p.name}}</td><td>{{p.nodes}}</td><td>{{p.tp}}</td><td>{{p.vram}}</td>
      </tr>
      {% endfor %}
      </tbody>
    </table>
    <button class="btn btn-success" style="margin-top:8px" onclick="dgxStartMultiNode()">Start Multi-Node Server</button>
  </div>
</div>
{% endif %}

<!-- Toast -->
<div class="toast" id="toast"></div>

<script>
// --- Tab switching ---
function switchTab(name) {
  document.querySelectorAll('.tab-panel').forEach(p => p.classList.remove('active'));
  document.querySelectorAll('.tab-btn').forEach(b => b.classList.remove('active'));
  document.getElementById('tab-' + name).classList.add('active');
  document.querySelectorAll('.tab-btn').forEach(b => {
    if (b.textContent.trim().toLowerCase().replace(/\s+/g,'') === name ||
        b.onclick.toString().includes("'" + name + "'"))
      b.classList.add('active');
  });
}

// --- Toast ---
function showToast(msg, type='') {
  const t = document.getElementById('toast');
  t.textContent = msg; t.className = 'toast show ' + type;
  setTimeout(() => t.className = 'toast', 3000);
}

// --- Helpers ---
async function apiPost(url, data={}) {
  const r = await fetch(url, {method:'POST', headers:{'Content-Type':'application/json'}, body:JSON.stringify(data)});
  return r.json();
}
async function apiGet(url) { return (await fetch(url)).json(); }

function getModel() {
  const sel = document.getElementById('model-select').value;
  const custom = document.getElementById('model-custom').value.trim();
  return sel || custom;
}

function getServerConfig() {
  return {
    model: getModel(),
    gpu_mem_util: parseFloat(document.getElementById('gpu-mem')?.value || 0.90),
    max_model_len: parseInt(document.getElementById('max-model-len').value) || 0,
    dtype: document.getElementById('dtype').value,
    host: document.getElementById('host').value,
    port: parseInt(document.getElementById('port').value),
    chat_template: document.getElementById('chat-template').value,
    tp_size: parseInt(document.getElementById('tp-size').value),
    prefix_caching: document.getElementById('prefix-caching').checked,
    extra_args: document.getElementById('extra-args').value,
    attention_backend: document.getElementById('attention-backend')?.value || '',
  };
}

// --- Model selection ---
function onModelSelect() {
  const sel = document.getElementById('model-select');
  const custom = document.getElementById('model-custom');
  const info = document.getElementById('model-info');
  if (!sel.value) {
    custom.style.display = 'block'; custom.focus();
    info.textContent = 'Custom model';
  } else {
    custom.style.display = 'none';
    const opt = sel.options[sel.selectedIndex];
    info.textContent = opt.dataset.vram + ' - ' + opt.dataset.note;
    const tpl = opt.dataset.tpl;
    if (tpl) document.getElementById('chat-template').value = tpl || '(auto)';
  }
}
onModelSelect();

// --- Server control ---
async function startServer() {
  const cfg = getServerConfig();
  if (!cfg.model) { showToast('Inserisci un modello!', 'error'); return; }
  document.getElementById('btn-start').disabled = true;
  document.getElementById('btn-stop').disabled = false;
  await apiPost('/api/server/start', cfg);
  showToast('Server in avvio...', 'success');
}

async function stopServer() {
  document.getElementById('btn-stop').disabled = true;
  await apiPost('/api/server/stop');
  showToast('Server in arresto...', 'success');
}

async function stopAll() {
  await apiPost('/api/server/stop-all');
  showToast('Arresto completo in corso...', 'success');
}

async function installVllm() {
  if (!confirm('Installare vLLM? Potrebbe richiedere qualche minuto.')) return;
  await apiPost('/api/server/install');
  showToast('Installazione in corso, controlla i log...', 'success');
}

// --- Server status polling ---
async function pollServerStatus() {
  try {
    const data = await apiGet('/api/server/status');
    const dot = document.getElementById('status-dot');
    const text = document.getElementById('status-text');
    if (data.online) {
      dot.className = 'status-dot online';
      text.textContent = 'Server: online';
      document.getElementById('btn-start').disabled = true;
      document.getElementById('btn-stop').disabled = false;
    } else if (data.running) {
      dot.className = 'status-dot loading';
      text.textContent = 'Server: caricamento...';
      document.getElementById('btn-start').disabled = true;
      document.getElementById('btn-stop').disabled = false;
    } else {
      dot.className = 'status-dot offline';
      text.textContent = 'Server: offline';
      document.getElementById('btn-start').disabled = false;
      document.getElementById('btn-stop').disabled = true;
    }
  } catch(e) {}
}
setInterval(pollServerStatus, 3000);
pollServerStatus();

// --- GPU polling ---
async function pollGPU() {
  try {
    const d = await apiGet('/api/gpu');
    if (!d.available) return;
    document.getElementById('gpu-name').textContent = 'GPU: ' + d.name;
    const pct = d.vram_total ? (d.vram_used / d.vram_total * 100) : 0;
    document.getElementById('vram-bar').style.width = pct + '%';
    document.getElementById('vram-text').textContent =
      d.vram_used + ' / ' + d.vram_total + ' MB  (' + pct.toFixed(0) + '%)';
    document.getElementById('util-bar').style.width = d.gpu_util + '%';
    document.getElementById('util-text').textContent = d.gpu_util + ' %';

    if (!d.is_macos) {
      const tempEl = document.getElementById('temp-text');
      tempEl.textContent = d.temp + ' C';
      tempEl.style.color = d.temp < 65 ? 'var(--green)' : d.temp < 80 ? 'var(--orange)' : 'var(--red)';
      document.getElementById('power-text').textContent = d.power.toFixed(1) + ' W';
    }
    document.getElementById('gpu-status').textContent =
      d.vram_used + '/' + d.vram_total + 'MB | ' + d.gpu_util + '% | ' + d.temp + 'C';

    drawGPUGraph(d);
  } catch(e) {}
}
setInterval(pollGPU, 2000);
pollGPU();

function drawGPUGraph(d) {
  const canvas = document.getElementById('gpu-canvas');
  const rect = canvas.parentElement.getBoundingClientRect();
  canvas.width = rect.width; canvas.height = rect.height;
  const ctx = canvas.getContext('2d');
  const w = canvas.width, h = canvas.height;
  ctx.clearRect(0,0,w,h);

  // Grid lines
  ctx.strokeStyle = '#333'; ctx.lineWidth = 0.5; ctx.setLineDash([2,4]);
  for (let i = 1; i < 4; i++) {
    const y = h * i / 4;
    ctx.beginPath(); ctx.moveTo(0,y); ctx.lineTo(w,y); ctx.stroke();
  }
  ctx.setLineDash([]);

  function drawLine(data, color) {
    if (data.length < 2) return;
    const max = Math.max(...data, 1);
    ctx.strokeStyle = color; ctx.lineWidth = 2; ctx.beginPath();
    data.forEach((v, i) => {
      const x = w * i / (data.length - 1);
      const y = h - (v / max) * (h - 10) - 5;
      i === 0 ? ctx.moveTo(x, y) : ctx.lineTo(x, y);
    });
    ctx.stroke();
  }
  drawLine(d.history_vram_pct || [], '#7aa2f7');
  drawLine(d.history_util || [], '#9ece6a');
  drawLine(d.history_temp || [], '#f7768e');
}

// --- Log streaming (SSE) ---
let logSSE = null;
function startLogStream() {
  if (logSSE) logSSE.close();
  // Load existing logs first
  apiGet('/api/logs/history').then(data => {
    const el = document.getElementById('log-console');
    const preview = document.getElementById('server-log-preview');
    el.innerHTML = '';
    (data.lines || []).forEach(line => appendLogLine(el, line));
    // Preview: last 30 lines
    preview.innerHTML = '';
    (data.lines || []).slice(-30).forEach(line => appendLogLine(preview, line));
    el.scrollTop = el.scrollHeight;
    preview.scrollTop = preview.scrollHeight;
  });

  logSSE = new EventSource('/api/logs/stream');
  logSSE.onmessage = function(e) {
    const line = JSON.parse(e.data);
    appendLogLine(document.getElementById('log-console'), line);
    const preview = document.getElementById('server-log-preview');
    appendLogLine(preview, line);
    // Auto-scroll
    const el = document.getElementById('log-console');
    if (el.scrollHeight - el.scrollTop - el.clientHeight < 100)
      el.scrollTop = el.scrollHeight;
    preview.scrollTop = preview.scrollHeight;
  };
}

function appendLogLine(container, line) {
  const span = document.createElement('span');
  const lower = line.toLowerCase();
  if (lower.includes('error')) span.className = 'error';
  else if (lower.includes('warning') || lower.includes('warn')) span.className = 'warning';
  else if (line.startsWith('[INFO]')) span.className = 'info';
  else if (line.startsWith('[CMD]')) span.className = 'cmd';
  else if (line.startsWith('[INSTALL]')) span.className = 'install';
  span.textContent = line;
  container.appendChild(span);
}

function searchLogs() {
  const term = document.getElementById('log-search').value.toLowerCase();
  const el = document.getElementById('log-console');
  el.querySelectorAll('span').forEach(s => {
    if (term && s.textContent.toLowerCase().includes(term))
      s.style.background = '#854d0e';
    else
      s.style.background = '';
  });
}

async function clearLogs() {
  await apiPost('/api/logs/clear');
  document.getElementById('log-console').innerHTML = '';
  document.getElementById('server-log-preview').innerHTML = '';
}

startLogStream();

// --- Models ---
let selectedHFModel = null;
let selectedLocalModel = null;

async function searchHF() {
  const q = document.getElementById('hf-search').value.trim();
  if (!q) return;
  document.getElementById('hf-status').textContent = 'Cercando...';
  const data = await apiGet('/api/models/search?q=' + encodeURIComponent(q));
  const tbody = document.querySelector('#hf-table tbody');
  tbody.innerHTML = '';
  (data.models || []).forEach(m => {
    const tr = document.createElement('tr');
    tr.innerHTML = '<td>' + m.id + '</td><td style="text-align:right">' +
      m.downloads.toLocaleString() + '</td><td style="text-align:right">' + m.likes + '</td>';
    tr.onclick = () => {
      tbody.querySelectorAll('tr').forEach(r => r.classList.remove('selected'));
      tr.classList.add('selected');
      selectedHFModel = m.id;
    };
    tbody.appendChild(tr);
  });
  document.getElementById('hf-status').textContent = (data.models||[]).length + ' risultati';
}

function useSelectedHFModel() {
  if (!selectedHFModel) { showToast('Seleziona un modello', 'error'); return; }
  document.getElementById('model-select').value = '';
  document.getElementById('model-custom').style.display = 'block';
  document.getElementById('model-custom').value = selectedHFModel;
  switchTab('server');
  showToast('Modello impostato: ' + selectedHFModel, 'success');
}

async function refreshLocalModels() {
  document.getElementById('local-status').textContent = 'Scansione...';
  const data = await apiGet('/api/models/local');
  const tbody = document.querySelector('#local-table tbody');
  tbody.innerHTML = '';
  (data.models || []).forEach(m => {
    const tr = document.createElement('tr');
    tr.innerHTML = '<td>' + m.name + '</td><td style="text-align:right">' + m.size + '</td>';
    tr.onclick = () => {
      tbody.querySelectorAll('tr').forEach(r => r.classList.remove('selected'));
      tr.classList.add('selected');
      selectedLocalModel = m.name;
    };
    tbody.appendChild(tr);
  });
  document.getElementById('local-status').textContent = (data.models||[]).length + ' modelli in cache';
}

async function deleteLocalModel() {
  if (!selectedLocalModel) { showToast('Seleziona un modello', 'error'); return; }
  if (!confirm('Eliminare ' + selectedLocalModel + ' dalla cache?')) return;
  const r = await apiPost('/api/models/delete', {name: selectedLocalModel});
  if (r.error) showToast(r.error, 'error');
  else { showToast('Eliminato!', 'success'); refreshLocalModels(); }
}

refreshLocalModels();

// --- Benchmark ---
function onBenchPreset() {
  const v = document.getElementById('bench-preset').value;
  if (v) document.getElementById('bench-prompt').value = v;
}

async function runBenchmark() {
  const prompt = document.getElementById('bench-prompt').value.trim();
  if (!prompt) return;
  document.getElementById('btn-bench').disabled = true;
  document.getElementById('bench-status').textContent = 'In esecuzione (streaming)...';
  try {
    const data = await apiPost('/api/benchmark', {
      prompt, model: getModel(),
      max_tokens: parseInt(document.getElementById('bench-tokens').value),
      temperature: parseFloat(document.getElementById('bench-temp').value),
    });
    const tbody = document.querySelector('#bench-results tbody');
    tbody.innerHTML = '';
    if (data.error) {
      tbody.innerHTML = '<tr><td>Errore</td><td>' + data.error + '</td></tr>';
    } else {
      [['Modello', data.model], ['Tokens generati', data.tokens],
       ['Tempo totale', data.elapsed + 's'], ['Tokens/sec', data.tps],
       ['TTFT', data.ttft_ms ? data.ttft_ms + ' ms' : 'N/A'],
       ['Max Tokens', data.max_tokens], ['Temperatura', data.temperature],
      ].forEach(([k,v]) => {
        tbody.innerHTML += '<tr><td>' + k + '</td><td>' + v + '</td></tr>';
      });
      document.getElementById('bench-response').textContent = data.text || '';
    }
    document.getElementById('bench-status').textContent = 'Completato';
  } catch(e) {
    document.getElementById('bench-status').textContent = 'Errore: ' + e.message;
  }
  document.getElementById('btn-bench').disabled = false;
}

// --- WebUI ---
async function checkWebUIStatus() {
  const data = await apiGet('/api/webui/status');
  document.getElementById('webui-status').textContent = data.status;
}
async function startWebUI() {
  await apiPost('/api/webui/start', {
    container: document.getElementById('webui-container').value,
    image: document.getElementById('webui-image').value,
    port: parseInt(document.getElementById('webui-port').value),
  });
  showToast('Avvio Open WebUI...', 'success');
  setTimeout(checkWebUIStatus, 3000);
}
async function stopWebUI() {
  await apiPost('/api/webui/stop', {
    container: document.getElementById('webui-container').value,
  });
  showToast('Arresto Open WebUI...', 'success');
  setTimeout(checkWebUIStatus, 3000);
}
checkWebUIStatus();

// --- Profiles ---
let selectedProfile = null;

async function refreshProfiles() {
  const data = await apiGet('/api/profiles');
  const el = document.getElementById('profiles-list');
  el.innerHTML = '';
  Object.keys(data.profiles || {}).forEach(name => {
    const div = document.createElement('div');
    div.textContent = name;
    div.style.cssText = 'padding:8px 12px;cursor:pointer;border-radius:6px;margin-bottom:4px;';
    div.onmouseenter = () => div.style.background = 'var(--bg3)';
    div.onmouseleave = () => { if (selectedProfile !== name) div.style.background = ''; };
    div.onclick = () => {
      selectedProfile = name;
      el.querySelectorAll('div').forEach(d => d.style.background = '');
      div.style.background = 'var(--bg4)';
      document.getElementById('profile-detail').textContent =
        JSON.stringify(data.profiles[name], null, 2);
    };
    el.appendChild(div);
  });
}

async function loadProfile() {
  if (!selectedProfile) { showToast('Seleziona un profilo', 'error'); return; }
  const data = await apiGet('/api/profiles');
  const p = data.profiles[selectedProfile];
  if (!p) return;
  // Apply profile to form
  if (p.model) {
    const sel = document.getElementById('model-select');
    const found = Array.from(sel.options).find(o => o.value === p.model);
    if (found) { sel.value = p.model; } else {
      sel.value = '';
      document.getElementById('model-custom').style.display = 'block';
      document.getElementById('model-custom').value = p.model;
    }
  }
  if (p.gpu_mem_util && document.getElementById('gpu-mem')) {
    document.getElementById('gpu-mem').value = p.gpu_mem_util;
    document.getElementById('gpu-mem-val').textContent = parseFloat(p.gpu_mem_util).toFixed(2);
  }
  if (p.max_model_len !== undefined) document.getElementById('max-model-len').value = p.max_model_len;
  if (p.dtype) document.getElementById('dtype').value = p.dtype;
  if (p.host) document.getElementById('host').value = p.host;
  if (p.port) document.getElementById('port').value = p.port;
  if (p.extra_args !== undefined) document.getElementById('extra-args').value = p.extra_args;
  if (p.chat_template) document.getElementById('chat-template').value = p.chat_template;
  if (p.prefix_caching !== undefined) document.getElementById('prefix-caching').checked = p.prefix_caching;
  switchTab('server');
  showToast('Profilo caricato: ' + selectedProfile, 'success');
}

async function quickSaveProfile() {
  const name = prompt('Nome profilo:');
  if (!name) return;
  const cfg = getServerConfig();
  await apiPost('/api/profiles/save', {name, profile: cfg});
  showToast('Profilo salvato!', 'success');
  refreshProfiles();
}

async function deleteProfile() {
  if (!selectedProfile) return;
  if (!confirm('Eliminare il profilo "' + selectedProfile + '"?')) return;
  await apiPost('/api/profiles/delete', {name: selectedProfile});
  selectedProfile = null;
  document.getElementById('profile-detail').textContent = '';
  refreshProfiles();
  showToast('Profilo eliminato', 'success');
}

refreshProfiles();

// --- DGX Spark ---
const DGX_STATUS_LABELS = {
  'added': {label: 'Aggiunto', color: '#888'},
  'ssh_ok': {label: 'SSH OK', color: '#5b9bd5'},
  'installing': {label: 'Installazione...', color: '#e6a817'},
  'installed': {label: 'Installato', color: '#d48806'},
  'ready': {label: 'Pronto', color: '#52c41a'},
  'ray_connected': {label: 'Ray Connesso', color: '#00d46a'},
  'error': {label: 'Errore', color: '#e74c3c'},
};

function dgxStatusBadge(status) {
  const s = DGX_STATUS_LABELS[status] || {label: status, color: '#888'};
  const pulse = status === 'installing' ? 'animation:pulse 1.5s infinite;' : '';
  return '<span style="display:inline-flex;align-items:center;gap:4px;font-size:12px;' + pulse + '">' +
    '<span style="width:8px;height:8px;border-radius:50%;background:' + s.color + ';display:inline-block"></span>' +
    s.label + '</span>';
}

function dgxWorkerActions(ip, status) {
  const btns = [];
  const sm = 'style="font-size:11px;padding:2px 8px"';
  if (status === 'added') {
    btns.push('<button class="btn btn-primary" ' + sm + ' onclick="dgxWorkerSSHSetup(\'' + ip + '\')">Setup SSH</button>');
  }
  if (status === 'ssh_ok') {
    btns.push('<button class="btn btn-success" ' + sm + ' onclick="dgxWorkerInstall(\'' + ip + '\')">Installa vLLM</button>');
    btns.push('<button class="btn btn-secondary" ' + sm + ' onclick="dgxWorkerVerify(\'' + ip + '\')">Verifica</button>');
  }
  if (status === 'installed') {
    btns.push('<button class="btn btn-secondary" ' + sm + ' onclick="dgxWorkerVerify(\'' + ip + '\')">Verifica</button>');
  }
  if (status === 'error') {
    btns.push('<button class="btn btn-primary" ' + sm + ' onclick="dgxWorkerSSHSetup(\'' + ip + '\')">Riprova SSH</button>');
    btns.push('<button class="btn btn-success" ' + sm + ' onclick="dgxWorkerInstall(\'' + ip + '\')">Riprova Installa</button>');
  }
  if (status !== 'installing' && status !== 'ray_connected') {
    btns.push('<button class="btn btn-danger" ' + sm + ' onclick="dgxWorkerRemove(\'' + ip + '\')">Rimuovi</button>');
  }
  if (status === 'ready' || status === 'ray_connected') {
    btns.push('<button class="btn btn-secondary" ' + sm + ' onclick="dgxWorkerVerify(\'' + ip + '\')">Verifica</button>');
  }
  return btns.join(' ');
}

let _dgxPollTimer = null;

async function dgxLoadWorkers() {
  try {
    const data = await apiGet('/api/dgx/workers');
    const workers = data.workers || {};
    const keys = Object.keys(workers);
    const tbody = document.getElementById('dgx-workers-body');
    if (keys.length === 0) {
      tbody.innerHTML = '<tr><td colspan="5" style="text-align:center;color:var(--fg3)">Nessun worker configurato</td></tr>';
    } else {
      tbody.innerHTML = keys.map(ip => {
        const w = workers[ip];
        return '<tr>' +
          '<td>' + w.ip + '</td>' +
          '<td>' + w.user + '</td>' +
          '<td>' + (w.alias || '') + '</td>' +
          '<td>' + dgxStatusBadge(w.status) + '</td>' +
          '<td>' + dgxWorkerActions(w.ip, w.status) + '</td>' +
          '</tr>';
      }).join('');
    }
    // Auto-poll faster while any worker is installing
    const anyInstalling = keys.some(ip => workers[ip].status === 'installing');
    clearInterval(_dgxPollTimer);
    _dgxPollTimer = setInterval(dgxLoadWorkers, anyInstalling ? 5000 : 15000);
  } catch(e) {}
}

async function dgxAddWorker() {
  const ip = document.getElementById('dgx-add-ip').value.trim();
  const user = document.getElementById('dgx-add-user').value.trim() || 'root';
  const alias = document.getElementById('dgx-add-alias').value.trim();
  if (!ip) { showToast('Inserisci IP del worker', 'error'); return; }
  await apiPost('/api/dgx/workers/add', {ip, user, alias});
  document.getElementById('dgx-add-ip').value = '';
  document.getElementById('dgx-add-alias').value = '';
  showToast('Worker ' + ip + ' aggiunto', 'success');
  dgxLoadWorkers();
}

async function dgxWorkerRemove(ip) {
  if (!confirm('Rimuovere il worker ' + ip + '?')) return;
  const data = await apiPost('/api/dgx/workers/remove', {ip});
  if (data.error) { showToast(data.error, 'error'); return; }
  showToast('Worker ' + ip + ' rimosso', 'success');
  dgxLoadWorkers();
}

async function dgxWorkerSSHSetup(ip) {
  showToast('Setup SSH verso ' + ip + '...', '');
  await apiPost('/api/dgx/workers/ssh-setup', {ip});
  showToast('Setup SSH avviato, controlla i log', 'success');
  setTimeout(dgxLoadWorkers, 3000);
}

async function dgxWorkerInstall(ip) {
  if (!confirm('Installare vLLM su ' + ip + '? L\'operazione richiede 20-40 minuti.')) return;
  const data = await apiPost('/api/dgx/workers/install', {ip});
  if (data.error) { showToast(data.error, 'error'); return; }
  showToast('Installazione avviata su ' + ip + ' — segui nei Log', 'success');
  dgxLoadWorkers();
}

async function dgxWorkerVerify(ip) {
  showToast('Verifica worker ' + ip + '...', '');
  const data = await apiPost('/api/dgx/workers/verify', {ip});
  if (data.status === 'ok') {
    showToast('Worker ' + ip + ': tutto OK', 'success');
  } else {
    const r = data.results || {};
    const fails = Object.keys(r).filter(k => !r[k].ok).map(k => k + ': ' + r[k].info);
    showToast('Worker ' + ip + ' — problemi: ' + fails.join(', '), 'error');
  }
  dgxLoadWorkers();
}

async function refreshDGXCluster() {
  document.getElementById('dgx-cluster-status').textContent = 'Rilevamento...';
  try {
    const data = await apiGet('/api/dgx/cluster');
    document.getElementById('dgx-cluster').textContent = data.text || 'Nessun dato';
    document.getElementById('dgx-cluster-status').textContent = 'Aggiornato';
  } catch(e) {
    document.getElementById('dgx-cluster-status').textContent = 'Errore: ' + e.message;
  }
}

async function dgxDiscover() {
  const el = document.getElementById('dgx-nodes');
  el.textContent = 'Scansione nodi...';
  try {
    const data = await apiGet('/api/dgx/discover');
    el.textContent = data.text || 'Nessun nodo rilevato.';
  } catch(e) {
    el.textContent = 'Errore: ' + e.message;
  }
}

async function dgxTestConnectivity() {
  showToast('Test connettivita in corso...', '');
  const data = await apiGet('/api/dgx/connectivity');
  showToast('Connettivita: ' + data.status, data.status === 'OK' ? 'success' : 'error');
}

async function dgxTestNCCL() {
  showToast('Test NCCL in corso...', '');
  const data = await apiGet('/api/dgx/nccl-test');
  document.getElementById('dgx-nodes').textContent = data.text;
}

async function dgxApplyEnv() {
  await apiPost('/api/dgx/env', {
    NCCL_SOCKET_IFNAME: document.getElementById('dgx-env-nccl-socket').value,
    UCX_NET_DEVICES: document.getElementById('dgx-env-ucx').value,
    GLOO_SOCKET_IFNAME: document.getElementById('dgx-env-gloo').value,
    NCCL_IB_HCA: document.getElementById('dgx-env-ib').value,
  });
  showToast('Variabili ambiente aggiornate', 'success');
}

async function dgxStartCluster() {
  showToast('Avvio cluster Ray (head + worker)...', 'success');
  document.getElementById('dgx-ray-status').textContent = 'Ray: avvio cluster in corso...';
  await apiPost('/api/dgx/ray/start-cluster');
  setTimeout(() => { refreshDGXCluster(); dgxLoadWorkers(); }, 8000);
}

async function dgxStopCluster() {
  showToast('Stop cluster Ray...', '');
  document.getElementById('dgx-ray-status').textContent = 'Ray: arresto in corso...';
  await apiPost('/api/dgx/ray/stop-cluster');
  setTimeout(() => {
    document.getElementById('dgx-ray-status').textContent = 'Ray: fermato';
    refreshDGXCluster();
    dgxLoadWorkers();
  }, 3000);
}

async function dgxStartMultiNode() {
  const model = getModel();
  if (!model) { showToast('Seleziona un modello', 'error'); return; }
  const tp = parseInt(document.getElementById('dgx-tp').value);
  const pp = parseInt(document.getElementById('dgx-pp').value);
  const attn = document.getElementById('dgx-attn').value;
  const extra = pp > 1 ? '--pipeline-parallel-size ' + pp : '';
  await apiPost('/api/server/start', {
    model, tp_size: tp, attention_backend: attn,
    extra_args: extra,
    gpu_mem_util: parseFloat(document.getElementById('gpu-mem')?.value || 0.90),
    dtype: document.getElementById('dtype').value,
    host: '0.0.0.0',
    port: parseInt(document.getElementById('port').value),
  });
  showToast('Avvio multi-node server: TP=' + tp + ' PP=' + pp, 'success');
  switchTab('server');
}

// Auto-load DGX on page load
if (document.getElementById('tab-dgx')) {
  setTimeout(() => { refreshDGXCluster(); dgxLoadWorkers(); }, 1500);
}

// =========================================================================
// DGX Wizard
// =========================================================================

let wizState = {};
let wizCurrentStep = 0;
const WIZ_TOTAL_STEPS = 6;
let wizParamsSchema = null;
let wizCompletedSteps = new Set();

async function wizInit() {
  try {
    const data = await apiGet('/api/dgx/wizard/state');
    wizState = (data && data.wizard) || {};
    wizCurrentStep = wizState.current_step || 0;
    if (wizState.completed_steps) {
      wizCompletedSteps = new Set(wizState.completed_steps);
    }
    wizUpdateStepper();
    wizShowStep(wizCurrentStep);
  } catch(e) { /* fresh start */ }
}

function wizShowStep(n) {
  for (let i = 0; i < WIZ_TOTAL_STEPS; i++) {
    const panel = document.getElementById('wiz-step-' + i);
    if (panel) panel.classList.toggle('active', i === n);
  }
  wizCurrentStep = n;
  // Nav buttons
  const back = document.getElementById('wiz-btn-back');
  const next = document.getElementById('wiz-btn-next');
  if (back) back.style.visibility = n === 0 ? 'hidden' : 'visible';
  if (next) {
    next.textContent = n === WIZ_TOTAL_STEPS - 1 ? 'Fine' : 'Avanti';
    next.style.display = n === WIZ_TOTAL_STEPS - 1 ? 'none' : '';
  }
  wizUpdateStepper();
  // Auto-load for specific steps
  if (n === 1) wizCheckInstallation();
  if (n === 2 && !wizParamsSchema) wizLoadParamsSchema();
  if (n === 4) wizLoadWorkersList();
  if (n === 5) wizRenderSummary();
}

function wizUpdateStepper() {
  for (let i = 0; i < WIZ_TOTAL_STEPS; i++) {
    const circle = document.getElementById('wiz-sc-' + i);
    const label = document.getElementById('wiz-sl-' + i);
    const line = document.getElementById('wiz-line-' + i);
    if (!circle) continue;
    circle.className = 'step-circle';
    label.className = 'step-label';
    if (wizCompletedSteps.has(i)) {
      circle.classList.add('completed');
      circle.innerHTML = '&#10003;';
      label.classList.add('completed');
    } else if (i === wizCurrentStep) {
      circle.classList.add('active');
      circle.innerHTML = '' + (i + 1);
      label.classList.add('active');
    } else {
      circle.innerHTML = '' + (i + 1);
    }
    if (line) {
      line.className = 'step-line' + (wizCompletedSteps.has(i) ? ' completed' : '');
    }
  }
}

function wizGoToStep(n) {
  if (n <= wizCurrentStep || wizCompletedSteps.has(n) || wizCompletedSteps.has(n - 1) || n === 0) {
    wizShowStep(n);
  }
}

async function wizSaveState() {
  wizState.current_step = wizCurrentStep;
  wizState.completed_steps = Array.from(wizCompletedSteps);
  try {
    await apiPost('/api/dgx/wizard/state', {wizard: wizState});
  } catch(e) {}
}

function wizNext() {
  wizCompletedSteps.add(wizCurrentStep);
  // Save step data
  wizSaveCurrentStepData();
  if (wizCurrentStep < WIZ_TOTAL_STEPS - 1) {
    wizShowStep(wizCurrentStep + 1);
  }
  wizSaveState();
}

function wizBack() {
  if (wizCurrentStep > 0) {
    wizShowStep(wizCurrentStep - 1);
  }
}

function wizSaveCurrentStepData() {
  if (wizCurrentStep === 1) wizSaveInstallConfig();
  if (wizCurrentStep === 2) wizSaveParams();
  if (wizCurrentStep === 3) wizSaveModel();
  if (wizCurrentStep === 4) wizSaveRayConfig();
}

// --- Step 0: Hardware Detection ---
async function wizDetectHW() {
  const container = document.getElementById('wiz-hw-results');
  container.innerHTML = '<div style="color:var(--fg3)">Rilevamento in corso...</div>';
  try {
    const data = await apiGet('/api/dgx/wizard/hw-detect');
    let html = '';
    // GPU
    const gpu = data.gpu || {};
    html += hwCheckItem(gpu.ok, 'GPU', gpu.ok ? `${gpu.name} (${gpu.vram}, Compute ${gpu.compute_cap})` : (gpu.info || 'Non rilevata'));
    // CUDA
    const cuda = data.cuda || {};
    html += hwCheckItem(cuda.ok, 'CUDA Toolkit', cuda.ok ? `Versione ${cuda.version}` : (cuda.info || 'Non trovato'));
    // Disk
    const disk = data.disk || {};
    html += hwCheckItem(disk.ok, 'Disco', disk.info || `${disk.free_gb} GB liberi`);
    // RAM
    const ram = data.ram || {};
    html += hwCheckItem(ram.ok, 'RAM', ram.info || `${ram.total_gb} GB`);
    // NCCL
    const nccl = data.nccl || {};
    html += hwCheckItem(nccl.ok, 'NCCL', nccl.ok ? `Disponibile (${nccl.devices} device)` : (nccl.info || 'Non disponibile'));
    // Network
    const net = data.network || {};
    html += hwCheckItem(net.ok, 'Rete', net.ok ? `${net.interfaces.length} interfacce rilevate` : 'Nessuna interfaccia');
    if (net.interfaces && net.interfaces.length > 0) {
      html += '<div style="margin-left:32px;font-size:12px;color:var(--fg3);padding:4px 12px">';
      net.interfaces.forEach(i => { html += i + '<br>'; });
      html += '</div>';
    }
    // Existing install
    const inst = data.existing_install || {};
    if (inst.found) {
      html += hwCheckItem(true, 'Installazione', `Trovata in ${inst.path}` + (inst.vllm_version ? ` (vLLM ${inst.vllm_version})` : ''));
    } else {
      html += hwCheckItem(false, 'Installazione', 'Nessuna installazione trovata', 'warn');
    }
    container.innerHTML = html;
    wizState.hw_detection = data;
  } catch(e) {
    container.innerHTML = '<div style="color:var(--red)">Errore: ' + e.message + '</div>';
  }
}

function hwCheckItem(ok, label, value, level) {
  const cls = level === 'warn' ? 'warn' : (ok ? 'pass' : 'fail');
  const icon = ok ? '&#10003;' : (level === 'warn' ? '!' : '&#10007;');
  return `<div class="hw-check-item">
    <div class="hw-check-icon ${cls}">${icon}</div>
    <div class="hw-check-label">${label}</div>
    <div class="hw-check-value">${value}</div>
  </div>`;
}

// --- Step 1: Installation ---
async function wizSaveInstallConfig() {
  const data = {
    vllm_commit: document.getElementById('wiz-vllm-commit')?.value || '',
    triton_commit: document.getElementById('wiz-triton-commit')?.value || '',
    pytorch_index: document.getElementById('wiz-pytorch-index')?.value || '',
    install_dir: document.getElementById('wiz-install-dir')?.value || '',
  };
  wizState.install_config = data;
  try { await apiPost('/api/dgx/wizard/install-config', data); } catch(e) {}
}

async function wizCheckInstallation() {
  const box = document.getElementById('wiz-install-check-box');
  const icon = document.getElementById('wiz-install-check-icon');
  const title = document.getElementById('wiz-install-check-title');
  const details = document.getElementById('wiz-install-check-details');
  const btnInstall = document.getElementById('wiz-btn-install');
  const btnForce = document.getElementById('wiz-btn-force-install');
  box.style.display = 'block';
  icon.textContent = '...';
  title.textContent = 'Verifica installazione in corso...';
  details.textContent = '';
  try {
    const r = await apiGet('/api/dgx/wizard/install-check');
    if (r.installed) {
      icon.textContent = '\u2705';
      title.textContent = 'vLLM gia\u0027 installato';
      title.style.color = 'var(--green)';
      let det = 'vLLM ' + r.vllm_version + ' | PyTorch ' + r.torch_version;
      det += ' | CUDA: ' + (r.cuda_available ? 'OK' : 'NON DISPONIBILE');
      det += ' | Path: ' + r.path;
      details.textContent = det;
      if (!r.cuda_available) {
        details.style.color = 'var(--orange)';
        details.textContent += ' \u26a0\ufe0f PyTorch senza CUDA, reinstallazione consigliata';
      }
      btnInstall.textContent = 'Salta (gia\u0027 installato)';
      btnInstall.onclick = function() { wizShowStep(2); };
      btnForce.style.display = '';
    } else {
      icon.textContent = '\u26a0\ufe0f';
      title.textContent = 'vLLM non trovato';
      title.style.color = 'var(--orange)';
      details.textContent = 'Path verificato: ' + r.path;
      btnInstall.textContent = 'Avvia Installazione';
      btnInstall.onclick = function() { wizStartInstall(false); };
      btnForce.style.display = 'none';
    }
  } catch(e) {
    icon.textContent = '\u274c';
    title.textContent = 'Errore verifica';
    title.style.color = 'var(--red)';
    details.textContent = e.message;
  }
}

async function wizStartInstall(force) {
  await wizSaveInstallConfig();
  const statusEl = document.getElementById('wiz-install-status');
  const infoEl = document.getElementById('wiz-install-info');
  statusEl.textContent = force ? 'Reinstallazione avviata...' : 'Installazione avviata...';
  statusEl.style.color = 'var(--orange)';
  infoEl.textContent = 'Questa operazione richiede 20-30 minuti. Controlla i log nel tab Logs.';
  try {
    await apiPost('/api/dgx/wizard/install-start', { force: !!force });
    // Poll status
    const poll = setInterval(async () => {
      try {
        const s = await apiGet('/api/dgx/wizard/install-status');
        if (s.status === 'done') {
          clearInterval(poll);
          statusEl.textContent = 'Installazione completata!';
          statusEl.style.color = 'var(--green)';
          infoEl.textContent = '';
          showToast('Installazione vLLM completata', 'success');
          wizCheckInstallation();
        } else if (s.status === 'error') {
          clearInterval(poll);
          statusEl.textContent = 'Errore installazione';
          statusEl.style.color = 'var(--red)';
          infoEl.textContent = s.error || '';
          showToast('Errore installazione: ' + (s.error || ''), 'error');
        } else {
          statusEl.textContent = 'Installazione in corso...';
        }
      } catch(e) {}
    }, 5000);
  } catch(e) {
    statusEl.textContent = 'Errore: ' + e.message;
    statusEl.style.color = 'var(--red)';
  }
}

// --- Step 2: Engine Parameters ---
async function wizLoadParamsSchema() {
  try {
    const schema = await apiGet('/api/dgx/wizard/params-schema');
    wizParamsSchema = schema;
    const container = document.getElementById('wiz-params-container');
    let html = '';
    schema.forEach(cat => {
      const basicParams = cat.params.filter(p => !p.advanced);
      const advParams = cat.params.filter(p => p.advanced);
      html += `<div class="param-category">
        <div class="param-category-header" onclick="this.nextElementSibling.classList.toggle('show');this.nextElementSibling.nextElementSibling?.classList.toggle('show')">
          <span>${cat.label}</span><span style="color:var(--fg3);font-size:12px">(${cat.params.length} parametri)</span>
        </div>
        <div class="param-category-body show">`;
      basicParams.forEach(p => { html += wizRenderParam(p); });
      html += '</div>';
      if (advParams.length > 0) {
        html += `<div class="param-category-body">`;
        html += `<div class="advanced-toggle" onclick="this.parentElement.classList.toggle('show')">Mostra Avanzati (${advParams.length})</div>`;
        advParams.forEach(p => { html += wizRenderParam(p); });
        html += '</div>';
      }
      html += '</div>';
    });
    container.innerHTML = html;
    // Restore saved values
    wizRestoreParams();
  } catch(e) {
    document.getElementById('wiz-params-container').innerHTML = '<div style="color:var(--red)">Errore caricamento schema: ' + e.message + '</div>';
  }
}

function wizRenderParam(p) {
  const saved = (wizState.engine_params || {})[p.key];
  const val = saved !== undefined ? saved : p.default;
  const tipText = (p.tooltip || '') + (p.dgx_tip ? '\\n\\nDGX Spark: ' + p.dgx_tip : '');
  let inputHtml = '';
  if (p.type === 'select') {
    const opts = (p.options || []).map(o => {
      const sel = String(o) === String(val) ? ' selected' : '';
      const label = o === '' ? '(auto)' : o;
      return `<option value="${o}"${sel}>${label}</option>`;
    }).join('');
    inputHtml = `<select id="wiz-p-${p.key}">${opts}</select>`;
  } else if (p.type === 'checkbox') {
    const chk = val ? ' checked' : '';
    inputHtml = `<input type="checkbox" id="wiz-p-${p.key}"${chk}>`;
  } else if (p.type === 'range') {
    inputHtml = `<div style="display:flex;gap:8px;align-items:center">
      <input type="range" id="wiz-p-${p.key}" min="${p.min||0}" max="${p.max||1}" step="${p.step||0.01}" value="${val}"
        oninput="document.getElementById('wiz-rv-${p.key}').textContent=this.value">
      <span class="range-val" id="wiz-rv-${p.key}">${val}</span></div>`;
  } else {
    const typeAttr = p.type === 'number' ? 'number' : 'text';
    const minMax = p.type === 'number' ? ` min="${p.min||''}" max="${p.max||''}"` : '';
    inputHtml = `<input type="${typeAttr}" id="wiz-p-${p.key}" value="${val || ''}"${minMax}>`;
  }
  return `<div class="param-row">
    <label for="wiz-p-${p.key}">${p.label}${p.required ? ' *' : ''}</label>
    ${inputHtml}
    <span class="param-tip" data-tip="${tipText.replace(/"/g, '&quot;')}">?</span>
  </div>`;
}

function wizRestoreParams() {
  if (!wizState.engine_params) return;
  Object.entries(wizState.engine_params).forEach(([key, val]) => {
    const el = document.getElementById('wiz-p-' + key);
    if (!el) return;
    if (el.type === 'checkbox') el.checked = !!val;
    else el.value = val;
    const rv = document.getElementById('wiz-rv-' + key);
    if (rv) rv.textContent = val;
  });
}

async function wizSaveParams() {
  if (!wizParamsSchema) return;
  const params = {};
  wizParamsSchema.forEach(cat => {
    cat.params.forEach(p => {
      const el = document.getElementById('wiz-p-' + p.key);
      if (!el) return;
      if (p.type === 'checkbox') params[p.key] = el.checked;
      else if (p.type === 'number' || p.type === 'range') params[p.key] = el.value ? parseFloat(el.value) : p.default;
      else params[p.key] = el.value;
    });
  });
  wizState.engine_params = params;
  try { await apiPost('/api/dgx/wizard/params-save', params); } catch(e) {}
}

// --- Step 3: Model Selection ---
function wizSelectPreset(modelId, tp, nodes) {
  document.getElementById('wiz-selected-model').value = modelId;
  wizState.model_config = {model_id: modelId};
  if (tp) {
    const tpEl = document.getElementById('wiz-p-tensor_parallel_size');
    if (tpEl) tpEl.value = tp;
  }
  if (nodes) {
    const nnEl = document.getElementById('wiz-p-nnodes');
    if (nnEl) nnEl.value = nodes;
  }
  showToast('Modello selezionato: ' + modelId, 'info');
}

async function wizSearchModel() {
  const q = document.getElementById('wiz-model-search').value.trim();
  if (!q) return;
  const container = document.getElementById('wiz-model-results');
  container.innerHTML = '<div style="color:var(--fg3)">Ricerca...</div>';
  try {
    const data = await apiGet('/api/models/search?q=' + encodeURIComponent(q));
    if (!data.results || data.results.length === 0) {
      container.innerHTML = '<div style="color:var(--fg3)">Nessun risultato</div>';
      return;
    }
    let html = '<table class="data-table"><thead><tr><th>Modello</th><th>Downloads</th><th></th></tr></thead><tbody>';
    data.results.forEach(m => {
      html += `<tr><td>${m.id || m.modelId || m.name}</td><td>${m.downloads || ''}</td>
        <td><button class="btn btn-sm" onclick="wizSelectPreset('${m.id || m.modelId || m.name}')">Seleziona</button></td></tr>`;
    });
    html += '</tbody></table>';
    container.innerHTML = html;
  } catch(e) {
    container.innerHTML = '<div style="color:var(--red)">Errore: ' + e.message + '</div>';
  }
}

async function wizDownloadModel() {
  const model = document.getElementById('wiz-selected-model').value.trim();
  if (!model) { showToast('Seleziona un modello prima', 'warn'); return; }
  showToast('Download avviato per ' + model + '. Controlla i log.', 'info');
  try { await apiPost('/api/dgx/wizard/model-download', {model: model}); } catch(e) {}
}

async function wizSaveModel() {
  const model = document.getElementById('wiz-selected-model')?.value?.trim() || '';
  wizState.model_config = {model_id: model};
  // Also set it in engine_params
  if (model) {
    if (!wizState.engine_params) wizState.engine_params = {};
    wizState.engine_params.model = model;
    const modelEl = document.getElementById('wiz-p-model');
    if (modelEl) modelEl.value = model;
  }
  try { await apiPost('/api/dgx/wizard/model-save', {model_id: model}); } catch(e) {}
}

// --- Step 4: Ray Cluster ---
async function wizDetectNetwork() {
  const container = document.getElementById('wiz-net-results');
  container.innerHTML = '<div style="color:var(--fg3)">Rilevamento...</div>';
  try {
    const data = await apiGet('/api/dgx/wizard/net-detect');
    let html = '';
    if (data.interfaces && data.interfaces.length > 0) {
      html += '<div style="font-size:12px;margin-bottom:8px">';
      data.interfaces.forEach(i => { html += '<div style="padding:2px 0;color:var(--fg2)">' + i + '</div>'; });
      html += '</div>';
    }
    if (data.ip_info && data.ip_info.length > 0) {
      html += '<div style="font-size:12px;color:var(--fg3);margin-bottom:8px">';
      data.ip_info.forEach(i => { html += '<div>' + i + '</div>'; });
      html += '</div>';
    }
    // Auto-fill suggested values
    if (data.suggested) {
      const s = data.suggested;
      if (s.NCCL_SOCKET_IFNAME) document.getElementById('wiz-nccl-socket').value = s.NCCL_SOCKET_IFNAME;
      if (s.UCX_NET_DEVICES) document.getElementById('wiz-ucx-dev').value = s.UCX_NET_DEVICES;
      if (s.GLOO_SOCKET_IFNAME) document.getElementById('wiz-gloo-socket').value = s.GLOO_SOCKET_IFNAME;
      if (s.NCCL_IB_HCA) document.getElementById('wiz-nccl-hca').value = s.NCCL_IB_HCA;
      html += '<div style="color:var(--green);font-size:12px">Valori suggeriti applicati automaticamente.</div>';
    }
    container.innerHTML = html || '<div style="color:var(--fg3)">Nessuna interfaccia rilevata</div>';
  } catch(e) {
    container.innerHTML = '<div style="color:var(--red)">Errore: ' + e.message + '</div>';
  }
}

async function wizSaveRayConfig() {
  const data = {
    NCCL_SOCKET_IFNAME: document.getElementById('wiz-nccl-socket')?.value || '',
    UCX_NET_DEVICES: document.getElementById('wiz-ucx-dev')?.value || '',
    GLOO_SOCKET_IFNAME: document.getElementById('wiz-gloo-socket')?.value || '',
    NCCL_IB_HCA: document.getElementById('wiz-nccl-hca')?.value || '',
    VLLM_HOST_IP: document.getElementById('wiz-host-ip')?.value || '',
    MASTER_ADDR: document.getElementById('wiz-master-addr')?.value || '',
  };
  wizState.ray_config = data;
  try { await apiPost('/api/dgx/wizard/ray-config', data); } catch(e) {}
}

async function wizLoadWorkersList() {
  const container = document.getElementById('wiz-workers-list');
  try {
    const data = await apiGet('/api/dgx/workers');
    if (!data.workers || Object.keys(data.workers).length === 0) {
      container.innerHTML = '<div style="color:var(--fg3);font-size:12px">Nessun worker configurato. Aggiungi worker dal tab DGX Spark.</div>';
      return;
    }
    let html = '<table class="data-table"><thead><tr><th>IP</th><th>Alias</th><th>Stato</th></tr></thead><tbody>';
    Object.entries(data.workers).forEach(([ip, w]) => {
      const st = w.status || 'unknown';
      const color = st === 'ready' || st === 'ray_connected' ? 'var(--green)' : st === 'error' ? 'var(--red)' : 'var(--fg3)';
      html += `<tr><td>${ip}</td><td>${w.alias || ''}</td><td style="color:${color}">${st}</td></tr>`;
    });
    html += '</tbody></table>';
    container.innerHTML = html;
  } catch(e) {
    container.innerHTML = '<div style="color:var(--red)">Errore caricamento worker</div>';
  }
}

// --- Step 5: Verify & Launch ---
async function wizVerifyAll() {
  const container = document.getElementById('wiz-verify-results');
  container.innerHTML = '<div style="color:var(--fg3)">Verifica in corso...</div>';
  try {
    const data = await apiPost('/api/dgx/wizard/verify-all', {});
    let html = '<h4 style="font-size:13px;margin-bottom:8px">Nodo Locale</h4>';
    Object.entries(data.local || {}).forEach(([key, v]) => {
      html += hwCheckItem(v.ok, key.toUpperCase(), v.info || (v.ok ? 'OK' : 'Non disponibile'));
    });
    if (data.workers && Object.keys(data.workers).length > 0) {
      html += '<h4 style="font-size:13px;margin:12px 0 8px">Worker Remoti</h4>';
      Object.entries(data.workers).forEach(([ip, wr]) => {
        html += `<div style="font-weight:600;font-size:12px;padding:4px 12px;color:var(--fg2)">${ip}</div>`;
        Object.entries(wr.results || {}).forEach(([key, v]) => {
          html += hwCheckItem(v.ok, key.toUpperCase(), v.info || '');
        });
      });
    }
    if (data.all_passed) {
      html += '<div style="color:var(--green);font-weight:600;margin-top:12px;padding:8px 12px">Tutte le verifiche superate!</div>';
      document.getElementById('wiz-launch-btn').disabled = false;
    } else {
      html += '<div style="color:var(--orange);margin-top:12px;padding:8px 12px">Alcune verifiche non superate. Il lancio e\' comunque possibile.</div>';
      document.getElementById('wiz-launch-btn').disabled = false;
    }
    container.innerHTML = html;
  } catch(e) {
    container.innerHTML = '<div style="color:var(--red)">Errore verifica: ' + e.message + '</div>';
  }
}

function wizRenderSummary() {
  const container = document.getElementById('wiz-summary');
  const p = wizState.engine_params || {};
  const m = wizState.model_config || {};
  const r = wizState.ray_config || {};
  let html = '<table class="wizard-summary">';
  const addRow = (label, val) => { if (val) html += `<tr><td>${label}</td><td>${val}</td></tr>`; };
  addRow('Modello', p.model || m.model_id || '<em>Non configurato</em>');
  addRow('Dtype', p.dtype);
  addRow('Max Context', p.max_model_len || 'auto');
  addRow('GPU Memory Util', p.gpu_memory_utilization);
  addRow('Tensor Parallel', p.tensor_parallel_size);
  addRow('Pipeline Parallel', p.pipeline_parallel_size > 1 ? p.pipeline_parallel_size : '');
  addRow('Nodi', p.nnodes > 1 ? p.nnodes : '1');
  addRow('Host:Port', (p.host || '0.0.0.0') + ':' + (p.port || 8000));
  addRow('Attention Backend', p.attention_backend || 'auto');
  addRow('Quantizzazione', p.quantization || 'nessuna');
  addRow('Prefix Caching', p.enable_prefix_caching ? 'Attivo' : 'Disattivo');
  addRow('KV Cache Dtype', p.kv_cache_dtype || 'auto');
  addRow('Executor Backend', p.distributed_executor_backend || 'auto');
  if (r.NCCL_SOCKET_IFNAME) addRow('NCCL Interface', r.NCCL_SOCKET_IFNAME);
  if (r.VLLM_HOST_IP) addRow('VLLM Host IP', r.VLLM_HOST_IP);
  html += '</table>';
  container.innerHTML = html;
}

async function wizLaunch() {
  // Ensure all step data is saved
  await wizSaveParams();
  await wizSaveModel();
  await wizSaveRayConfig();
  await wizSaveState();

  showToast('Avvio server vLLM...', 'info');
  try {
    const data = await apiPost('/api/dgx/wizard/launch', {});
    if (data.error) {
      showToast('Errore: ' + data.error, 'error');
    } else {
      showToast('Server vLLM avviato! Controlla stato nel tab Server.', 'success');
      switchTab('server');
    }
  } catch(e) {
    showToast('Errore avvio: ' + e.message, 'error');
  }
}

// Init wizard on page load
if (document.getElementById('tab-dgx-wizard')) {
  setTimeout(wizInit, 500);
}
</script>
</body>
</html>
"""


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="vLLM Manager Web Interface")
    parser.add_argument("--host", default="0.0.0.0", help="Bind host (default: 0.0.0.0)")
    parser.add_argument("--port", type=int, default=5000, help="Web UI port (default: 5000)")
    parser.add_argument("--debug", action="store_true", help="Enable Flask debug mode")
    args = parser.parse_args()

    # Start background services
    gpu_mon.start()
    threading.Thread(target=detector.detect, daemon=True).start()
    threading.Thread(target=drain_log_queue, daemon=True).start()

    log_queue.put("[INFO] vLLM Manager Web v%s avviato\n" % APP_VERSION)
    log_queue.put("[INFO] Piattaforma: %s (%s)\n" % (platform_info.display_name, platform_info.arch))
    log_queue.put("[INFO] Interfaccia web su http://%s:%d\n" % (args.host, args.port))

    app.run(host=args.host, port=args.port, debug=args.debug, threaded=True)


if __name__ == "__main__":
    main()
