#!/usr/bin/env python3
"""vLLM Manager — Desktop app to manage vLLM on WSL2."""

import json
import os
import queue
import re
import subprocess
import sys
import threading
import time
import tkinter as tk
from tkinter import ttk, messagebox, filedialog, scrolledtext
from datetime import datetime
from pathlib import Path
from urllib import request, error, parse
from collections import deque

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

APP_NAME = "vLLM Manager"
APP_VERSION = "1.0.0"
WINDOW_SIZE = "1100x750"
CONFIG_DIR = Path(os.environ.get("APPDATA", Path.home())) / "vLLM Manager"
CONFIG_FILE = CONFIG_DIR / "config.json"
WSL_DISTRO = "Ubuntu-22.04"
VLLM_VENV = "/root/vllm-env"

PRESET_MODELS = [
    {"name": "Qwen/Qwen2.5-1.5B-Instruct", "vram": "~3 GB", "note": "Test veloce", "chat_template": ""},
    {"name": "Qwen/Qwen2.5-7B-Instruct", "vram": "~14 GB", "note": "Buon bilanciamento", "chat_template": ""},
    {"name": "meta-llama/Llama-3.1-8B-Instruct", "vram": "~16 GB", "note": "Al limite 16 GB", "chat_template": ""},
    {"name": "TheBloke/Mistral-7B-Instruct-v0.2-AWQ", "vram": "~4 GB", "note": "Quantizzato AWQ", "chat_template": "mistral"},
    {"name": "TheBloke/Llama-2-13B-GPTQ", "vram": "~8 GB", "note": "13B quantizzato GPTQ", "chat_template": "llama-2"},
]

# Built-in chat templates for models that don't include one
CHAT_TEMPLATES = {
    "": "",  # auto (model defines its own)
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

NO_WINDOW = 0x08000000 if sys.platform == "win32" else 0

# ---------------------------------------------------------------------------
# Backend: WSLBridge
# ---------------------------------------------------------------------------

class WSLBridge:
    """Execute commands inside WSL2 via subprocess."""

    def __init__(self, distro: str = WSL_DISTRO):
        self.distro = distro

    def run(self, cmd: str, timeout: int = 30) -> tuple:
        """Run a command synchronously. Returns (returncode, stdout, stderr)."""
        try:
            proc = subprocess.run(
                ["wsl", "-d", self.distro, "--", "bash", "-lc", cmd],
                capture_output=True, text=True, timeout=timeout,
                creationflags=NO_WINDOW,
            )
            return proc.returncode, proc.stdout, proc.stderr
        except subprocess.TimeoutExpired:
            return -1, "", "Command timed out"
        except Exception as e:
            return -1, "", str(e)

    def popen(self, cmd: str) -> subprocess.Popen:
        """Start a long-running command, return Popen handle."""
        return subprocess.Popen(
            ["wsl", "-d", self.distro, "--", "bash", "-lc", cmd],
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, bufsize=1,
            creationflags=NO_WINDOW,
        )

    def run_raw(self, cmd: str, timeout: int = 30) -> tuple:
        """Run a command with bash -c (no login shell). For kill/pkill etc."""
        try:
            proc = subprocess.run(
                ["wsl", "-d", self.distro, "--", "bash", "-c", cmd],
                capture_output=True, text=True, timeout=timeout,
                creationflags=NO_WINDOW,
            )
            return proc.returncode, proc.stdout, proc.stderr
        except subprocess.TimeoutExpired:
            return -1, "", "Command timed out"
        except Exception as e:
            return -1, "", str(e)

    def shutdown(self):
        """Terminate the entire WSL distro."""
        try:
            subprocess.run(
                ["wsl", "--terminate", self.distro],
                timeout=15, creationflags=NO_WINDOW,
            )
        except Exception:
            pass

    def is_available(self) -> bool:
        rc, out, _ = self.run("echo ok", timeout=10)
        return rc == 0 and "ok" in out


# ---------------------------------------------------------------------------
# Backend: VLLMProcess
# ---------------------------------------------------------------------------

class VLLMProcess:
    """Manage the lifecycle of a vLLM server process."""

    def __init__(self, wsl: WSLBridge, log_queue: queue.Queue):
        self.wsl = wsl
        self.log_queue = log_queue
        self.proc = None
        self._reader_thread = None
        self.running = False

    def start(self, model: str, gpu_mem_util: float = 0.90,
              max_model_len=None, dtype: str = "auto",
              enable_prefix_caching: bool = True, extra_args: str = "",
              host: str = "0.0.0.0", port: int = 8000,
              chat_template: str = ""):
        if self.running:
            return

        # Write chat template file inside WSL if needed
        tpl_content = CHAT_TEMPLATES.get(chat_template, "")
        if tpl_content:
            self.wsl.run(
                "mkdir -p /tmp/vllm-manager && cat > /tmp/vllm-manager/chat_template.jinja << 'ENDTPL'\n"
                "%s\nENDTPL" % tpl_content,
                timeout=5,
            )

        parts = [
            "source %s/bin/activate &&" % VLLM_VENV,
            "python -m vllm.entrypoints.openai.api_server",
            "--model '%s'" % model,
            "--gpu-memory-utilization %s" % gpu_mem_util,
            "--dtype %s" % dtype,
            "--host %s" % host,
            "--port %s" % port,
        ]
        if tpl_content:
            parts.append("--chat-template /tmp/vllm-manager/chat_template.jinja")
        if max_model_len:
            parts.append("--max-model-len %s" % max_model_len)
        if enable_prefix_caching:
            parts.append("--enable-prefix-caching")
        if extra_args.strip():
            parts.append(extra_args.strip())

        cmd = " ".join(parts)
        self.log_queue.put("[CMD] %s\n" % cmd)
        self.proc = self.wsl.popen(cmd)
        self.running = True
        self._reader_thread = threading.Thread(target=self._read_output, daemon=True)
        self._reader_thread.start()

    def stop(self):
        if not self.running or self.proc is None:
            return
        self.log_queue.put("[INFO] Stopping vLLM server...\n")
        # Kill the vLLM process inside WSL using bash -c (not login shell)
        self.wsl.run_raw(
            "pkill -9 -f 'vllm.entrypoints.openai.api_server'; "
            "pkill -9 -f 'from multiprocessing.resource_tracker'; "
            "sleep 0.5",
            timeout=10,
        )
        # Terminate the wsl.exe wrapper on Windows side
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
    """HTTP client for the vLLM OpenAI-compatible API."""

    def __init__(self, base_url: str = "http://localhost:8000"):
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

    def chat(self, model, messages, max_tokens=256,
             temperature=0.7, stream=False):
        body = json.dumps({
            "model": model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
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
        """Read SSE stream, measure TTFT and collect tokens."""
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
        full_text = "".join(tokens)
        return {
            "text": full_text,
            "tokens_count": len(tokens),
            "elapsed": elapsed,
            "ttft": ttft,
        }


# ---------------------------------------------------------------------------
# Backend: ConfigManager
# ---------------------------------------------------------------------------

class ConfigManager:
    """Save / load profiles and settings to %APPDATA%/vLLM Manager/config.json."""

    def __init__(self):
        CONFIG_DIR.mkdir(parents=True, exist_ok=True)
        self.data = self._load()

    def _load(self):
        if CONFIG_FILE.exists():
            try:
                return json.loads(CONFIG_FILE.read_text("utf-8"))
            except Exception:
                pass
        return {"profiles": {}, "settings": {}}

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

    def get_settings(self):
        return self.data.setdefault("settings", {})

    def save_settings(self, settings):
        self.data["settings"] = settings
        self.save()


# ---------------------------------------------------------------------------
# Backend: GpuMonitor
# ---------------------------------------------------------------------------

class GpuMonitor:
    """Thread that polls nvidia-smi every N seconds for GPU metrics."""

    def __init__(self, wsl, interval=2.0):
        self.wsl = wsl
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

    def _poll(self):
        while not self._stop.is_set():
            try:
                rc, out, _ = self.wsl.run(
                    "nvidia-smi --query-gpu=memory.used,memory.total,"
                    "utilization.gpu,temperature.gpu,power.draw,name "
                    "--format=csv,noheader,nounits",
                    timeout=5,
                )
                if rc == 0 and out.strip():
                    parts = [p.strip() for p in out.strip().split(",")]
                    if len(parts) >= 6:
                        self.latest = {
                            "vram_used": int(parts[0]),
                            "vram_total": int(parts[1]),
                            "gpu_util": int(parts[2]),
                            "temp": int(parts[3]),
                            "power": float(parts[4]) if parts[4].replace('.','',1).isdigit() else 0.0,
                            "name": parts[5],
                        }
                        self.history_vram.append(self.latest["vram_used"])
                        self.history_util.append(self.latest["gpu_util"])
                        self.history_temp.append(self.latest["temp"])
            except Exception:
                pass
            self._stop.wait(self.interval)


# ---------------------------------------------------------------------------
# UI: Application
# ---------------------------------------------------------------------------

class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("%s v%s" % (APP_NAME, APP_VERSION))
        self.geometry(WINDOW_SIZE)
        self.minsize(900, 600)
        self.protocol("WM_DELETE_WINDOW", self._on_close)

        # Backend objects
        self.wsl = WSLBridge()
        self.log_queue = queue.Queue()
        self.vllm = VLLMProcess(self.wsl, self.log_queue)
        self.api = VLLMApi()
        self.config = ConfigManager()
        self.gpu_mon = GpuMonitor(self.wsl)

        # State
        self.server_online = False

        # Style
        style = ttk.Style(self)
        style.theme_use("clam")

        # Build UI
        self._build_status_bar()
        self.notebook = ttk.Notebook(self)
        self.notebook.pack(fill="both", expand=True, padx=6, pady=(6, 0))

        self._build_server_tab()
        self._build_gpu_tab()
        self._build_logs_tab()
        self._build_models_tab()
        self._build_benchmark_tab()
        self._build_webui_tab()
        self._build_profiles_tab()

        # Start background services
        self.gpu_mon.start()
        self._poll_logs()
        self._poll_server_status()
        self._poll_gpu_ui()

    # ---- status bar -------------------------------------------------------

    def _build_status_bar(self):
        bar = ttk.Frame(self)
        bar.pack(fill="x", side="bottom", padx=6, pady=4)
        self.status_indicator = tk.Canvas(bar, width=14, height=14,
                                          highlightthickness=0)
        self.status_indicator.pack(side="left")
        self._draw_indicator(False)
        self.status_label = ttk.Label(bar, text="Server: offline")
        self.status_label.pack(side="left", padx=(4, 12))
        self.gpu_status_label = ttk.Label(bar, text="GPU: —")
        self.gpu_status_label.pack(side="left")
        ttk.Label(bar, text="  %s v%s" % (APP_NAME, APP_VERSION)).pack(side="right")

    def _draw_indicator(self, online):
        c = self.status_indicator
        c.delete("all")
        color = "#22c55e" if online else "#ef4444"
        c.create_oval(2, 2, 12, 12, fill=color, outline=color)

    # ---- Tab 1: Server ----------------------------------------------------

    def _build_server_tab(self):
        frame = ttk.Frame(self.notebook, padding=10)
        self.notebook.add(frame, text=" Server ")

        # Model selection
        mf = ttk.LabelFrame(frame, text="Modello", padding=8)
        mf.pack(fill="x", pady=(0, 8))

        ttk.Label(mf, text="Modello:").grid(row=0, column=0, sticky="w")
        self.model_var = tk.StringVar()
        self.model_combo = ttk.Combobox(mf, textvariable=self.model_var, width=55)
        self.model_combo["values"] = [m["name"] for m in PRESET_MODELS]
        if PRESET_MODELS:
            self.model_combo.current(0)
        self.model_combo.grid(row=0, column=1, padx=8, sticky="ew")

        info_frame = ttk.Frame(mf)
        info_frame.grid(row=1, column=0, columnspan=3, sticky="ew", pady=(6, 0))
        self.model_info_label = ttk.Label(info_frame, text="", foreground="#666")
        self.model_info_label.pack(side="left")
        self.model_combo.bind("<<ComboboxSelected>>", self._on_model_selected)

        mf.columnconfigure(1, weight=1)

        # Parameters
        pf = ttk.LabelFrame(frame, text="Parametri", padding=8)
        pf.pack(fill="x", pady=(0, 8))

        # GPU memory utilization
        ttk.Label(pf, text="GPU Memory Utilization:").grid(row=0, column=0, sticky="w")
        self.gpu_mem_var = tk.DoubleVar(value=0.90)
        self.gpu_mem_slider = ttk.Scale(pf, from_=0.5, to=0.99,
                                         variable=self.gpu_mem_var, orient="horizontal")
        self.gpu_mem_slider.grid(row=0, column=1, sticky="ew", padx=8)
        self.gpu_mem_label = ttk.Label(pf, text="0.90")
        self.gpu_mem_label.grid(row=0, column=2, padx=(0, 8))
        self.gpu_mem_slider.configure(command=lambda v: self.gpu_mem_label.configure(
            text="%.2f" % float(v)))

        # Max model len
        ttk.Label(pf, text="Max Model Len (0=auto):").grid(row=1, column=0, sticky="w", pady=4)
        self.max_len_var = tk.IntVar(value=0)
        ttk.Entry(pf, textvariable=self.max_len_var, width=10).grid(
            row=1, column=1, sticky="w", padx=8)

        # Dtype
        ttk.Label(pf, text="Dtype:").grid(row=2, column=0, sticky="w")
        self.dtype_var = tk.StringVar(value="auto")
        dtype_combo = ttk.Combobox(pf, textvariable=self.dtype_var, width=12,
                                    values=["auto", "float16", "bfloat16", "float32"],
                                    state="readonly")
        dtype_combo.grid(row=2, column=1, sticky="w", padx=8)

        # Prefix caching
        self.prefix_cache_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(pf, text="Enable Prefix Caching",
                         variable=self.prefix_cache_var).grid(
            row=3, column=0, columnspan=2, sticky="w", pady=4)

        # Host / Port
        ttk.Label(pf, text="Host:").grid(row=4, column=0, sticky="w")
        self.host_var = tk.StringVar(value="0.0.0.0")
        ttk.Entry(pf, textvariable=self.host_var, width=16).grid(
            row=4, column=1, sticky="w", padx=8)

        ttk.Label(pf, text="Port:").grid(row=5, column=0, sticky="w", pady=4)
        self.port_var = tk.IntVar(value=8000)
        ttk.Entry(pf, textvariable=self.port_var, width=8).grid(
            row=5, column=1, sticky="w", padx=8)

        # Chat template
        ttk.Label(pf, text="Chat Template:").grid(row=6, column=0, sticky="w")
        self.chat_template_var = tk.StringVar(value="")
        chat_tpl_combo = ttk.Combobox(pf, textvariable=self.chat_template_var, width=16,
                                       values=["(auto)"] + [k for k in CHAT_TEMPLATES if k])
        chat_tpl_combo.current(0)
        chat_tpl_combo.grid(row=6, column=1, sticky="w", padx=8)
        ttk.Label(pf, text="(necessario per modelli TheBloke/quantizzati)",
                   foreground="#888").grid(row=6, column=2, sticky="w")

        # Extra args
        ttk.Label(pf, text="Extra Args:").grid(row=7, column=0, sticky="w")
        self.extra_args_var = tk.StringVar()
        ttk.Entry(pf, textvariable=self.extra_args_var, width=50).grid(
            row=7, column=1, columnspan=2, sticky="ew", padx=8, pady=4)

        pf.columnconfigure(1, weight=1)

        # Buttons
        bf = ttk.Frame(frame)
        bf.pack(fill="x", pady=8)
        self.start_btn = ttk.Button(bf, text="Start Server", command=self._start_server)
        self.start_btn.pack(side="left", padx=(0, 8))
        self.stop_btn = ttk.Button(bf, text="Stop Server", command=self._stop_server,
                                    state="disabled")
        self.stop_btn.pack(side="left", padx=(0, 8))
        self.stop_all_btn = ttk.Button(bf, text="Stoppa Tutto (+ WSL)",
                                        command=self._stop_all)
        self.stop_all_btn.pack(side="left", padx=(0, 8))
        ttk.Button(bf, text="Salva Profilo", command=self._quick_save_profile).pack(
            side="left", padx=(0, 8))

        # Server log preview
        self.server_log_preview = tk.Text(frame, height=8, state="disabled",
                                           bg="#1e1e1e", fg="#d4d4d4",
                                           font=("Consolas", 9), wrap="word")
        frame.rowconfigure(4, weight=1)
        self.server_log_preview.pack(fill="both", expand=True)

        # Initial model info (must be after chat_template_var is created)
        self._on_model_selected(None)

    def _on_model_selected(self, _event):
        name = self.model_var.get()
        for m in PRESET_MODELS:
            if m["name"] == name:
                self.model_info_label.configure(
                    text="VRAM: %s  —  %s" % (m["vram"], m["note"]))
                tpl = m.get("chat_template", "")
                self.chat_template_var.set(tpl if tpl else "(auto)")
                return
        self.model_info_label.configure(text="Custom model")

    def _start_server(self):
        model = self.model_var.get().strip()
        if not model:
            messagebox.showwarning("Modello mancante", "Inserisci un nome modello.")
            return
        max_len = self.max_len_var.get()
        self.start_btn.configure(state="disabled")
        self.stop_btn.configure(state="normal")

        chat_tpl = self.chat_template_var.get()
        if chat_tpl == "(auto)":
            chat_tpl = ""

        def _run():
            # Ensure WSL is running (may have been shut down by "Stoppa Tutto")
            self.log_queue.put("[INFO] Checking WSL...\n")
            if not self.wsl.is_available():
                self.log_queue.put("[INFO] WSL not running, starting it...\n")
                # Any wsl command will wake up the distro
                self.wsl.run("echo WSL ready", timeout=30)
                if not self.wsl.is_available():
                    self.log_queue.put("[ERROR] Could not start WSL!\n")
                    self.after(0, lambda: (
                        self.start_btn.configure(state="normal"),
                        self.stop_btn.configure(state="disabled"),
                        messagebox.showerror("Errore WSL",
                                             "Impossibile avviare WSL. Verifica l'installazione."),
                    ))
                    return
                self.log_queue.put("[INFO] WSL started.\n")
            self.vllm.start(
                model=model,
                gpu_mem_util=self.gpu_mem_var.get(),
                max_model_len=max_len if max_len > 0 else None,
                dtype=self.dtype_var.get(),
                enable_prefix_caching=self.prefix_cache_var.get(),
                extra_args=self.extra_args_var.get(),
                host=self.host_var.get(),
                port=self.port_var.get(),
                chat_template=chat_tpl,
            )
        threading.Thread(target=_run, daemon=True).start()

    def _stop_server(self):
        self.stop_btn.configure(state="disabled")

        def _run():
            self.vllm.stop()
            self.after(0, lambda: self.start_btn.configure(state="normal"))
        threading.Thread(target=_run, daemon=True).start()

    def _stop_all(self):
        """Stop vLLM server and Docker containers (keeps WSL running)."""
        self.stop_btn.configure(state="disabled")
        self.stop_all_btn.configure(state="disabled")

        def _run():
            # 1. Kill vLLM inside WSL
            if self.vllm.running:
                self.vllm.stop()
            else:
                self.wsl.run_raw(
                    "pkill -9 -f 'vllm.entrypoints.openai.api_server'; "
                    "pkill -9 -f 'from multiprocessing.resource_tracker'",
                    timeout=5,
                )
            self.log_queue.put("[INFO] vLLM killed.\n")
            # 2. Stop Docker containers
            container = self.webui_container_var.get()
            self._docker_run(["stop", container], timeout=15)
            self.log_queue.put("[INFO] Docker container '%s' stopped.\n" % container)
            self.after(0, lambda: (
                self.start_btn.configure(state="normal"),
                self.stop_all_btn.configure(state="normal"),
                self._check_webui_status(),
            ))
        threading.Thread(target=_run, daemon=True).start()

    def _quick_save_profile(self):
        from tkinter import simpledialog
        name = simpledialog.askstring("Salva Profilo", "Nome profilo:",
                                       parent=self)
        if not name:
            return
        profile = self._collect_profile()
        self.config.save_profile(name, profile)
        self._refresh_profiles_list()
        messagebox.showinfo("Profilo Salvato", "Profilo '%s' salvato." % name)

    def _collect_profile(self):
        return {
            "model": self.model_var.get(),
            "gpu_mem_util": self.gpu_mem_var.get(),
            "max_model_len": self.max_len_var.get(),
            "dtype": self.dtype_var.get(),
            "prefix_caching": self.prefix_cache_var.get(),
            "host": self.host_var.get(),
            "port": self.port_var.get(),
            "extra_args": self.extra_args_var.get(),
            "chat_template": self.chat_template_var.get(),
        }

    def _apply_profile(self, profile):
        self.model_var.set(profile.get("model", ""))
        self.gpu_mem_var.set(profile.get("gpu_mem_util", 0.90))
        self.gpu_mem_label.configure(text="%.2f" % profile.get("gpu_mem_util", 0.90))
        self.max_len_var.set(profile.get("max_model_len", 0))
        self.dtype_var.set(profile.get("dtype", "auto"))
        self.prefix_cache_var.set(profile.get("prefix_caching", True))
        self.host_var.set(profile.get("host", "0.0.0.0"))
        self.port_var.set(profile.get("port", 8000))
        self.extra_args_var.set(profile.get("extra_args", ""))
        self.chat_template_var.set(profile.get("chat_template", "(auto)"))
        self._on_model_selected(None)

    # ---- Tab 2: GPU Monitor -----------------------------------------------

    def _build_gpu_tab(self):
        frame = ttk.Frame(self.notebook, padding=10)
        self.notebook.add(frame, text=" GPU Monitor ")

        # GPU name
        self.gpu_name_label = ttk.Label(frame, text="GPU: rilevamento...",
                                         font=("Segoe UI", 12, "bold"))
        self.gpu_name_label.pack(anchor="w", pady=(0, 10))

        # Metrics grid
        gf = ttk.Frame(frame)
        gf.pack(fill="x", pady=(0, 10))

        # VRAM
        ttk.Label(gf, text="VRAM:").grid(row=0, column=0, sticky="w", pady=4)
        self.vram_bar = ttk.Progressbar(gf, length=400, mode="determinate")
        self.vram_bar.grid(row=0, column=1, padx=8, sticky="ew")
        self.vram_label = ttk.Label(gf, text="— / — MB")
        self.vram_label.grid(row=0, column=2, padx=4)

        # GPU utilization
        ttk.Label(gf, text="GPU Util:").grid(row=1, column=0, sticky="w", pady=4)
        self.util_bar = ttk.Progressbar(gf, length=400, mode="determinate")
        self.util_bar.grid(row=1, column=1, padx=8, sticky="ew")
        self.util_label = ttk.Label(gf, text="— %")
        self.util_label.grid(row=1, column=2, padx=4)

        # Temperature
        ttk.Label(gf, text="Temp:").grid(row=2, column=0, sticky="w", pady=4)
        self.temp_label = ttk.Label(gf, text="— C", font=("Segoe UI", 11, "bold"))
        self.temp_label.grid(row=2, column=1, sticky="w", padx=8)

        # Power
        ttk.Label(gf, text="Power:").grid(row=3, column=0, sticky="w", pady=4)
        self.power_label = ttk.Label(gf, text="— W")
        self.power_label.grid(row=3, column=1, sticky="w", padx=8)

        gf.columnconfigure(1, weight=1)

        # Canvas for history graph
        graph_frame = ttk.LabelFrame(frame, text="Storico (ultimi 2 minuti)", padding=6)
        graph_frame.pack(fill="both", expand=True)
        self.gpu_canvas = tk.Canvas(graph_frame, bg="#1e1e1e", height=200,
                                     highlightthickness=0)
        self.gpu_canvas.pack(fill="both", expand=True)

    def _update_gpu_ui(self):
        d = self.gpu_mon.latest
        if not d:
            return
        self.gpu_name_label.configure(text="GPU: %s" % d.get("name", "?"))
        used = d["vram_used"]
        total = d["vram_total"]
        pct = (used / total * 100) if total else 0
        self.vram_bar["value"] = pct
        self.vram_label.configure(text="%d / %d MB  (%.0f%%)" % (used, total, pct))

        util = d["gpu_util"]
        self.util_bar["value"] = util
        self.util_label.configure(text="%d%%" % util)

        temp = d["temp"]
        color = "#22c55e" if temp < 65 else ("#f59e0b" if temp < 80 else "#ef4444")
        self.temp_label.configure(text="%d C" % temp, foreground=color)
        self.power_label.configure(text="%.1f W" % d["power"])

        # Update status bar GPU info
        self.gpu_status_label.configure(
            text="GPU: %d/%d MB  |  %d%%  |  %d C" % (used, total, util, temp))

        # Draw graph
        self._draw_gpu_graph()

    def _draw_gpu_graph(self):
        c = self.gpu_canvas
        c.delete("all")
        w = c.winfo_width()
        h = c.winfo_height()
        if w < 20 or h < 20:
            return

        # Grid lines
        for i in range(1, 4):
            y = h * i / 4
            c.create_line(0, y, w, y, fill="#333", dash=(2, 4))
            c.create_text(w - 4, y - 8, text="%d%%" % (100 - 25 * i),
                          fill="#555", anchor="e", font=("Consolas", 7))

        def _draw_line(data, color):
            if len(data) < 2:
                return
            max_val = max(max(data), 1)
            points = []
            n = len(data)
            for i, v in enumerate(data):
                x = w * i / (n - 1) if n > 1 else 0
                y = h - (v / max_val) * (h - 10) - 5
                points.append(x)
                points.append(y)
            if len(points) >= 4:
                c.create_line(points, fill=color, width=2, smooth=True)

        # VRAM as percentage
        total = self.gpu_mon.latest.get("vram_total", 1)
        vram_pct = [v / total * 100 for v in self.gpu_mon.history_vram]
        _draw_line(vram_pct, "#3b82f6")
        _draw_line(list(self.gpu_mon.history_util), "#22c55e")
        _draw_line(list(self.gpu_mon.history_temp), "#ef4444")

        # Legend
        y0 = 12
        for label, color in [("VRAM %", "#3b82f6"), ("Util %", "#22c55e"),
                              ("Temp", "#ef4444")]:
            c.create_rectangle(8, y0 - 4, 20, y0 + 4, fill=color, outline=color)
            c.create_text(24, y0, text=label, fill="#aaa", anchor="w",
                          font=("Consolas", 8))
            y0 += 16

    # ---- Tab 3: Logs ------------------------------------------------------

    def _build_logs_tab(self):
        frame = ttk.Frame(self.notebook, padding=10)
        self.notebook.add(frame, text=" Logs ")

        # Search bar
        sf = ttk.Frame(frame)
        sf.pack(fill="x", pady=(0, 6))
        ttk.Label(sf, text="Cerca:").pack(side="left")
        self.log_search_var = tk.StringVar()
        search_entry = ttk.Entry(sf, textvariable=self.log_search_var, width=30)
        search_entry.pack(side="left", padx=6)
        ttk.Button(sf, text="Trova", command=self._search_logs).pack(side="left")
        ttk.Button(sf, text="Pulisci Log", command=self._clear_logs).pack(side="right")

        self.log_text = tk.Text(frame, state="disabled", bg="#1e1e1e", fg="#d4d4d4",
                                 font=("Consolas", 9), wrap="word")
        scrollbar = ttk.Scrollbar(frame, command=self.log_text.yview)
        self.log_text.configure(yscrollcommand=scrollbar.set)
        scrollbar.pack(side="right", fill="y")
        self.log_text.pack(fill="both", expand=True)

        # Configure tags for coloring
        self.log_text.tag_configure("error", foreground="#ef4444")
        self.log_text.tag_configure("warning", foreground="#f59e0b")
        self.log_text.tag_configure("info", foreground="#3b82f6")
        self.log_text.tag_configure("highlight", background="#854d0e", foreground="#fef08a")

        # Bind Ctrl+F
        self.bind_all("<Control-f>", lambda e: (
            self.notebook.select(2),
            search_entry.focus_set(),
        ))

    def _search_logs(self):
        self.log_text.tag_remove("highlight", "1.0", "end")
        term = self.log_search_var.get()
        if not term:
            return
        start = "1.0"
        while True:
            pos = self.log_text.search(term, start, "end", nocase=True)
            if not pos:
                break
            end_pos = "%s+%dc" % (pos, len(term))
            self.log_text.tag_add("highlight", pos, end_pos)
            start = end_pos
        # Scroll to first match
        first = self.log_text.tag_ranges("highlight")
        if first:
            self.log_text.see(first[0])

    def _clear_logs(self):
        self.log_text.configure(state="normal")
        self.log_text.delete("1.0", "end")
        self.log_text.configure(state="disabled")

    # ---- Tab 4: Models ----------------------------------------------------

    def _build_models_tab(self):
        frame = ttk.Frame(self.notebook, padding=10)
        self.notebook.add(frame, text=" Models ")

        # Search bar
        sf = ttk.Frame(frame)
        sf.pack(fill="x", pady=(0, 8))
        ttk.Label(sf, text="Cerca su HuggingFace:").pack(side="left")
        self.hf_search_var = tk.StringVar()
        entry = ttk.Entry(sf, textvariable=self.hf_search_var, width=40)
        entry.pack(side="left", padx=8)
        entry.bind("<Return>", lambda e: self._search_hf())
        ttk.Button(sf, text="Cerca", command=self._search_hf).pack(side="left")
        self.hf_status_label = ttk.Label(sf, text="")
        self.hf_status_label.pack(side="left", padx=8)

        # Results treeview
        tree_frame = ttk.Frame(frame)
        tree_frame.pack(fill="both", expand=True)

        cols = ("model_id", "downloads", "likes")
        self.models_tree = ttk.Treeview(tree_frame, columns=cols, show="headings",
                                         height=15)
        self.models_tree.heading("model_id", text="Model ID")
        self.models_tree.heading("downloads", text="Downloads")
        self.models_tree.heading("likes", text="Likes")
        self.models_tree.column("model_id", width=500)
        self.models_tree.column("downloads", width=120, anchor="e")
        self.models_tree.column("likes", width=80, anchor="e")

        tree_scroll = ttk.Scrollbar(tree_frame, command=self.models_tree.yview)
        self.models_tree.configure(yscrollcommand=tree_scroll.set)
        self.models_tree.pack(fill="both", expand=True, side="left")
        tree_scroll.pack(fill="y", side="left")

        # Button panel
        bp = ttk.Frame(frame)
        bp.pack(fill="x", pady=(8, 0))
        ttk.Button(bp, text="Usa questo modello",
                    command=self._use_selected_model).pack(side="left")

        # --- Local models / cleanup section ---
        cleanup_frame = ttk.LabelFrame(frame, text="Modelli scaricati (HuggingFace cache)",
                                        padding=8)
        cleanup_frame.pack(fill="x", pady=(10, 0))

        local_cols = ("local_model", "size")
        self.local_models_tree = ttk.Treeview(cleanup_frame, columns=local_cols,
                                               show="headings", height=5)
        self.local_models_tree.heading("local_model", text="Modello")
        self.local_models_tree.heading("size", text="Dimensione")
        self.local_models_tree.column("local_model", width=500)
        self.local_models_tree.column("size", width=120, anchor="e")
        self.local_models_tree.pack(fill="x", side="top")

        cbf = ttk.Frame(cleanup_frame)
        cbf.pack(fill="x", pady=(6, 0))
        ttk.Button(cbf, text="Aggiorna Lista",
                    command=self._refresh_local_models).pack(side="left", padx=(0, 8))
        ttk.Button(cbf, text="Elimina Selezionato",
                    command=self._delete_local_model).pack(side="left", padx=(0, 8))
        self.local_models_status = ttk.Label(cbf, text="")
        self.local_models_status.pack(side="left", padx=8)

        self._refresh_local_models()

    def _search_hf(self):
        query = self.hf_search_var.get().strip()
        if not query:
            return
        self.hf_status_label.configure(text="Cercando...")
        for item in self.models_tree.get_children():
            self.models_tree.delete(item)

        def _fetch():
            try:
                url = ("https://huggingface.co/api/models?"
                       "search=%s&limit=20"
                       "&sort=downloads&direction=-1" % parse.quote(query))
                req_obj = request.Request(url, headers={"User-Agent": "vLLM-Manager/1.0"})
                with request.urlopen(req_obj, timeout=15) as resp:
                    models = json.loads(resp.read())
                self.after(0, lambda: self._populate_hf_results(models))
            except Exception as e:
                self.after(0, lambda: self.hf_status_label.configure(
                    text="Errore: %s" % e))
        threading.Thread(target=_fetch, daemon=True).start()

    def _populate_hf_results(self, models):
        for item in self.models_tree.get_children():
            self.models_tree.delete(item)
        for m in models:
            mid = m.get("modelId", m.get("id", "?"))
            dl = m.get("downloads", 0)
            likes = m.get("likes", 0)
            self.models_tree.insert("", "end", values=(mid, "{:,}".format(dl), likes))
        self.hf_status_label.configure(text="%d risultati" % len(models))

    def _use_selected_model(self):
        sel = self.models_tree.selection()
        if not sel:
            messagebox.showinfo("Nessuna selezione", "Seleziona un modello dalla lista.")
            return
        model_id = self.models_tree.item(sel[0])["values"][0]
        self.model_var.set(model_id)
        self._on_model_selected(None)
        self.notebook.select(0)  # Switch to Server tab

    def _refresh_local_models(self):
        for item in self.local_models_tree.get_children():
            self.local_models_tree.delete(item)
        self.local_models_status.configure(text="Scansione...")

        def _scan():
            # Use du -sh which outputs "SIZE\tPATH" per line — no escaping issues
            rc, out, _ = self.wsl.run_raw(
                "du -sh ~/.cache/huggingface/hub/models--*/ 2>/dev/null",
                timeout=30,
            )
            models = []
            if rc == 0 and out.strip():
                for line in out.strip().splitlines():
                    # Format: "6.8G\t/root/.cache/huggingface/hub/models--TheBloke--Llama-2-13B-GPTQ/"
                    parts = line.split("\t", 1)
                    if len(parts) == 2:
                        size = parts[0].strip()
                        path = parts[1].strip().rstrip("/")
                        dirname = path.rsplit("/", 1)[-1]  # models--TheBloke--Llama-2-13B-GPTQ
                        name = dirname.replace("models--", "", 1).replace("--", "/", 1)
                        models.append((name, size))
            self.after(0, lambda: self._show_local_models(models))
        threading.Thread(target=_scan, daemon=True).start()

    def _show_local_models(self, models):
        for item in self.local_models_tree.get_children():
            self.local_models_tree.delete(item)
        for name, size in models:
            self.local_models_tree.insert("", "end", values=(name, size))
        total = len(models)
        self.local_models_status.configure(
            text="%d modell%s in cache" % (total, "o" if total == 1 else "i"))

    def _delete_local_model(self):
        sel = self.local_models_tree.selection()
        if not sel:
            messagebox.showinfo("Nessuna selezione",
                                "Seleziona un modello dalla lista locale.")
            return
        model_name = self.local_models_tree.item(sel[0])["values"][0]
        model_size = self.local_models_tree.item(sel[0])["values"][1]
        if not messagebox.askyesno(
                "Conferma Eliminazione",
                "Eliminare '%s' (%s) dalla cache?\n\n"
                "Il modello verra' riscaricato al prossimo utilizzo."
                % (model_name, model_size)):
            return
        # Convert model name back to directory format
        dir_name = "models--" + model_name.replace("/", "--")
        self.local_models_status.configure(text="Eliminazione in corso...")

        def _delete():
            rc, out, err = self.wsl.run_raw(
                "rm -rf ~/.cache/huggingface/hub/%s" % dir_name,
                timeout=60,
            )
            if rc == 0:
                self.after(0, lambda: (
                    self.local_models_status.configure(text="Eliminato!"),
                    self._refresh_local_models(),
                ))
            else:
                msg = err.strip() or "errore sconosciuto"
                self.after(0, lambda: messagebox.showerror("Errore", msg))
        threading.Thread(target=_delete, daemon=True).start()

    # ---- Tab 5: Benchmark -------------------------------------------------

    def _build_benchmark_tab(self):
        frame = ttk.Frame(self.notebook, padding=10)
        self.notebook.add(frame, text=" Benchmark ")

        # Prompt selection
        pf = ttk.LabelFrame(frame, text="Prompt", padding=8)
        pf.pack(fill="x", pady=(0, 8))

        self.bench_prompts = {
            "Breve - Spiega cos'e' Python": "Spiega cos'e' Python in 2 frasi.",
            "Media - Scrivi una funzione": "Scrivi una funzione Python che calcola i numeri di Fibonacci fino a n.",
            "Lunga - Scrivi un articolo": "Scrivi un articolo dettagliato di 500 parole sui vantaggi del machine learning.",
            "Custom": "",
        }
        self.bench_prompt_var = tk.StringVar(value=list(self.bench_prompts.keys())[0])
        prompt_combo = ttk.Combobox(pf, textvariable=self.bench_prompt_var,
                                     values=list(self.bench_prompts.keys()),
                                     state="readonly", width=50)
        prompt_combo.pack(anchor="w", pady=(0, 6))
        prompt_combo.bind("<<ComboboxSelected>>", self._on_bench_prompt_changed)

        self.bench_prompt_text = tk.Text(pf, height=3, font=("Consolas", 9))
        self.bench_prompt_text.pack(fill="x")
        self.bench_prompt_text.insert("1.0", list(self.bench_prompts.values())[0])

        # Parameters
        param_f = ttk.Frame(frame)
        param_f.pack(fill="x", pady=(0, 8))
        ttk.Label(param_f, text="Max Tokens:").pack(side="left")
        self.bench_tokens_var = tk.IntVar(value=256)
        self.bench_tokens_label = ttk.Label(param_f, text="256")
        ttk.Scale(param_f, from_=32, to=2048, variable=self.bench_tokens_var,
                   orient="horizontal", length=200,
                   command=lambda v: self.bench_tokens_label.configure(
                       text=str(int(float(v))))).pack(side="left", padx=8)
        self.bench_tokens_label.pack(side="left")

        ttk.Label(param_f, text="   Temperatura:").pack(side="left", padx=(16, 0))
        self.bench_temp_var = tk.DoubleVar(value=0.7)
        ttk.Entry(param_f, textvariable=self.bench_temp_var, width=5).pack(
            side="left", padx=4)

        # Run button
        bf = ttk.Frame(frame)
        bf.pack(fill="x", pady=(0, 8))
        self.bench_run_btn = ttk.Button(bf, text="Esegui Benchmark",
                                         command=self._run_benchmark)
        self.bench_run_btn.pack(side="left")
        self.bench_status = ttk.Label(bf, text="")
        self.bench_status.pack(side="left", padx=8)

        # Results table
        rf = ttk.LabelFrame(frame, text="Risultati", padding=8)
        rf.pack(fill="both", expand=True)

        cols = ("metric", "value")
        self.bench_tree = ttk.Treeview(rf, columns=cols, show="headings", height=8)
        self.bench_tree.heading("metric", text="Metrica")
        self.bench_tree.heading("value", text="Valore")
        self.bench_tree.column("metric", width=250)
        self.bench_tree.column("value", width=300)
        self.bench_tree.pack(fill="both", expand=True)

        # Response preview
        self.bench_response_text = tk.Text(rf, height=6, state="disabled",
                                            bg="#1e1e1e", fg="#d4d4d4",
                                            font=("Consolas", 9), wrap="word")
        self.bench_response_text.pack(fill="x", pady=(6, 0))

    def _on_bench_prompt_changed(self, _event):
        key = self.bench_prompt_var.get()
        self.bench_prompt_text.delete("1.0", "end")
        if key in self.bench_prompts and self.bench_prompts[key]:
            self.bench_prompt_text.insert("1.0", self.bench_prompts[key])

    def _run_benchmark(self):
        if not self.server_online:
            messagebox.showwarning("Server Offline",
                                    "Avvia il server prima di eseguire il benchmark.")
            return
        prompt = self.bench_prompt_text.get("1.0", "end").strip()
        if not prompt:
            return
        self.bench_run_btn.configure(state="disabled")
        self.bench_status.configure(text="In esecuzione (streaming)...")

        def _run():
            models = self.api.list_models()
            model = models[0] if models else self.model_var.get()
            messages = [{"role": "user", "content": prompt}]
            max_tok = self.bench_tokens_var.get()
            temp = self.bench_temp_var.get()

            t0 = time.perf_counter()
            result = self.api.chat(model, messages, max_tokens=max_tok,
                                    temperature=temp, stream=True)
            total_time = time.perf_counter() - t0

            self.after(0, lambda: self._show_bench_results(result, total_time, model))

        threading.Thread(target=_run, daemon=True).start()

    def _show_bench_results(self, result, total_time, model):
        self.bench_run_btn.configure(state="normal")
        self.bench_status.configure(text="Completato")

        for item in self.bench_tree.get_children():
            self.bench_tree.delete(item)

        if "error" in result:
            self.bench_tree.insert("", "end",
                                    values=("Errore", result["error"]))
            return

        text = result.get("text", "")
        n_tokens = result.get("tokens_count", 0)
        elapsed = result.get("elapsed", total_time)
        ttft = result.get("ttft")
        tps = n_tokens / elapsed if elapsed > 0 else 0

        rows = [
            ("Modello", model),
            ("Tokens generati", str(n_tokens)),
            ("Tempo totale", "%.2fs" % elapsed),
            ("Tokens/sec", "%.1f" % tps),
            ("TTFT (Time to First Token)",
             "%.0f ms" % (ttft * 1000) if ttft else "N/A"),
            ("Max Tokens richiesti", str(self.bench_tokens_var.get())),
            ("Temperatura", str(self.bench_temp_var.get())),
        ]
        for metric, value in rows:
            self.bench_tree.insert("", "end", values=(metric, value))

        # Show response
        self.bench_response_text.configure(state="normal")
        self.bench_response_text.delete("1.0", "end")
        self.bench_response_text.insert("1.0", text[:2000])
        self.bench_response_text.configure(state="disabled")

    # ---- Tab 6: WebUI (Open WebUI / Docker) --------------------------------

    def _build_webui_tab(self):
        frame = ttk.Frame(self.notebook, padding=10)
        self.notebook.add(frame, text=" WebUI ")

        info = ttk.LabelFrame(frame, text="Open WebUI (Docker)", padding=8)
        info.pack(fill="x", pady=(0, 8))

        self.webui_status_label = ttk.Label(info, text="Status: sconosciuto",
                                             font=("Segoe UI", 10, "bold"))
        self.webui_status_label.pack(anchor="w", pady=(0, 6))

        bf = ttk.Frame(info)
        bf.pack(fill="x")
        self.webui_start_btn = ttk.Button(bf, text="Start Open WebUI",
                                           command=self._start_webui)
        self.webui_start_btn.pack(side="left", padx=(0, 8))
        self.webui_stop_btn = ttk.Button(bf, text="Stop Open WebUI",
                                          command=self._stop_webui)
        self.webui_stop_btn.pack(side="left", padx=(0, 8))
        ttk.Button(bf, text="Apri nel Browser",
                    command=self._open_webui_browser).pack(side="left", padx=(0, 8))
        ttk.Button(bf, text="Aggiorna Status",
                    command=self._check_webui_status).pack(side="left")

        # Docker command config
        cf = ttk.LabelFrame(frame, text="Configurazione", padding=8)
        cf.pack(fill="x", pady=(0, 8))
        ttk.Label(cf, text="Container name:").grid(row=0, column=0, sticky="w")
        self.webui_container_var = tk.StringVar(value="open-webui")
        ttk.Entry(cf, textvariable=self.webui_container_var, width=30).grid(
            row=0, column=1, padx=8, sticky="w")
        ttk.Label(cf, text="Docker image:").grid(row=1, column=0, sticky="w", pady=4)
        self.webui_image_var = tk.StringVar(
            value="ghcr.io/open-webui/open-webui:main")
        ttk.Entry(cf, textvariable=self.webui_image_var, width=50).grid(
            row=1, column=1, padx=8, sticky="ew")
        ttk.Label(cf, text="Port:").grid(row=2, column=0, sticky="w")
        self.webui_port_var = tk.IntVar(value=3000)
        ttk.Entry(cf, textvariable=self.webui_port_var, width=8).grid(
            row=2, column=1, padx=8, sticky="w")
        cf.columnconfigure(1, weight=1)

        # Docker log
        lf = ttk.LabelFrame(frame, text="Docker Log", padding=6)
        lf.pack(fill="both", expand=True)
        self.webui_log = tk.Text(lf, state="disabled", bg="#1e1e1e", fg="#d4d4d4",
                                  font=("Consolas", 9), wrap="word")
        self.webui_log.pack(fill="both", expand=True)

        self._check_webui_status()

    def _webui_log_append(self, text):
        self.webui_log.configure(state="normal")
        self.webui_log.insert("end", text + "\n")
        self.webui_log.see("end")
        self.webui_log.configure(state="disabled")

    def _docker_run(self, args, timeout=30):
        """Run a docker command via Docker Desktop (docker.exe on Windows)."""
        try:
            proc = subprocess.run(
                ["docker"] + args,
                capture_output=True, text=True, timeout=timeout,
                creationflags=NO_WINDOW,
            )
            return proc.returncode, proc.stdout, proc.stderr
        except Exception as e:
            return -1, "", str(e)

    def _check_webui_status(self):
        def _check():
            container = self.webui_container_var.get()
            rc, out, _ = self._docker_run(
                ["ps", "--filter", "name=%s" % container, "--format", "{{.Status}}"],
                timeout=10)
            status = out.strip() if rc == 0 and out.strip() else "non in esecuzione"
            self.after(0, lambda: self.webui_status_label.configure(
                text="Status: %s" % status))
        threading.Thread(target=_check, daemon=True).start()

    def _start_webui(self):
        container = self.webui_container_var.get()
        image = self.webui_image_var.get()
        port = self.webui_port_var.get()
        self._webui_log_append("Avvio %s..." % container)

        def _run():
            # Try to start existing container first
            rc, out, err = self._docker_run(["start", container], timeout=15)
            if rc == 0:
                self.after(0, lambda: self._webui_log_append(
                    "Container '%s' avviato." % container))
            else:
                # Run new container
                rc2, out2, err2 = self._docker_run([
                    "run", "-d", "--name", container,
                    "-p", "%d:8080" % port,
                    "--add-host=host.docker.internal:host-gateway",
                    "-v", "open-webui:/app/backend/data",
                    "--restart", "always",
                    image,
                ], timeout=60)
                msg = out2.strip() or err2.strip() or "done"
                self.after(0, lambda: self._webui_log_append("docker run: %s" % msg))
            self.after(500, self._check_webui_status)
        threading.Thread(target=_run, daemon=True).start()

    def _stop_webui(self):
        container = self.webui_container_var.get()
        self._webui_log_append("Arresto %s..." % container)

        def _run():
            rc, out, err = self._docker_run(["stop", container], timeout=20)
            msg = "Container fermato." if rc == 0 else (err.strip() or "errore")
            self.after(0, lambda: self._webui_log_append(msg))
            self.after(500, self._check_webui_status)
        threading.Thread(target=_run, daemon=True).start()

    def _open_webui_browser(self):
        port = self.webui_port_var.get()
        import webbrowser
        webbrowser.open("http://localhost:%d" % port)

    # ---- Tab 7: Profiles ---------------------------------------------------

    def _build_profiles_tab(self):
        frame = ttk.Frame(self.notebook, padding=10)
        self.notebook.add(frame, text=" Profili ")

        lf = ttk.LabelFrame(frame, text="Profili Salvati", padding=8)
        lf.pack(fill="both", expand=True, side="left")

        self.profiles_listbox = tk.Listbox(lf, width=35, font=("Segoe UI", 10))
        self.profiles_listbox.pack(fill="both", expand=True)
        self.profiles_listbox.bind("<<ListboxSelect>>", self._on_profile_selected)
        self._refresh_profiles_list()

        # Details panel
        dp = ttk.Frame(frame, padding=(10, 0, 0, 0))
        dp.pack(fill="both", expand=True, side="left")

        self.profile_detail_text = tk.Text(dp, state="disabled", height=12,
                                            font=("Consolas", 9), bg="#f5f5f5")
        self.profile_detail_text.pack(fill="both", expand=True, pady=(0, 8))

        bf = ttk.Frame(dp)
        bf.pack(fill="x")
        ttk.Button(bf, text="Carica Profilo", command=self._load_profile).pack(
            fill="x", pady=2)
        ttk.Button(bf, text="Salva Corrente come Nuovo",
                    command=self._quick_save_profile).pack(fill="x", pady=2)
        ttk.Button(bf, text="Elimina Profilo", command=self._delete_profile).pack(
            fill="x", pady=2)
        ttk.Separator(bf).pack(fill="x", pady=6)
        ttk.Button(bf, text="Esporta Profili (JSON)",
                    command=self._export_profiles).pack(fill="x", pady=2)
        ttk.Button(bf, text="Importa Profili (JSON)",
                    command=self._import_profiles).pack(fill="x", pady=2)

    def _refresh_profiles_list(self):
        self.profiles_listbox.delete(0, "end")
        for name in self.config.get_profiles():
            self.profiles_listbox.insert("end", name)

    def _on_profile_selected(self, _event):
        sel = self.profiles_listbox.curselection()
        if not sel:
            return
        name = self.profiles_listbox.get(sel[0])
        profile = self.config.get_profiles().get(name, {})
        self.profile_detail_text.configure(state="normal")
        self.profile_detail_text.delete("1.0", "end")
        self.profile_detail_text.insert("1.0", json.dumps(profile, indent=2))
        self.profile_detail_text.configure(state="disabled")

    def _load_profile(self):
        sel = self.profiles_listbox.curselection()
        if not sel:
            messagebox.showinfo("Nessuna selezione", "Seleziona un profilo.")
            return
        name = self.profiles_listbox.get(sel[0])
        profile = self.config.get_profiles().get(name)
        if profile:
            self._apply_profile(profile)
            self.notebook.select(0)
            messagebox.showinfo("Profilo Caricato", "Profilo '%s' caricato." % name)

    def _delete_profile(self):
        sel = self.profiles_listbox.curselection()
        if not sel:
            return
        name = self.profiles_listbox.get(sel[0])
        if messagebox.askyesno("Conferma", "Eliminare il profilo '%s'?" % name):
            self.config.delete_profile(name)
            self._refresh_profiles_list()

    def _export_profiles(self):
        path = filedialog.asksaveasfilename(
            defaultextension=".json", filetypes=[("JSON", "*.json")],
            title="Esporta Profili")
        if not path:
            return
        profiles = self.config.get_profiles()
        Path(path).write_text(json.dumps(profiles, indent=2), "utf-8")
        messagebox.showinfo("Esportazione", "Profili esportati in:\n%s" % path)

    def _import_profiles(self):
        path = filedialog.askopenfilename(
            filetypes=[("JSON", "*.json")], title="Importa Profili")
        if not path:
            return
        try:
            imported = json.loads(Path(path).read_text("utf-8"))
            if not isinstance(imported, dict):
                raise ValueError("Formato non valido")
            for name, profile in imported.items():
                self.config.save_profile(name, profile)
            self._refresh_profiles_list()
            messagebox.showinfo("Importazione",
                                 "Importati %d profili." % len(imported))
        except Exception as e:
            messagebox.showerror("Errore", "Impossibile importare:\n%s" % e)

    # ---- Polling / background updates -------------------------------------

    def _poll_logs(self):
        """Drain log queue and append to UI."""
        count = 0
        while count < 200:  # Limit per cycle
            try:
                line = self.log_queue.get_nowait()
            except queue.Empty:
                break
            self._append_log(line)
            count += 1
        self.after(250, self._poll_logs)

    def _append_log(self, line):
        for widget in (self.log_text, self.server_log_preview):
            widget.configure(state="normal")
            tag = None
            lower = line.lower()
            if "error" in lower:
                tag = "error"
            elif "warning" in lower or "warn" in lower:
                tag = "warning"
            elif line.startswith("[INFO]") or line.startswith("[CMD]"):
                tag = "info"
            if tag and widget is self.log_text:
                widget.insert("end", line, tag)
            else:
                widget.insert("end", line)
            widget.see("end")
            widget.configure(state="disabled")

    def _poll_server_status(self):
        """Check if vLLM API is responsive."""
        def _check():
            online = self.api.health()
            self.after(0, lambda: self._update_server_status(online))
        threading.Thread(target=_check, daemon=True).start()
        self.after(5000, self._poll_server_status)

    def _update_server_status(self, online):
        self.server_online = online
        self._draw_indicator(online)
        if online:
            self.status_label.configure(text="Server: online")
            self.start_btn.configure(state="disabled")
            self.stop_btn.configure(state="normal")
        elif self.vllm.running:
            self.status_label.configure(text="Server: caricamento...")
            self.start_btn.configure(state="disabled")
            self.stop_btn.configure(state="normal")
        else:
            self.status_label.configure(text="Server: offline")
            self.start_btn.configure(state="normal")
            self.stop_btn.configure(state="disabled")

    def _poll_gpu_ui(self):
        self._update_gpu_ui()
        self.after(2000, self._poll_gpu_ui)

    # ---- Cleanup ----------------------------------------------------------

    def _on_close(self):
        if self.vllm.running or self.server_online:
            answer = messagebox.askyesnocancel(
                "Server attivo",
                "Il server vLLM e' ancora attivo.\n\n"
                "Si = Stoppa server e esci\n"
                "No = Esci senza fermare nulla\n"
                "Annulla = Torna all'app")
            if answer is None:  # Cancel
                return
            if answer:  # Yes — stop server
                self.vllm.stop()
        self.gpu_mon.stop()
        self.destroy()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    app = App()
    app.mainloop()


if __name__ == "__main__":
    main()
