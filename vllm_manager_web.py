#!/usr/bin/env python3
"""vLLM Manager — Web interface for remote management of vLLM servers.

Run with: python vllm_manager_web.py [--host 0.0.0.0] [--port 5000]
Then open http://<server-ip>:5000 in your browser.

Requires: pip install flask
"""

import abc
import argparse
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
              tp_size=1, attention_backend=""):
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
                                  is_dgx=platform_info.is_dgx,
                                  is_macos=platform_info.os == "macos",
                                  app_version=APP_VERSION,
                                  chat_templates=["(auto)"] + [k for k in CHAT_TEMPLATES if k])


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
    def _install():
        if platform_info.is_dgx:
            _install_vllm_dgx()
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


def _install_vllm_dgx():
    """Full DGX Spark installation: uv, PyTorch, Triton, patched vLLM."""
    install_dir = DGX_INSTALL_DIR_DEFAULT
    venv_dir = install_dir + "/.vllm"
    activate = "source %s/bin/activate" % venv_dir

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
            activate, DGX_PYTORCH_INDEX),
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
            activate, install_dir, triton_dir, DGX_TRITON_COMMIT),
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
        "git submodule update --init --recursive" % (install_dir, DGX_VLLM_COMMIT),
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
        "python use_existing_torch.py 2>/dev/null || true" % (activate, vllm_dir),
        timeout=30,
    )

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
    <div class="btn-row" style="margin-bottom:8px">
      <button class="btn btn-success" onclick="dgxRayStartHead()">Avvia Ray Head</button>
      <button class="btn btn-primary" onclick="dgxRayConnectWorker()">Connetti Worker</button>
      <button class="btn btn-danger" onclick="dgxRayStop()">Stop Ray</button>
    </div>
    <div id="dgx-ray-status" style="font-size:13px;margin-bottom:8px;color:var(--fg2)">Ray: sconosciuto</div>
    <div class="form-row">
      <div>
        <label>Worker IP</label>
        <input type="text" id="dgx-worker-ip" placeholder="192.168.1.x">
      </div>
      <div>
        <label>User</label>
        <input type="text" id="dgx-worker-user" value="root">
      </div>
    </div>
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

  <!-- SSH Configuration -->
  <div class="card">
    <h3>SSH Configuration</h3>
    <div class="form-row">
      <div>
        <label>Target IP</label>
        <input type="text" id="dgx-ssh-ip" placeholder="192.168.1.x">
      </div>
      <div>
        <label>User</label>
        <input type="text" id="dgx-ssh-user" value="root">
      </div>
    </div>
    <div class="btn-row" style="margin-top:4px">
      <button class="btn btn-secondary" onclick="dgxSSHSetup()">Setup SSH Keys</button>
      <button class="btn btn-secondary" onclick="dgxSSHTest()">Test Connessione</button>
      <span id="dgx-ssh-status" style="font-size:12px;color:var(--fg3);align-self:center"></span>
    </div>
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

async function dgxRayStartHead() {
  await apiPost('/api/dgx/ray/start-head');
  document.getElementById('dgx-ray-status').textContent = 'Ray: avvio head in corso...';
  showToast('Avvio Ray head...', 'success');
  setTimeout(refreshDGXCluster, 5000);
}

async function dgxRayConnectWorker() {
  const ip = document.getElementById('dgx-worker-ip').value.trim();
  const user = document.getElementById('dgx-worker-user').value.trim();
  if (!ip) { showToast('Inserisci IP del worker', 'error'); return; }
  await apiPost('/api/dgx/ray/connect-worker', {ip, user});
  document.getElementById('dgx-ray-status').textContent = 'Ray: connessione worker ' + ip + '...';
  showToast('Connessione worker ' + ip + '...', 'success');
}

async function dgxRayStop() {
  await apiPost('/api/dgx/ray/stop');
  document.getElementById('dgx-ray-status').textContent = 'Ray: fermato';
  showToast('Ray fermato', 'success');
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

async function dgxSSHSetup() {
  const ip = document.getElementById('dgx-ssh-ip').value.trim();
  const user = document.getElementById('dgx-ssh-user').value.trim();
  if (!ip) { showToast('Inserisci IP target', 'error'); return; }
  document.getElementById('dgx-ssh-status').textContent = 'Configurazione SSH...';
  await apiPost('/api/dgx/ssh/setup', {ip, user});
  document.getElementById('dgx-ssh-status').textContent = 'Setup avviato, controlla i log.';
}

async function dgxSSHTest() {
  const ip = document.getElementById('dgx-ssh-ip').value.trim();
  const user = document.getElementById('dgx-ssh-user').value.trim();
  if (!ip) { showToast('Inserisci IP target', 'error'); return; }
  document.getElementById('dgx-ssh-status').textContent = 'Test connessione...';
  const data = await apiPost('/api/dgx/ssh/test', {ip, user});
  if (data.status === 'ok') {
    document.getElementById('dgx-ssh-status').textContent = 'OK: connesso a ' + data.hostname + ' (' + ip + ')';
  } else {
    document.getElementById('dgx-ssh-status').textContent = 'FAIL: ' + (data.error || 'connessione fallita');
  }
}

// Auto-load DGX cluster on page load if DGX
if (document.getElementById('tab-dgx')) {
  setTimeout(refreshDGXCluster, 1500);
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
