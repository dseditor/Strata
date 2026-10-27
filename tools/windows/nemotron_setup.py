"""strata-nemotron on Windows: check the PC, install, download, start.  Run by START-NEMOTRON.bat inside .venv.

Every step is skipped when it is already done, so the same double-click is the first install and every later start:
  1. the PC: an RTX 50-series card (sm_120), driver 580+, RAM and disk
  2. the Python packages of requirements.txt (+ huggingface_hub), once
  3. the Windows engine (build/strata-nemotron.exe, the vision encoder, cuBLAS) from the fork's release
  4. the model's two shards from Hugging Face (resumable), and mmproj-BF16.gguf if images are wanted
  5. the tokenizer folder, taken from the GGUF
  6. the server on http://127.0.0.1:8080/ with the browser opened when the model is ready

  --yes          no questions (images on unless --no-images)
  --no-images    text only: no mmproj download, no vision encoder
  --port N       the server's port (8080)
  --no-open      do not open the browser
  --force        start even when the RAM check fails
"""
from __future__ import annotations

import argparse
import ctypes
import json
import os
import shutil
import subprocess
import sys
import time
import urllib.request
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
MODELS = ROOT / "models"
RELEASE_ZIP = ("https://github.com/dseditor/Strata/releases/download/windows-nemotron-v1/"
               "strata-nemotron-windows-x64.zip")
MODEL_REPO = "Cudecnik/Nemotron-3.5-Super-120B-A12B-NVFP4-Strata-Blackwell"
SHARDS = ["Nemotron-3.5-Super-120B-A12B-NVFP4-00001-of-00002.gguf",
          "Nemotron-3.5-Super-120B-A12B-NVFP4-00002-of-00002.gguf"]
MMPROJ_REPO = "unsloth/NVIDIA-Nemotron-3.5-Super-VL-120B-A12B-GGUF"
MMPROJ = "mmproj-BF16.gguf"
CONFIG = ROOT / "docs" / "nemotron" / "strata-nemotron.windows.json"
MARK = ROOT / ".venv" / ".nemotron-packages"
MIN_RAM_GIB = 88          # the experts take ~60 GiB page-locked; the README's 96 GB machine shows ~89 GiB usable
MIN_DRIVER = 580


def say(msg=""):
    print(msg, flush=True)


def stop(msg):
    say()
    say("  " + msg.replace("\n", "\n  "))
    sys.exit(1)


def ask(question, default_yes, args):
    if args.yes:
        return default_yes
    hint = "[Y/n]" if default_yes else "[y/N]"
    ans = input(f"  {question} {hint} ").strip().lower()
    return default_yes if not ans else ans.startswith("y")


# ---------------------------------------------------------------- 1. the PC
def ram_gib():
    class MS(ctypes.Structure):
        _fields_ = [("dwLength", ctypes.c_ulong), ("dwMemoryLoad", ctypes.c_ulong),
                    ("ullTotalPhys", ctypes.c_ulonglong), ("ullAvailPhys", ctypes.c_ulonglong),
                    ("ullTotalPageFile", ctypes.c_ulonglong), ("ullAvailPageFile", ctypes.c_ulonglong),
                    ("ullTotalVirtual", ctypes.c_ulonglong), ("ullAvailVirtual", ctypes.c_ulonglong),
                    ("ullAvailExtendedVirtual", ctypes.c_ulonglong)]
    m = MS()
    m.dwLength = ctypes.sizeof(MS)
    ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(m))
    return m.ullTotalPhys / 2**30


def check_pc(args):
    say("[1/6] Checking this PC")
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=name,compute_cap,driver_version,memory.total",
                              "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=30).stdout
    except (OSError, subprocess.TimeoutExpired):
        stop("nvidia-smi was not found: an NVIDIA graphics driver is needed (RTX 50-series, driver 580 or newer).")
    gpus = [[p.strip() for p in line.split(",")] for line in out.strip().splitlines() if line.strip()]
    if not gpus:
        stop("No NVIDIA GPU was found.")
    ok = [g for g in gpus if float(g[1]) >= 12.0]
    for g in gpus:
        say(f"      GPU  {g[0]}  compute {g[1]}  driver {g[2]}  {int(g[3]) / 1024:.0f} GB")
    if not ok:
        stop("This engine needs an RTX 50-series card (Blackwell, compute 12.0): it is built on their FP4 tensor\n"
             "cores. RTX 40/30-series cards cannot run it. Nothing was downloaded.")
    if int(ok[0][2].split(".")[0]) < MIN_DRIVER:
        stop(f"The NVIDIA driver is {ok[0][2]}; CUDA 13 needs {MIN_DRIVER} or newer. Update the driver and run again.")
    ram = ram_gib()
    say(f"      RAM  {ram:.0f} GiB")
    if ram < MIN_RAM_GIB:
        msg = (f"This PC has {ram:.0f} GiB of RAM. The model keeps about 60 GiB of experts in RAM, and 64 GB machines\n"
               f"do not fit; 96 GB is the tested size.")
        if not args.force:
            stop(msg + " Run with --force to try anyway.")
        say("      " + msg.replace("\n", " ") + " (--force: going on)")


def check_disk(need_disk_gib):
    free = shutil.disk_usage(ROOT).free / 2**30
    say(f"      disk {free:.0f} GiB free here, {need_disk_gib:.0f} GiB still to download")
    if free < need_disk_gib + 2:
        stop(f"Not enough free disk on this drive: {need_disk_gib:.0f} GiB is needed. Move this folder to a bigger drive.")


# ---------------------------------------------------------------- 2. packages
def packages():
    if MARK.exists():
        return
    say("[2/6] Installing the Python packages into .venv (once)")
    subprocess.run([sys.executable, "-m", "pip", "install", "--disable-pip-version-check", "-q",
                    "-r", str(ROOT / "requirements.txt"), "huggingface_hub"], check=True)
    MARK.write_text("ok", encoding="utf-8")


# ---------------------------------------------------------------- 3. engine
def engine(images):
    need = [ROOT / "build" / "strata-nemotron.exe", ROOT / "build" / "cublas64_13.dll",
            ROOT / "build" / "cublasLt64_13.dll"]
    if images:
        need.append(ROOT / "build-vision" / "bin" / "strata-vision.exe")
    if all(p.exists() for p in need):
        return
    say("[3/6] Downloading the Windows engine (about 450 MB)")
    tmp = ROOT / "strata-nemotron-windows-x64.zip.part"
    with urllib.request.urlopen(RELEASE_ZIP, timeout=60) as r, open(tmp, "wb") as f:
        total = int(r.headers.get("Content-Length") or 0)
        done, t0 = 0, time.time()
        while chunk := r.read(1 << 20):
            f.write(chunk)
            done += len(chunk)
            if total:
                print(f"\r      {done / 2**20:.0f} / {total / 2**20:.0f} MB", end="", flush=True)
    say(f"\r      {done / 2**20:.0f} MB in {time.time() - t0:.0f} s" + " " * 20)
    if total and done != total:
        tmp.unlink(missing_ok=True)
        stop("The engine download was cut off. Run again.")
    with zipfile.ZipFile(tmp) as z:
        z.extractall(ROOT)
    tmp.unlink()


# ---------------------------------------------------------------- 4. model
def remote_sizes(repo, names):
    from huggingface_hub import HfApi
    return {p.path: p.size for p in HfApi().get_paths_info(repo, names)}


def todo_files(images):
    """(repo, file, size) still missing or incomplete in models/."""
    want = [(MODEL_REPO, s) for s in SHARDS] + ([(MMPROJ_REPO, MMPROJ)] if images else [])
    sizes = {}
    for repo in {r for r, _ in want}:
        sizes.update(remote_sizes(repo, [f for r, f in want if r == repo]))
    return [(r, f, sizes.get(f, 0)) for r, f in want
            if not ((MODELS / f).exists() and (MODELS / f).stat().st_size == sizes.get(f, -1))]


def model(files):
    if not files:
        return
    from huggingface_hub import hf_hub_download
    say("[4/6] Downloading the model from Hugging Face (resumable: run again if it stops)")
    MODELS.mkdir(exist_ok=True)
    for repo, f, size in files:
        say(f"      {f}  {size / 2**30:.1f} GiB")
        hf_hub_download(repo, f, local_dir=str(MODELS))


# ---------------------------------------------------------------- 5. tokenizer
def tokenizer():
    if (MODELS / "tokenizer" / "vocab.json").exists():
        return
    say("[5/6] Extracting the tokenizer from the GGUF")
    subprocess.run([sys.executable, str(ROOT / "tools" / "strata_tokenizer.py"), "--gguf", str(MODELS / SHARDS[0]),
                    "--out", str(MODELS)], check=True)


# ---------------------------------------------------------------- 6. server
def serve(args, images):
    cfg = CONFIG
    if not images:
        c = json.loads(CONFIG.read_text(encoding="utf-8"))
        c.pop("vision", None)
        cfg = MODELS / "nemotron-text-only.json"
        cfg.write_text(json.dumps(c, indent=2), encoding="utf-8")
    say(f"[6/6] Starting the model: it takes 1-4 minutes to load (the first start reads ~70 GB from disk).")
    say(f"      Then chat at http://127.0.0.1:{args.port}/  or use http://127.0.0.1:{args.port}/v1 from an OpenAI client.")
    say("      Close this window to stop the model.")
    env = dict(os.environ, PATH=str(ROOT / "build") + os.pathsep + os.environ.get("PATH", ""))
    cmd = [sys.executable, str(ROOT / "serve" / "server.py"), "--engine", "strata", "--config", str(cfg),
           "--port", str(args.port)] + ([] if args.no_open else ["--open"])
    return subprocess.call(cmd, cwd=str(ROOT), env=env)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--yes", action="store_true")
    ap.add_argument("--no-images", action="store_true")
    ap.add_argument("--port", type=int, default=8080)
    ap.add_argument("--no-open", action="store_true")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    say("Nemotron-3.5-Super-120B (strata-nemotron, Windows)")
    say()
    check_pc(args)
    packages()
    # images: asked once on the first install; later starts follow what is on disk
    if args.no_images:
        images = False
    elif (MODELS / MMPROJ).exists():
        images = True
    elif (MODELS / SHARDS[0]).exists() and (MODELS / "tokenizer").exists():
        images = False                                  # installed before without images
    else:
        images = ask("Also install image input (1.7 GB more, 2.9 GB of VRAM while running)?", True, args)
    files = todo_files(images)
    engine_gib = 0 if (ROOT / "build" / "strata-nemotron.exe").exists() else 0.5
    check_disk(sum(s for _, _, s in files) / 2**30 + engine_gib)
    engine(images)
    model(files)
    tokenizer()
    return serve(args, images)


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        sys.exit(130)
