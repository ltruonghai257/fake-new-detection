#!/usr/bin/env python3
"""
Auto-sync training artifacts from Vast.ai to local machine.
Runs on your local machine in a terminal and polls every N minutes.

Usage:
    python vastai/autosync_vastai.py                # uses saved .vastai_config.json
    python vastai/autosync_vastai.py --interval 3   # sync every 3 minutes
    python vastai/autosync_vastai.py --ip <IP> --port <PORT> --interval 5
    Ctrl+C to stop cleanly.
"""

import subprocess
import sys
import json
import time
import argparse
from pathlib import Path
from datetime import datetime
from typing import Optional, Tuple

# ── Config ───────────────────────────────────────────────────────────────────
REMOTE_BASE  = "/workspace/fake-new-detection"
LOCAL_BASE   = str(Path(__file__).parent.parent)

# Directories to sync from remote → local (remote_subpath, local_subpath)
SYNC_TARGETS = [
    ("training/checkpoints_coolant/", "training/checkpoints_coolant/"),
    ("checkpoints/",                  "checkpoints/"),
    ("stage2_results/",               "stage2_results/"),
    ("logs/",                         "logs/"),
    ("mlruns/",                       "mlruns/"),
]

# ANSI colours
class C:
    GREEN  = "\033[92m"
    YELLOW = "\033[93m"
    RED    = "\033[91m"
    CYAN   = "\033[96m"
    BOLD   = "\033[1m"
    END    = "\033[0m"

def ts() -> str:
    return datetime.now().strftime("%H:%M:%S")

def log_ok(msg: str):   print(f"{C.GREEN}[{ts()}] ✓ {msg}{C.END}")
def log_warn(msg: str): print(f"{C.YELLOW}[{ts()}] ⚠ {msg}{C.END}")
def log_err(msg: str):  print(f"{C.RED}[{ts()}] ✗ {msg}{C.END}")
def log_info(msg: str): print(f"{C.CYAN}[{ts()}] ℹ {msg}{C.END}")

# ── Helpers ───────────────────────────────────────────────────────────────────
def load_config() -> Optional[dict]:
    config_path = Path(__file__).parent / ".vastai_config.json"
    if config_path.exists():
        return json.loads(config_path.read_text())
    # Also check project root (where setup_vastai.py saves it)
    config_path = Path(__file__).parent.parent / ".vastai_config.json"
    if config_path.exists():
        return json.loads(config_path.read_text())
    return None

def find_ssh_key() -> Optional[str]:
    ssh_dir = Path.home() / ".ssh"
    for name in ["id_ed25519", "id_rsa", "id_ecdsa"]:
        if (ssh_dir / name).exists():
            return str(ssh_dir / name)
    return None

def rsync_pull(ip: str, port: str, key: str, remote_dir: str, local_dir: str) -> Tuple[bool, str]:
    """
    Pull remote_dir → local_dir via rsync (delta, no delete).
    Returns (success, stderr_or_empty).
    """
    Path(local_dir).mkdir(parents=True, exist_ok=True)
    ssh_opt = f"ssh -p {port} -i {key} -o StrictHostKeyChecking=no -o ConnectTimeout=15 -o BatchMode=yes"
    cmd = (
        f'rsync -az --update --progress '
        f'-e "{ssh_opt}" '
        f'root@{ip}:{remote_dir}/ {local_dir}/'
    )
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    return result.returncode == 0, result.stderr.strip()

def remote_dir_exists(ip: str, port: str, key: str, remote_dir: str) -> bool:
    ssh_opt = f"ssh -p {port} -i {key} -o StrictHostKeyChecking=no -o ConnectTimeout=10 -o BatchMode=yes"
    result = subprocess.run(
        f'{ssh_opt} root@{ip} "test -d {remote_dir} && echo ok"',
        shell=True, capture_output=True, text=True
    )
    return result.returncode == 0 and "ok" in result.stdout

def local_size(path: str) -> str:
    result = subprocess.run(f"du -sh {path} 2>/dev/null", shell=True, capture_output=True, text=True)
    if result.returncode == 0 and result.stdout.strip():
        return result.stdout.split()[0]
    return "0"

# ── Core sync loop ────────────────────────────────────────────────────────────
def run_sync_cycle(ip: str, port: str, key: str) -> dict:
    """Run one sync cycle. Returns stats dict."""
    synced, skipped, failed = [], [], []

    for remote_sub, local_sub in SYNC_TARGETS:
        remote_dir = f"{REMOTE_BASE}/{remote_sub.rstrip('/')}"
        local_dir  = f"{LOCAL_BASE}/{local_sub.rstrip('/')}"

        if not remote_dir_exists(ip, port, key, remote_dir):
            skipped.append(remote_sub)
            continue

        ok, err = rsync_pull(ip, port, key, remote_dir, local_dir)
        if ok:
            size = local_size(local_dir)
            synced.append(f"{remote_sub} ({size})")
        else:
            failed.append(remote_sub)
            if err:
                log_warn(f"  rsync stderr: {err[:120]}")

    return {"synced": synced, "skipped": skipped, "failed": failed}

def run_watcher(ip: str, port: str, key: str, interval_min: int):
    cycle = 0
    print(f"\n{C.BOLD}Auto-sync started → {ip}:{port}{C.END}")
    print(f"  Syncing every {interval_min} min.  Press Ctrl+C to stop.\n")

    while True:
        cycle += 1
        log_info(f"Cycle {cycle}: syncing from {ip}:{port} ...")
        stats = run_sync_cycle(ip, port, key)

        if stats["synced"]:
            for s in stats["synced"]:
                log_ok(f"  {s}")
        if stats["skipped"]:
            log_info(f"  Not yet on remote: {', '.join(stats['skipped'])}")
        if stats["failed"]:
            for f in stats["failed"]:
                log_err(f"  failed: {f}")
            log_warn("  (instance may be stopped — will retry next cycle)")

        if not stats["synced"] and not stats["failed"]:
            log_info("  Nothing to sync yet (training may not have started)")

        next_at = datetime.now().strftime("%H:%M:%S")
        print(f"  Next sync in {interval_min} min  [{next_at}]\n")
        time.sleep(interval_min * 60)

# ── Entry point ───────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Auto-sync Vast.ai training artifacts to local machine")
    parser.add_argument("--ip",       type=str, help="Instance IP (overrides saved config)")
    parser.add_argument("--port",     type=str, help="SSH port (overrides saved config)")
    parser.add_argument("--interval", type=int, default=5, help="Sync interval in minutes (default: 5)")
    args = parser.parse_args()

    # Resolve connection
    config = load_config()
    ip   = args.ip   or (config and config.get("ip"))
    port = args.port or (config and config.get("port"))

    if not ip or not port:
        print(f"{C.RED}No connection details found.{C.END}")
        print("Run setup first:  python vastai/setup_vastai.py --ip <IP> --port <PORT>")
        print("Or pass directly: python vastai/autosync_vastai.py --ip <IP> --port <PORT>")
        sys.exit(1)

    key = find_ssh_key()
    if not key:
        print(f"{C.RED}No SSH key found in ~/.ssh/{C.END}")
        sys.exit(1)

    try:
        run_watcher(ip, port, key, args.interval)
    except KeyboardInterrupt:
        print(f"\n{C.BOLD}Auto-sync stopped.{C.END}")
        print(f"Local artifacts are in: {LOCAL_BASE}")
        sys.exit(0)

if __name__ == "__main__":
    main()
