#!/usr/bin/env python3
"""
Watchdog for browsecomp_eval.py
If a question runs for > MAX_IDLE seconds, kills the process and restarts
from the NEXT question, skipping the stuck one.
"""
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

MAX_IDLE = 3600  # seconds before killing a stuck question

def get_proc_info():
    """Return (pid, indices_list, out_prefix) of running browsecomp process."""
    result = subprocess.run(
        ["pgrep", "-af", "browsecomp_eval.py"], capture_output=True, text=True
    )
    if result.returncode != 0:
        return None
    line = result.stdout.strip().split("\n")[0]
    pid = int(line.split()[0])
    # Parse --indices and --out from command line
    indices = re.search(r"--indices\s+([\d,]+)", line)
    out = re.search(r"--out\s+(\S+)", line)
    extra = re.search(r"--max-iters\s+(\d+)", line)
    max_iters = int(extra.group(1)) if extra else 16
    extra2 = re.search(r"--max-search\s+(\d+)", line)
    max_search = int(extra2.group(1)) if extra2 else 35
    if not indices or not out:
        return None
    idx_list = [int(x) for x in indices.group(1).split(",")]
    return pid, idx_list, out.group(1), max_iters, max_search


def get_current_question(log_path):
    """Return the q_idx (0-based) of the question currently being processed."""
    text = Path(log_path).read_text(errors="replace")
    # Find all [Q{n}] headers
    headers = re.findall(r"\[Q(\d+)\]", text)
    # Find all completed questions (lat= marker)
    completed = re.findall(r"lat=[\d.]+s", text)
    if not headers:
        return None
    # Current question is the last header
    return int(headers[-1]) - 1  # convert to 0-based index


def get_log_path(out_prefix):
    return out_prefix + ".log"


def main():
    if len(sys.argv) < 2:
        print("Usage: watchdog.py <log_file>")
        sys.exit(1)

    log_path = sys.argv[1]
    print(f"[watchdog] Monitoring {log_path} (max idle: {MAX_IDLE}s)")

    last_completion_count = 0
    last_progress_time = time.time()

    while True:
        time.sleep(60)

        info = get_proc_info()
        if info is None:
            print("[watchdog] No browsecomp process found — exiting.")
            break

        pid, idx_list, out_prefix, max_iters, max_search = info

        # Count completions
        try:
            text = Path(log_path).read_text(errors="replace")
            completion_count = len(re.findall(r"lat=[\d.]+s", text))
        except Exception:
            completion_count = last_completion_count

        if completion_count > last_completion_count:
            last_completion_count = completion_count
            last_progress_time = time.time()
            print(f"[watchdog] Progress: {completion_count} completions")
            continue

        idle = time.time() - last_progress_time
        if idle > MAX_IDLE:
            stuck_idx = get_current_question(log_path)
            print(f"[watchdog] STUCK on Q{stuck_idx+1 if stuck_idx is not None else '?'} for {idle:.0f}s — killing PID {pid}")
            os.kill(pid, 9)
            time.sleep(3)

            if stuck_idx is None:
                print("[watchdog] Could not determine current question, stopping.")
                break

            # Find remaining indices (skip stuck one)
            remaining = [i for i in idx_list if i > stuck_idx]
            if not remaining:
                print("[watchdog] No remaining questions.")
                break

            print(f"[watchdog] Skipping Q{stuck_idx+1}, restarting from Q{remaining[0]+1} ({len(remaining)} left)")

            env = os.environ.copy()
            env.setdefault("AZURE_OPENAI_MODEL", "ismail-gpt-5")
            env.setdefault("AZURE_OPENAI_MAX_TOKENS", "12000")
            env.setdefault("AZURE_OPENAI_REASONING_EFFORT", "low")
            env.setdefault("BROWSECOMP_TIMEOUT", "800")

            cmd = [
                "python3", "-u", "examples/browsecomp_eval.py",
                "--indices", ",".join(str(i) for i in remaining),
                "--out", out_prefix,
                "--max-iters", str(max_iters),
                "--max-search", str(max_search),
            ]
            with open(log_path, "a") as log_f:
                log_f.write(f"\n[watchdog] Skipped Q{stuck_idx+1} after {idle:.0f}s, restarting...\n")
                proc = subprocess.Popen(cmd, env=env, stdout=log_f, stderr=log_f)
                print(f"[watchdog] Restarted as PID {proc.pid}")

            # Reset tracking
            last_progress_time = time.time()
            time.sleep(10)


if __name__ == "__main__":
    main()
