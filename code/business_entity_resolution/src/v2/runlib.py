"""Crash-safe run utilities for the v2 work.

* run directories  : cache/runs/<run_id>/ with manifest.json (code/config/input fingerprints, status)
* atomic writes    : write to <file>.tmp, validate, os.replace
* process lock     : cache/runs/<run_id>/.lock holds the owning PID; a live owner blocks a second writer
* memory watchdog  : aborts the process if system free RAM < MIN_FREE_MB (protects the machine)
* progress log     : experiments/v2/PROGRESS.md is appended by `note()`
"""
import hashlib
import json
import os
import sys
import threading
import time
from pathlib import Path

import polars as pl
import psutil

ROOT = Path(__file__).resolve().parents[3]  # repo root (package: business_entity_resolution/src/v2)
SRC = ROOT / "business_entity_resolution" / "src"
sys.path.insert(0, str(SRC))
sys.stdout.reconfigure(encoding="utf-8")
pl.Config.set_tbl_rows(40)
pl.Config.set_tbl_cols(24)
from config import cache_path  # noqa: E402

RUNS = cache_path("runs")
RUNS.mkdir(exist_ok=True)
PROGRESS = ROOT / "experiments" / "v2" / "PROGRESS.md"
PROGRESS.parent.mkdir(parents=True, exist_ok=True)
MIN_FREE_MB = 1500


def _watchdog():
    while True:
        if psutil.virtual_memory().available / 2**20 < MIN_FREE_MB:
            print("\n!!! WATCHDOG: low system memory - aborting to protect the machine", flush=True)
            os._exit(3)
        time.sleep(0.5)


threading.Thread(target=_watchdog, daemon=True).start()


def mem():
    return f"rss={psutil.Process().memory_info().rss/2**30:.2f}GB free={psutil.virtual_memory().available/2**30:.2f}GB"


def sha(path, n=1 << 20):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        while chunk := fh.read(n):
            h.update(chunk)
    return h.hexdigest()


def code_fingerprint(files):
    h = hashlib.sha256()
    for f in sorted(files):
        h.update(Path(f).read_bytes())
    return h.hexdigest()[:16]


class Run:
    def __init__(self, run_id: str, config: dict, code_files=()):
        self.id = run_id
        self.dir = RUNS / run_id
        self.dir.mkdir(parents=True, exist_ok=True)
        self.lock = self.dir / ".lock"
        if self.lock.exists():
            pid = int(self.lock.read_text() or 0)
            if pid and psutil.pid_exists(pid) and pid != os.getpid():
                raise SystemExit(f"run {run_id} is locked by live process {pid}")
        self.lock.write_text(str(os.getpid()))
        self.manifest_path = self.dir / "manifest.json"
        m = json.loads(self.manifest_path.read_text()) if self.manifest_path.exists() else {}
        fp = code_fingerprint([__file__, *code_files])
        if m and m.get("config") != config:
            # configuration changed -> invalidate completed stages
            m["stages"] = {}
        elif m and m.get("code_fp") != fp:
            # code edited: completed stages keep their checksummed outputs; the change is recorded
            m.setdefault("code_fp_history", []).append({"old": m.get("code_fp"), "new": fp, "time": time.ctime()})
        m.update({"run_id": run_id, "config": config, "code_fp": fp, "updated": time.ctime()})
        m.setdefault("stages", {})
        self.m = m
        self._save()

    def _save(self):
        tmp = self.manifest_path.with_suffix(".tmp")
        tmp.write_text(json.dumps(self.m, indent=1, default=str))
        os.replace(tmp, self.manifest_path)

    def done(self, stage):
        s = self.m["stages"].get(stage)
        if not s or s.get("status") != "complete":
            return False
        return all((self.dir / f).exists() and sha(self.dir / f) == h for f, h in s.get("files", {}).items())

    def complete(self, stage, files=(), **info):
        self.m["stages"][stage] = {"status": "complete", "time": time.ctime(),
                                   "files": {f: sha(self.dir / f) for f in files}, **info}
        self._save()

    def path(self, name):
        return self.dir / name

    def write_parquet(self, df: pl.DataFrame, name: str):
        """atomic parquet write with read-back row-count validation"""
        p = self.dir / name
        tmp = p.with_suffix(".tmp")
        df.write_parquet(tmp)
        n = pl.scan_parquet(tmp).select(pl.len()).collect().item()
        if n != df.height:
            raise IOError(f"validation failed for {name}: {n} != {df.height}")
        os.replace(tmp, p)
        return p

    def release(self):
        if self.lock.exists():
            self.lock.unlink()


def note(text: str):
    """append a timestamped line to the durable progress file"""
    with open(PROGRESS, "a", encoding="utf-8") as fh:
        fh.write(f"- [{time.strftime('%Y-%m-%d %H:%M')}] {text}\n")
    print("NOTE:", text, flush=True)
