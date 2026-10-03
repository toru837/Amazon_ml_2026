"""
Step 2: candidate generation for a split (train or test).

Writes cache/<split>_cands/<country>_<chunk>.parquet with columns
  q_idx   : row in the concatenated query table (Source 2 rows, then Source 3 rows)
  s1_idx  : row in the Source 1 table
  cos_name, cos_addr, cos_comb, rank (rank of this S1 among the query's candidates)
"""
import sys
import time
import gc
import argparse

import numpy as np
import polars as pl
import torch

from config import cache_path
import retrieval
from retrieval import CountryIndex

K_APPROX = 40   # GPU approximate neighbours per query
K_KEEP = 10     # exact re-scored neighbours kept per query
CHUNK = 50_000


def scan_tables(split: str):
    """Lazy frames; materialised one country at a time to bound memory."""
    cols = ["country", "name_core", "addr_norm"]
    s1 = pl.scan_parquet(cache_path(f"{split}_source1_norm.parquet")).select(["entity_id"] + cols).with_row_index("s1_idx")
    q = pl.concat([pl.scan_parquet(cache_path(f"{split}_source2_norm.parquet")).select(cols),
                   pl.scan_parquet(cache_path(f"{split}_source3_norm.parquet")).select(cols)]).with_row_index("q_idx")
    return s1, q


def _build_index(s1c: pl.DataFrame) -> CountryIndex:
    try:
        return CountryIndex(s1c)
    except torch.cuda.OutOfMemoryError:
        retrieval.DEVICE = "cpu"
        gc.collect()
        torch.cuda.empty_cache()
        print("  CUDA out of memory while building index; retrying on CPU", flush=True)
        return CountryIndex(s1c)


def generate(split: str, only_country: str | None = None, chunk_size: int = CHUNK):
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    out_dir = cache_path(f"{split}_cands")
    out_dir.mkdir(parents=True, exist_ok=True)  # resumable: finished chunks are skipped
    s1, q = scan_tables(split)
    s1_sizes = dict(s1.group_by("country").len().collect().iter_rows())
    s1_countries = set(s1_sizes)
    countries = sorted(
        s1_sizes,
        key=lambda c: (c.casefold() in {"us", "usa", "united states", "india"}, s1_sizes[c]),
    )
    if only_country is not None:
        if only_country not in s1_sizes:
            raise ValueError(f"Unknown {split} country: {only_country}")
        countries = [only_country]
    q_countries = set(q.select(pl.col("country").unique()).collect()["country"].to_list())
    orphan = q_countries - s1_countries
    if orphan:
        print(f"note: query countries without any Source 1 record (no candidates possible): {orphan}")
    total = 0
    for country in countries:
        t0 = time.time()
        s1c = s1.filter(pl.col("country") == country).collect()
        qc = q.filter(pl.col("country") == country).collect()
        if qc.height == 0:
            continue
        done_marker = out_dir / f"{country}.done"
        if done_marker.exists():
            print(f"[{split}/{country}] done already, skipping")
            continue
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        idx = _build_index(s1c)
        s1_map = s1c["s1_idx"].to_numpy()
        print(f"[{split}/{country}] index {s1c.height:,} S1 built in {time.time()-t0:.0f}s; {qc.height:,} queries", flush=True)
        for ci, s in enumerate(range(0, qc.height, chunk_size)):
            out_file = out_dir / f"{country}_{s:09d}.parquet"  # named by query offset -> resumable
            if out_file.exists():
                continue
            part = qc.slice(s, chunk_size)
            try:
                Qn, Qa = idx.encode(part)
                cand = idx.candidates(Qn, Qa, K=K_APPROX, k=K_KEEP)
            except torch.cuda.OutOfMemoryError:
                retrieval.DEVICE = "cpu"
                del idx
                gc.collect()
                torch.cuda.empty_cache()
                print(f"  CUDA out of memory in chunk {ci}; rebuilding index on CPU", flush=True)
                idx = CountryIndex(s1c)
                Qn, Qa = idx.encode(part)
                cand = idx.candidates(Qn, Qa, K=K_APPROX, k=K_KEEP)
            q_map = part["q_idx"].to_numpy()
            cand = cand.with_columns(
                pl.Series("q_idx", q_map[cand["q_row"].to_numpy()], dtype=pl.UInt32),
                pl.Series("s1_idx", s1_map[cand["s1_row"].to_numpy()], dtype=pl.UInt32),
            ).select("q_idx", "s1_idx", "cos_name", "cos_addr", "cos_comb", "rank")
            cand.write_parquet(out_file.with_suffix(".tmp"))
            out_file.with_suffix(".tmp").replace(out_file)
            total += cand.height
            print(f"  chunk {ci}: {part.height:,} queries -> {cand.height:,} pairs ({time.time()-t0:.0f}s)", flush=True)
        done_marker.touch()
        del idx
        del s1c, qc
        gc.collect()
        if torch.cuda.is_available():
            print(
                f"[{split}/{country}] peak CUDA allocated="
                f"{torch.cuda.max_memory_allocated() / 2**30:.2f} GiB, reserved="
                f"{torch.cuda.max_memory_reserved() / 2**30:.2f} GiB",
                flush=True,
            )
        torch.cuda.empty_cache()
    print(f"[{split}] total candidate pairs: {total:,}")


def patch_queries(split: str, flag: str = "has_indic"):
    """Re-generate the candidates of the queries whose `flag` is set (e.g. after their names were
    re-normalized) and splice them into the existing chunk files. The S1 index is unchanged, so this
    is equivalent to a full regeneration."""
    out_dir = cache_path(f"{split}_cands")
    marker = out_dir / f"patched_{flag}.done"
    if marker.exists():
        print(f"[{split}] {flag} patch already applied")
        return
    s1, q = scan_tables(split)
    flags = pl.concat([pl.scan_parquet(cache_path(f"{split}_{s}_norm.parquet")).select(flag)
                       for s in ("source2", "source3")]).with_row_index("q_idx")
    q = q.join(flags, on="q_idx").filter(pl.col(flag))
    countries = q.select(pl.col("country").unique()).collect()["country"].to_list()
    for country in countries:
        t0 = time.time()
        qc = q.filter(pl.col("country") == country).collect()
        s1c = s1.filter(pl.col("country") == country).collect()
        if s1c.height == 0:
            continue
        idx = CountryIndex(s1c)
        s1_map = s1c["s1_idx"].to_numpy()
        parts = []
        for s in range(0, qc.height, CHUNK):
            part = qc.slice(s, CHUNK)
            Qn, Qa = idx.encode(part)
            cand = idx.candidates(Qn, Qa, K=K_APPROX, k=K_KEEP)
            parts.append(cand.with_columns(
                pl.Series("q_idx", part["q_idx"].to_numpy()[cand["q_row"].to_numpy()], dtype=pl.UInt32),
                pl.Series("s1_idx", s1_map[cand["s1_row"].to_numpy()], dtype=pl.UInt32),
            ).select("q_idx", "s1_idx", "cos_name", "cos_addr", "cos_comb", "rank"))
        new = pl.concat(parts)
        patched = qc.select(pl.col("q_idx").cast(pl.UInt32))
        for f in sorted(out_dir.glob(f"{country}_*.parquet")):
            if f.name == f"{country}_patch_{flag}.parquet":
                continue  # previous patch output is replaced below
            old = pl.read_parquet(f, memory_map=False)  # no mmap: file may be rewritten in place (Windows)
            keep = old.join(patched, on="q_idx", how="anti")
            if keep.height != old.height:
                keep.write_parquet(f)
        new.write_parquet(out_dir / f"{country}_patch_{flag}.parquet")
        print(f"[{split}/{country}] re-generated candidates for {qc.height:,} {flag} queries ({time.time()-t0:.0f}s)", flush=True)
        del idx
        torch.cuda.empty_cache()
    marker.touch()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("splits", nargs="*", default=["train", "test"])
    parser.add_argument("--country")
    parser.add_argument("--chunk-size", type=int, default=CHUNK)
    args = parser.parse_args()
    for sp_ in args.splits or ["train", "test"]:
        generate(sp_, args.country, args.chunk_size)
