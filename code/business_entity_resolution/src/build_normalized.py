"""Step 1: normalize names/addresses of every record (multiprocess) and cache as parquet."""
import os
import sys
import time
from multiprocessing import Pool
from pathlib import Path

import polars as pl
import pyarrow.parquet as pq

from config import cache_path
from normalization import normalize_address, normalize_name

N_PROC = max(1, (os.cpu_count() or 2) - 1)


def _norm_batch(batch):
    names, addrs = batch
    out = []
    for n, a in zip(names, addrs):
        nn, nc, dom, ind = normalize_name(n)
        an, nums = normalize_address(a)
        out.append((nn, nc, dom, ind, an, nums))
    return out


def normalize_frame(df: pl.DataFrame, pool, batch=20000) -> pl.DataFrame:
    names = df["business_name"].to_list()
    addrs = df["business_address"].to_list()
    batches = [(names[i:i + batch], addrs[i:i + batch]) for i in range(0, len(names), batch)]
    rows = [r for part in pool.imap(_norm_batch, batches) for r in part]
    cols = list(zip(*rows)) if rows else [[]] * 6
    return df.select("entity_id", "country").with_columns(
        pl.Series("name_norm", cols[0], dtype=pl.Utf8),
        pl.Series("name_core", cols[1], dtype=pl.Utf8),
        pl.Series("is_domain", cols[2], dtype=pl.Boolean),
        pl.Series("has_indic", cols[3], dtype=pl.Boolean),
        pl.Series("addr_norm", cols[4], dtype=pl.Utf8),
        pl.Series("addr_nums", cols[5], dtype=pl.Utf8),
        pl.Series("addr_missing", [len(a.strip()) == 0 for a in addrs], dtype=pl.Boolean),
    )


def _ids_match(
    raw: Path, normalized: Path, rows: int, chunk_rows: int, raw_offset: int = 0
) -> bool:
    raw_scan = pl.scan_parquet(raw).select("entity_id")
    norm_scan = pl.scan_parquet(normalized).select("entity_id")
    if norm_scan.select(pl.len()).collect().item() != rows:
        return False
    for start in range(0, rows, chunk_rows):
        size = min(chunk_rows, rows - start)
        raw_ids = raw_scan.slice(raw_offset + start, size).collect().get_column("entity_id")
        norm_ids = norm_scan.slice(start, size).collect().get_column("entity_id")
        if not raw_ids.equals(norm_ids):
            return False
    return True


def normalize_source(split: str, src: str, chunk_rows: int = 500_000):
    raw = cache_path(f"{split}_{src}.parquet")
    out = cache_path(f"{split}_{src}_norm.parquet")
    if out.exists() and _ids_match(raw, out, pl.scan_parquet(raw).select(pl.len()).collect().item(), chunk_rows):
        print(f"skip {out.name} (complete row count and entity_id order)", flush=True)
        return

    rows = pl.scan_parquet(raw).select(pl.len()).collect().item()
    chunk_dir = cache_path(f"norm_chunks_{split}_{src}")
    chunk_dir.mkdir(parents=True, exist_ok=True)
    chunk_files = []
    raw_scan = pl.scan_parquet(raw)
    with Pool(1) as pool:
        for start in range(0, rows, chunk_rows):
            size = min(chunk_rows, rows - start)
            chunk_path = chunk_dir / f"{start:09d}.parquet"
            raw_chunk = raw_scan.slice(start, size).collect(engine="streaming")
            if chunk_path.exists() and _ids_match(raw, chunk_path, size, chunk_rows, start):
                chunk_files.append(chunk_path)
                print(f"skip chunk {start:,}:{start + size:,} (complete)", flush=True)
                continue
            normalize_frame(raw_chunk, pool).write_parquet(chunk_path.with_suffix(".tmp"))
            chunk_path.with_suffix(".tmp").replace(chunk_path)
            chunk_files.append(chunk_path)
            print(f"{split}_{src}: normalized rows {start:,}:{start + size:,}", flush=True)

    tmp = out.with_suffix(".parquet.tmp")
    writer = None
    try:
        for chunk_path in chunk_files:
            table = pq.read_table(chunk_path)
            if writer is None:
                writer = pq.ParquetWriter(tmp, table.schema, compression="zstd")
            writer.write_table(table)
        if writer is None:
            raise RuntimeError(f"No normalized chunks produced for {split}_{src}")
    finally:
        if writer is not None:
            writer.close()
    if not _ids_match(raw, tmp, rows, chunk_rows):
        tmp.unlink(missing_ok=True)
        raise IOError(f"entity_id row count/order validation failed for {out.name}")
    os.replace(tmp, out)
    print(f"{split}_{src}: {rows:,} rows normalized in chunks", flush=True)


def main(splits=("train", "test"), sources=None, chunk_rows=500_000):
    if sources is not None:
        for split in splits:
            for src in sources:
                normalize_source(split, src, chunk_rows)
        return
    with Pool(N_PROC) as pool:
        for split in splits:
            for src in ("source1", "source2", "source3"):
                out = cache_path(f"{split}_{src}_norm.parquet")
                if out.exists():
                    print(f"skip {out.name} (exists)")
                    continue
                t = time.time()
                df = pl.read_parquet(cache_path(f"{split}_{src}.parquet"))
                normalize_frame(df, pool).write_parquet(out)
                print(f"{split}_{src}: {df.height:,} rows normalized in {time.time()-t:.0f}s", flush=True)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("splits", nargs="*", default=["train", "test"])
    parser.add_argument("--split", choices=("train", "test"))
    parser.add_argument("--source", choices=("source1", "source2", "source3"))
    parser.add_argument("--chunk-rows", type=int, default=500_000)
    args = parser.parse_args()
    if args.chunk_rows <= 0:
        parser.error("--chunk-rows must be positive")
    splits = (args.split,) if args.split else tuple(args.splits)
    sources = (args.source,) if args.source else None
    main(splits or ("train", "test"), sources, args.chunk_rows)
