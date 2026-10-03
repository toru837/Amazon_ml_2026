"""v4test = v2test with French-locale normalization for France records only.
US/India artifacts are hard-linked (identical content, no extra disk); France is re-normalized, re-retrieved
and re-featurized. Resumable via the v2 run manifest."""
import os
import shutil
import sys
from multiprocessing import Pool
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "v2"))
sys.path.insert(0, str(HERE))
import polars as pl  # noqa: E402

from runlib import Run, cache_path, mem, note  # noqa: E402

SRC, DST = "v2test", "v4test"


def link(src: Path, dst: Path):
    if dst.exists():
        return
    try:
        os.link(src, dst)
    except OSError:
        shutil.copy2(src, dst)


def _fr_batch(args):
    from fr_locale import normalize_address_fr, normalize_name_fr
    names, addrs = args
    out = []
    for n, a in zip(names, addrs):
        nn, nc, dom, ind = normalize_name_fr(n)
        an, nums = normalize_address_fr(a)
        out.append((nn, nc, dom, ind, an, nums))
    return out


def renorm(table: str, pool):
    """re-normalize France rows of <DST>_<table>_norm.parquet from the raw test table"""
    raw = pl.read_parquet(cache_path(f"test_{table}.parquet"))
    norm = pl.read_parquet(cache_path(f"{SRC}_{table}_norm.parquet"))
    assert raw.height == norm.height and (raw["entity_id"] == norm["entity_id"]).all()
    idx = (raw["country"] == "France").arg_true().to_numpy()
    names, addrs = raw["business_name"].gather(idx).to_list(), raw["business_address"].gather(idx).to_list()
    B = 20000
    res = [r for part in pool.imap(_fr_batch, [(names[i:i + B], addrs[i:i + B]) for i in range(0, len(names), B)]) for r in part]
    cols = list(zip(*res))
    upd = {"name_norm": cols[0], "name_core": cols[1], "is_domain": cols[2], "has_indic": cols[3], "addr_norm": cols[4], "addr_nums": cols[5]}
    out = norm
    for c, vals in upd.items():
        s = out[c].to_list() if c in out.columns else None
        if s is None:
            continue
        for k, i in enumerate(idx):
            s[i] = vals[k]
        out = out.with_columns(pl.Series(c, s, dtype=out[c].dtype))
    if "name_norm_raw" in out.columns:  # dictionary step keeps the pre-dictionary name; France has no Indic script
        s = out["name_norm_raw"].to_list()
        for k, i in enumerate(idx):
            s[i] = cols[0][k]
        out = out.with_columns(pl.Series("name_norm_raw", s))
    out.write_parquet(cache_path(f"{DST}_{table}_norm.parquet"))
    changed = sum(1 for k, i in enumerate(idx) if norm["addr_norm"][int(i)] != cols[4][k] or norm["name_core"][int(i)] != cols[1][k])
    note(f"v4 renorm {table}: {len(idx):,} France rows re-normalized, {changed:,} changed")


def main():
    run = Run("v4_data", {"base": SRC, "locale": "fr v1"}, code_files=[__file__, str(HERE / "fr_locale.py")])
    if not run.done("norm"):
        with Pool(max(1, (os.cpu_count() or 2) - 2)) as pool:
            for t in ("source1", "source2", "source3"):
                renorm(t, pool)
        for t in ("source1", "source2", "source3"):
            link(cache_path(f"test_{t}.parquet"), cache_path(f"{DST}_{t}.parquet"))
        run.complete("norm")
    if not run.done("links"):
        for kind in ("cands", "feats"):
            d = cache_path(f"{DST}_{kind}"); d.mkdir(exist_ok=True)
            for f in cache_path(f"{SRC}_{kind}").iterdir():
                if f.name.startswith("France"):
                    continue  # France is regenerated
                link(f, d / f.name)
        run.complete("links")
    print("v4_prepare done", mem())
    run.release()


if __name__ == "__main__":
    main()
