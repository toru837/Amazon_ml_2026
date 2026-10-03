"""Direct-evidence extra features (v2). Depend only on (record, candidate S1, S1-catalog statistics):
invariant to how many other S2/S3 records exist (duplication-invariant by construction).

Memory design: tokens are 64-bit hashes held in compact exploded tables (id, tok[, pos]); IDF is a
per-country catalog statistic over S1 only. Resumable per candidate file.
usage: python featx.py <split>   -> cache/<split>_featx/<same file names as cands>"""
import sys

import polars as pl

from runlib import Run, cache_path, mem, note

split = sys.argv[1]
CAP = 12.0
XCOLS = ["nm_wcov_q", "nm_wcov_s", "nm_qonly_idf_max", "nm_sonly_idf_max", "nm_shared_idf",
         "ad_wcov_q", "ad_qonly_idf_max", "ad_qonly_n_rare", "ad_shared_idf",
         "prem_eq", "prem_in_s", "s_prem_in_q", "q_nums_subset", "prem_close", "raw_name_eq", "raw_addr_eq"]
run = Run(f"featx_{split}", {"cap": CAP, "cols": XCOLS, "v": 2}, code_files=[__file__])
SEEDH = 7


def explode_tokens(df, idcol, col, kind):
    if kind == "name":
        tok = pl.col(col).str.split(" ")
    elif kind == "alpha":
        tok = pl.col(col).str.split(" ").list.eval(pl.element().filter(pl.element().str.contains(r"^[a-z]{2,}$")))
    else:  # numbers, keep position
        return (df.select(idcol, pl.col(col).str.extract_all(r"\d+").alias("t")).explode("t").drop_nulls()
                .with_columns(pl.int_range(pl.len()).over(idcol).cast(pl.UInt8).alias("pos"),
                              pl.col("t").cast(pl.Int64, strict=False).alias("v")).drop("t").drop_nulls())
    return (df.select(idcol, tok.alias("t")).explode("t").drop_nulls().filter(pl.col("t") != "")
            .select(idcol, pl.col("t").hash(SEEDH).alias("h")).unique())


if not run.done("tokens"):
    s1 = pl.read_parquet(cache_path(f"{split}_source1_norm.parquet"), columns=["country", "name_core", "addr_norm", "addr_nums"]).with_row_index("s1_idx")
    raw = pl.read_parquet(cache_path(f"{split}_source1.parquet"), columns=["business_name", "business_address"])
    s1h = s1.select("s1_idx", "country").with_columns(
        raw["business_name"].str.to_lowercase().str.strip_chars().hash(SEEDH).alias("rn"),
        raw["business_address"].str.to_lowercase().str.strip_chars().hash(SEEDH).alias("ra"))
    del raw
    run.write_parquet(s1h, "s1_meta.parquet")
    for kind, col, fname in (("name", "name_core", "s1_ntok"), ("alpha", "addr_norm", "s1_atok"), ("num", "addr_nums", "s1_nums")):
        run.write_parquet(explode_tokens(s1, "s1_idx", col, kind), f"{fname}.parquet")
    del s1
    # IDF per country over S1 (catalog statistic)
    N = s1h.group_by("country").agg(pl.len().alias("N"))
    for src, dst in (("s1_ntok", "idf_n"), ("s1_atok", "idf_a")):
        t = pl.read_parquet(run.path(f"{src}.parquet")).join(s1h.select("s1_idx", "country"), on="s1_idx")
        idf = t.group_by("country", "h").agg(pl.len().alias("df")).join(N, on="country").select(
            "country", "h", (pl.col("N").log() - (1 + pl.col("df")).log()).clip(0, CAP).alias("idf"))
        run.write_parquet(idf, f"{dst}.parquet")
    run.write_parquet(N.with_columns(pl.col("N").log().clip(0, CAP).alias("idf_unseen")).select("country", "idf_unseen"), "idf_unseen.parquet")
    print("s1 tokens", mem(), flush=True)
    q_parts = []
    for s in ("source2", "source3"):
        q_parts.append(pl.read_parquet(cache_path(f"{split}_{s}_norm.parquet"), columns=["name_core", "addr_norm", "addr_nums"]))
    q = pl.concat(q_parts).with_row_index("q_idx"); del q_parts
    for kind, col, fname in (("name", "name_core", "q_ntok"), ("alpha", "addr_norm", "q_atok"), ("num", "addr_nums", "q_nums")):
        pieces = [explode_tokens(q.slice(s0, 1_000_000).select("q_idx", col), "q_idx", col, kind) for s0 in range(0, q.height, 1_000_000)]
        run.write_parquet(pl.concat(pieces), f"{fname}.parquet")
        del pieces
        q = q.drop(col)
        print(" ", fname, mem(), flush=True)
    del q
    raw = pl.concat([pl.read_parquet(cache_path(f"{split}_{s}.parquet"), columns=["business_name", "business_address"]) for s in ("source2", "source3")])
    qh = raw.select(pl.col("business_name").str.to_lowercase().str.strip_chars().hash(SEEDH).alias("rn"),
                    pl.col("business_address").str.to_lowercase().str.strip_chars().hash(SEEDH).alias("ra")).with_row_index("q_idx")
    del raw
    run.write_parquet(qh, "q_meta.parquet")
    run.complete("tokens")
    print("q tokens", mem(), flush=True)

S1M = pl.read_parquet(run.path("s1_meta.parquet"))
QM = pl.read_parquet(run.path("q_meta.parquet"))
T = {k: pl.read_parquet(run.path(f"{k}.parquet")) for k in ("s1_ntok", "s1_atok", "s1_nums", "q_ntok", "q_atok", "q_nums")}
IDF_N, IDF_A = pl.read_parquet(run.path("idf_n.parquet")), pl.read_parquet(run.path("idf_a.parquet"))
UNSEEN = pl.read_parquet(run.path("idf_unseen.parquet"))
print("loaded token tables", mem(), flush=True)


def weighted(p, qt, st, idf, prefix):
    q = p.select("pid", "q_idx", "country").join(qt, on="q_idx").drop("q_idx")
    s = p.select("pid", "s1_idx", "country").join(st, on="s1_idx").drop("s1_idx")
    add = lambda d: d.join(idf, on=["country", "h"], how="left").join(UNSEEN, on="country").with_columns(
        pl.col("idf").fill_null(pl.col("idf_unseen"))).select("pid", "h", "idf")
    q, s = add(q), add(s)
    shared = q.join(s.select("pid", "h"), on=["pid", "h"], how="semi")
    qonly = q.join(s.select("pid", "h"), on=["pid", "h"], how="anti")
    sonly = s.join(q.select("pid", "h"), on=["pid", "h"], how="anti")
    a = (p.select("pid")
         .join(q.group_by("pid").agg(pl.col("idf").sum().alias("q_sum")), on="pid", how="left")
         .join(s.group_by("pid").agg(pl.col("idf").sum().alias("s_sum")), on="pid", how="left")
         .join(shared.group_by("pid").agg(pl.col("idf").sum().alias("sh")), on="pid", how="left")
         .join(qonly.group_by("pid").agg(pl.col("idf").max().alias("qo"), (pl.col("idf") >= 6).sum().alias("qr")), on="pid", how="left")
         .join(sonly.group_by("pid").agg(pl.col("idf").max().alias("so")), on="pid", how="left")
         .with_columns(pl.all().exclude("pid").fill_null(0.0)))
    return a.select("pid", (pl.col("sh") / pl.col("q_sum").clip(1e-6)).alias(f"{prefix}_wcov_q"),
                    (pl.col("sh") / pl.col("s_sum").clip(1e-6)).alias(f"{prefix}_wcov_s"),
                    pl.col("qo").alias(f"{prefix}_qonly_idf_max"), pl.col("so").alias(f"{prefix}_sonly_idf_max"),
                    pl.col("qr").alias(f"{prefix}_qonly_n_rare"), pl.col("sh").alias(f"{prefix}_shared_idf"))


def numbers(p):
    qn = p.select("pid", "q_idx").join(T["q_nums"], on="q_idx").drop("q_idx")
    sn = p.select("pid", "s1_idx").join(T["s1_nums"], on="s1_idx").drop("s1_idx")
    q1 = qn.filter(pl.col("pos") == 0).select("pid", pl.col("v").alias("q1"))
    s1 = sn.filter(pl.col("pos") == 0).select("pid", pl.col("v").alias("s1"))
    q1_in_s = q1.join(sn.select("pid", pl.col("v").alias("q1")), on=["pid", "q1"], how="semi").select("pid").unique().with_columns(pl.lit(True).alias("prem_in_s"))
    s1_in_q = s1.join(qn.select("pid", pl.col("v").alias("s1")), on=["pid", "s1"], how="semi").select("pid").unique().with_columns(pl.lit(True).alias("s_prem_in_q"))
    q_not_in_s = qn.join(sn.select("pid", "v"), on=["pid", "v"], how="anti").select("pid").unique().with_columns(pl.lit(True).alias("has_extra"))
    d = (p.select("pid").join(q1, on="pid", how="left").join(s1, on="pid", how="left")
         .join(q1_in_s, on="pid", how="left").join(s1_in_q, on="pid", how="left").join(q_not_in_s, on="pid", how="left"))
    qf, sf = pl.col("q1").cast(pl.Float64), pl.col("s1").cast(pl.Float64)
    return d.select("pid", (pl.col("q1") == pl.col("s1")).fill_null(False).alias("prem_eq"),
                    pl.col("prem_in_s").fill_null(False), pl.col("s_prem_in_q").fill_null(False),
                    pl.col("has_extra").is_null().alias("q_nums_subset"),
                    ((qf - sf).abs() / pl.max_horizontal(qf, sf, pl.lit(1.0))).fill_null(-1.0).alias("prem_close"))


def featurize(c: pl.DataFrame) -> pl.DataFrame:
    p = (c.select("q_idx", "s1_idx").with_row_index("pid")
         .join(S1M.rename({"rn": "s_rn", "ra": "s_ra"}), on="s1_idx", how="left").join(QM, on="q_idx", how="left"))
    raw = p.select("pid", (pl.col("rn") == pl.col("s_rn")).alias("raw_name_eq"), (pl.col("ra") == pl.col("s_ra")).alias("raw_addr_eq"))
    wn = weighted(p, T["q_ntok"], T["s1_ntok"], IDF_N, "nm")
    wa = weighted(p, T["q_atok"], T["s1_atok"], IDF_A, "ad").select("pid", "ad_wcov_q", "ad_qonly_idf_max", "ad_qonly_n_rare", "ad_shared_idf")
    out = p.select("pid", "q_idx", "s1_idx").join(wn, on="pid").join(wa, on="pid").join(numbers(p), on="pid").join(raw, on="pid")
    return out.sort("pid").select("q_idx", "s1_idx", *[pl.col(x).cast(pl.Float32) for x in XCOLS])


if len(sys.argv) > 2 and sys.argv[2] == "selftest":
    f0 = sorted(cache_path(f"{split}_cands").glob("*.parquet"))[0]
    c = pl.read_parquet(f0, columns=["q_idx", "s1_idx"], memory_map=False).head(50_000)
    x = featurize(c)
    assert x.height == c.height and x.select(pl.all().null_count()).sum_horizontal().item() == 0
    print(x.drop("q_idx", "s1_idx").mean().transpose(include_header=True))
    raise SystemExit(0)
in_dir, out_dir = cache_path(f"{split}_cands"), cache_path(f"{split}_featx")
out_dir.mkdir(exist_ok=True)
files = sorted(in_dir.glob("*.parquet"))
for i, f in enumerate(files):
    stage = f"file_{f.name}"
    if run.done(stage) and (out_dir / f.name).exists():
        continue
    c = pl.read_parquet(f, columns=["q_idx", "s1_idx"], memory_map=False)
    x = pl.concat([featurize(c.slice(s, 400_000)) for s in range(0, c.height, 400_000)])
    assert x.height == c.height, (x.height, c.height)
    tmp = out_dir / (f.name + ".tmp")
    x.write_parquet(tmp)
    tmp.replace(out_dir / f.name)
    run.complete(stage, rows=x.height)
    print(f"[{i+1}/{len(files)}] {f.name} {x.height:,}", mem(), flush=True)
note(f"featx {split}: {len(files)} files done")
run.release()
