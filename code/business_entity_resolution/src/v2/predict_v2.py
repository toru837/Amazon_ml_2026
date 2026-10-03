"""Write a coherent submission bundle from ONE verified run into cache/runs/submission_<model>/.
usage: python predict_v2.py <model_run_dir_name> <threshold>   e.g. model_C4_stack_on_C3_direct_x_more_data 0.7
 * scores v2test with the model (train_v2.py scoring stage), keeps each record's best S1 (one S1 per record)
 * matching_results.tsv  = records whose best probability >= threshold, grouped by S1
 * candidate_pairs.tsv   = every (S1, record) pair the model scored (all retrieved candidates)
 * checks: matches subset of candidates, one S1 per record, every test S1 present; official validator"""
import json
import subprocess
import sys

import polars as pl

from runlib import ROOT, RUNS, Run, cache_path, mem, note, sha
from predict import write_grouped
from train import id_tables

model_run, T = sys.argv[1], float(sys.argv[2])
name = model_run.removeprefix("model_")
if not (RUNS / model_run / "top3_v2test.parquet").exists():
    raise SystemExit(f"score v2test first (train_v2.py <cfg> v2test / stack_v2.py <first> v2test) for {model_run}")
run = Run(f"submission_{name}", {"model": model_run, "threshold": T})
top = pl.read_parquet(RUNS / model_run / "top3_v2test.parquet")
s1, q = id_tables("test")
s1_ids, q_ids = s1["entity_id"], q["entity_id"]
del s1, q
assert top["q_idx"].n_unique() == top.height, "one row per record expected"
matches = top.filter(pl.col("p1") >= T).select("s1_idx", "q_idx")
found = 0
for f in sorted(cache_path("v2test_cands").glob("*.parquet")):  # streamed subset check (bounded memory)
    found += matches.join(pl.read_parquet(f, columns=["s1_idx", "q_idx"], memory_map=False), on=["s1_idx", "q_idx"], how="semi").height
assert found == matches.height, f"{matches.height - found} matches are not candidates"
cand = pl.concat([pl.read_parquet(f, columns=["s1_idx", "q_idx"], memory_map=False) for f in sorted(cache_path("v2test_cands").glob("*.parquet"))])
n, n_c = write_grouped(s1_ids, q_ids, cand, "candidate_entity_ids", run.path("candidate_pairs.tsv"))
del cand
n2, n_m = write_grouped(s1_ids, q_ids, matches, "matched_entity_ids", run.path("matching_results.tsv"))
assert n == n2 == len(s1_ids)
res = {"model": model_run, "threshold": T, "matched_records": matches.height, "accept_rate": matches.height / top.height,
       "s1_with_matches": n_m, "s1_rows": n, "s1_with_candidates": n_c}
res["validator"] = "run separately: utils/validate_submission.py needs ~10 GB RAM (see PROGRESS.md)"
res["sha_matching"] = sha(run.path("matching_results.tsv"))
res["sha_candidates"] = sha(run.path("candidate_pairs.tsv"))
run.path("bundle.json").write_text(json.dumps(res, indent=1))
run.complete("bundle", files=["matching_results.tsv", "candidate_pairs.tsv", "bundle.json"])
note(f"submission bundle {name} t={T}: {matches.height:,} matches (accept {res['accept_rate']:.4f}), validator pending (separate step)")
run.release()
print(json.dumps(res, indent=1), mem())
