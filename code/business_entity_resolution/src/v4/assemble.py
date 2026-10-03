"""Assemble a candidate submission on the France-locale split (v4test).
US/India: C4 (threshold 0.7), unchanged. France: source model + threshold per variant.
usage: python assemble.py <variant> <france_source: c4|frp_<name>> <france_threshold>
Writes output/v4_<variant>/{matching_results.tsv, candidate_pairs.tsv (hard link), bundle.json, validation.log}."""
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "v2"))
import polars as pl  # noqa: E402

from runlib import ROOT, RUNS, cache_path, mem, note, sha  # noqa: E402
from predict import write_grouped  # noqa: E402

variant, fsrc, tfr = sys.argv[1], sys.argv[2], float(sys.argv[3])
C4 = RUNS / "model_C4_stack_on_C3_direct_x_more_data"
s1c = pl.read_parquet(cache_path("test_source1.parquet"), columns=["country"]).with_row_index("s1_idx")
c4 = pl.read_parquet(C4 / "top3_v4test.parquet").select("q_idx", "s1_idx", "p1").join(s1c, on="s1_idx")
USIN = sys.argv[4] if len(sys.argv) > 4 else "c4"
if USIN == "c4":
    usin = c4.filter((pl.col("country") != "France") & (pl.col("p1") >= 0.7)).select("s1_idx", "q_idx")
else:  # frozen v3 role model on v4test (US/India rows identical to v3 on v2test), its frozen threshold 0.8
    v3 = pl.read_parquet(cache_path("runs/v3/roles/pred_v4test.parquet")).join(s1c, on="s1_idx")
    usin = v3.filter((pl.col("country") != "France") & (pl.col("p") >= 0.8)).select("s1_idx", "q_idx")
if fsrc == "c4":
    fr = c4.filter((pl.col("country") == "France") & (pl.col("p1") >= tfr)).select("s1_idx", "q_idx")
    n_fr_records = c4.filter(pl.col("country") == "France").height
else:
    t = pl.read_parquet(RUNS / fsrc / "top1_france.parquet")
    n_fr_records = t.height
    assert t.join(s1c, on="s1_idx")["country"].unique().to_list() == ["France"]
    fr = t.filter(pl.col("p1") >= tfr).select("s1_idx", "q_idx")
assert n_fr_records == 1_434_993, n_fr_records
matches = pl.concat([usin, fr])
assert matches["q_idx"].n_unique() == matches.height
found = 0
for f in sorted(cache_path("v4test_cands").glob("*.parquet")):
    found += matches.join(pl.read_parquet(f, columns=["s1_idx", "q_idx"], memory_map=False), on=["s1_idx", "q_idx"], how="semi").height
assert found == matches.height, "matches not subset of candidates"
dest = Path(os.environ.get("ER_FINAL_OUTPUT_DIR", ROOT.parent / "output")).resolve()
dest.mkdir(exist_ok=True)
s1_ids = pl.read_parquet(cache_path("test_source1.parquet"), columns=["entity_id"])["entity_id"]
q_ids = pl.concat([pl.read_parquet(cache_path(f"test_{s}.parquet"), columns=["entity_id"]) for s in ("source2", "source3")])["entity_id"]
tmp = dest / "matching_results.tmp"
n, nm = write_grouped(s1_ids, q_ids, matches, "matched_entity_ids", tmp)
assert n == len(s1_ids)
os.replace(tmp, dest / "matching_results.tsv")
cand = dest / "candidate_pairs.tsv"
candidate_source = RUNS / "submission_C4_stack_on_C3_direct_x_more_data_v4test" / "candidate_pairs.tsv"
if not candidate_source.exists():
    raise FileNotFoundError(f"missing v4test candidate export: {candidate_source}")
shutil.copy2(candidate_source, cand)
data_dir = Path(os.environ.get("ER_DATA_DIR", ROOT / "dataset")).resolve()
validator = HERE.parent / "utils" / "validate_submission.py"
r = subprocess.run([sys.executable, str(validator), "--matching", str(dest / "matching_results.tsv"),
                    "--candidate", str(dest / "__not_loaded__.tsv"), "--test-dir", str(data_dir / "test")],
                   text=True, capture_output=True, encoding="utf-8", errors="replace", env={**os.environ, "PYTHONIOENCODING": "utf-8"})
candidate_source.parent.joinpath("validation.log").write_text(r.stdout + r.stderr, encoding="utf-8")
assert r.returncode == 0, "validator failed"
b = {"variant": variant, "us_india": "C4 t=0.7 (unchanged)" if USIN == "c4" else "v3 roles t=0.8 (unchanged)", "france_source": fsrc, "france_threshold": tfr,
     "matched_records": matches.height, "france_matched": fr.height, "france_accept": fr.height / n_fr_records,
     "s1_with_matches": nm, "matching_sha256": sha(dest / "matching_results.tsv"), "candidate_sha256": sha(cand),
     "validator": "PASS", "streamed_candidate_subset": "PASS", "leaderboard_score": None}
candidate_source.parent.joinpath("bundle.json").write_text(json.dumps(b, indent=1))
note(f"ASSEMBLED v4_{variant}: France {fsrc} t={tfr} accept {b['france_accept']:.4f}; total matches {matches.height:,}; validator PASS")
print(json.dumps(b, indent=1), mem())
