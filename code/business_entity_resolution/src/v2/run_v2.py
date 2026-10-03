"""End-to-end v2 pipeline (every stage resumable; completed stages are skipped via run manifests).

Prerequisite (base pipeline in business_entity_resolution/src, also resumable):
  prepare_data -> build_normalized -> candidates (train, test) -> build_features (train, test)
Then this script:
  1. v2_prepare     split manifest, TRAIN-fold Indic dictionary, v2train/v2test tables, Indic re-retrieval
  2. base features  for v2train/v2test (reused by copy when candidates are byte-identical, else recomputed)
  3. make_qf        compact label/fold table
  4. featx          direct-evidence features (train, test)
  5. train_v2 C3    first-stage direct model (TRAIN fold only), scored on train and test
  6. stack_v2       C4 per-record second stage on C3, scored on train and test
  7. predict_v2     matching_results.tsv + candidate_pairs.tsv from the same run, validated
usage: python run_v2.py [threshold=0.7]"""
import filecmp
import shutil
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
PY = [sys.executable, "-u", "-W", "ignore"]
T = sys.argv[1] if len(sys.argv) > 1 else "0.7"


def run(*args, cwd=HERE):
    print(">>", " ".join(args), flush=True)
    subprocess.run(PY + list(args), cwd=cwd, check=True)


def main():
    from runlib import cache_path  # noqa: E402  (sets up paths)
    run("v2_prepare.py")
    src = HERE.parent  # business_entity_resolution/src
    for sp in ("train", "test"):
        dst = cache_path(f"v2{sp}_feats")
        same = all(filecmp.cmp(a, cache_path(f"v2{sp}_cands") / a.name, shallow=False)
                   for a in sorted(cache_path(f"{sp}_cands").glob("*.parquet")))
        if not dst.exists():
            if same and cache_path(f"{sp}_feats").exists():
                shutil.copytree(cache_path(f"{sp}_feats"), dst)  # identical inputs -> identical features
            else:
                run("-c", f"import build_features; build_features.build('v2{sp}')", cwd=src)
    run("make_qf.py")
    run("featx.py", "v2train")
    run("featx.py", "v2test")
    run("train_v2.py", "C3_direct_x_more_data", "v2train")
    run("train_v2.py", "C3_direct_x_more_data", "v2test")
    run("stack_v2.py", "C3_direct_x_more_data", "v2train", "v2test")
    run("predict_v2.py", "model_C4_stack_on_C3_direct_x_more_data", T)


if __name__ == "__main__":
    main()
