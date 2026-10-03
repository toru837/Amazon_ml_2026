"""Export the complete v4test inference candidate set to the C4 bundle cache."""
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

import polars as pl  # noqa: E402

from config import cache_path  # noqa: E402
from predict import write_grouped  # noqa: E402
from runlib import RUNS  # noqa: E402


def main():
    candidates_dir = cache_path("v4test_cands")
    files = sorted(candidates_dir.glob("*.parquet"))
    if not files:
        raise FileNotFoundError(f"no v4test candidate parquet files found in {candidates_dir}")

    s1_ids = pl.read_parquet(cache_path("test_source1.parquet"), columns=["entity_id"])["entity_id"]
    q_ids = pl.concat(
        [
            pl.read_parquet(cache_path(f"test_{source}.parquet"), columns=["entity_id"])
            for source in ("source2", "source3")
        ]
    )["entity_id"]
    candidates = pl.concat(
        [pl.read_parquet(path, columns=["s1_idx", "q_idx"]) for path in files]
    )

    dest = RUNS / "submission_C4_stack_on_C3_direct_x_more_data_v4test"
    dest.mkdir(parents=True, exist_ok=True)
    rows, nonempty = write_grouped(
        s1_ids, q_ids, candidates, "candidate_entity_ids", dest / "candidate_pairs.tsv"
    )
    if rows != len(s1_ids):
        raise RuntimeError(f"candidate export has {rows} rows; expected {len(s1_ids)}")
    print(f"wrote {rows:,} S1 rows; {nonempty:,} contain candidates")


if __name__ == "__main__":
    main()
