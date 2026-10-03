"""v2 model configurations (shared by train_v2.py, stack_v2.py, predict_v2.py)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # business_entity_resolution/src
from features import FEATURES  # noqa: E402

XCOLS = ["nm_wcov_q", "nm_wcov_s", "nm_qonly_idf_max", "nm_sonly_idf_max", "nm_shared_idf",
         "ad_wcov_q", "ad_qonly_idf_max", "ad_qonly_n_rare", "ad_shared_idf",
         "prem_eq", "prem_in_s", "s_prem_in_q", "q_nums_subset", "prem_close", "raw_name_eq", "raw_addr_eq"]
NO_POP = [f for f in FEATURES if f != "s1_rank1_deg"]  # s1_rank1_deg counts other records -> population dependent
CONFIGS = {
    "B0_baseline_refit": dict(feats=FEATURES, frac=0.10, leaves=255, extra=False),
    "C1_no_population_feat": dict(feats=NO_POP, frac=0.10, leaves=255, extra=False),
    "C2_direct_x": dict(feats=NO_POP + XCOLS, frac=0.10, leaves=255, extra=True),
    "C3_direct_x_more_data": dict(feats=NO_POP + XCOLS, frac=0.25, leaves=255, extra=True),
}
