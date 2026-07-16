import pandas as pd

from credit_line_review.config import SCORED_ACCOUNTS_PARQUET


def build_scored_accounts(
    table: pd.DataFrame,
    predicted_limit,
    cluster_labels,
    flag_threshold: float,
) -> pd.DataFrame:
    scored = table[["id_card", "client_id", "credit_limit", "is_zero_limit"]].copy()
    scored["predicted_limit"] = predicted_limit
    scored["residual"] = scored["credit_limit"] - scored["predicted_limit"]
    scored["cluster"] = cluster_labels
    # Course: Module 3 - Classification. There's no ground-truth "true
    # miscalibration" label, but thresholding the residual still turns this
    # into a binary flag/don't-flag decision, so Module 3's precision/recall
    # framing still applies to it (see the proposal doc §6 for how, absent
    # a real label, this gets evaluated as a business-plausibility check instead).
    scored["flag"] = "in_range"
    scored.loc[scored["residual"] < -flag_threshold, "flag"] = "underlimited"
    scored.loc[scored["residual"] > flag_threshold, "flag"] = "overlimited"
    # Zero-limit (closed/frozen) accounts aren't a real credit-line-review
    # candidate - their residual would otherwise read as extreme
    # "underlimited", drowning out the genuine cases. Give them their own
    # category instead (see Task 2 EDA finding + Task 4's is_zero_limit flag).
    scored.loc[scored["is_zero_limit"], "flag"] = "closed_account"
    return scored


def export_scored_accounts(scored: pd.DataFrame, path=SCORED_ACCOUNTS_PARQUET) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    scored.to_parquet(path, index=False)
