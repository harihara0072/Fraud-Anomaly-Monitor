import pandas as pd

from credit_line_review.export import build_scored_accounts, export_scored_accounts


def test_build_scored_accounts_flags_under_and_over_limited():
    table = pd.DataFrame({
        "id_card": [1, 2, 3],
        "client_id": [10, 20, 30],
        "credit_limit": [1000.0, 5000.0, 5000.0],
        "is_zero_limit": [False, False, False],
    })
    scored = build_scored_accounts(
        table,
        predicted_limit=[5000.0, 5000.0, 1000.0],
        cluster_labels=[0, 1, 0],
        flag_threshold=500.0,
    )
    assert scored.loc[0, "flag"] == "underlimited"
    assert scored.loc[1, "flag"] == "in_range"
    assert scored.loc[2, "flag"] == "overlimited"


def test_build_scored_accounts_flags_zero_limit_rows_as_closed_account():
    table = pd.DataFrame({
        "id_card": [1, 2],
        "client_id": [10, 20],
        "credit_limit": [0.0, 5000.0],
        "is_zero_limit": [True, False],
    })
    scored = build_scored_accounts(
        table,
        predicted_limit=[4000.0, 5000.0],
        cluster_labels=[0, 1],
        flag_threshold=500.0,
    )
    # Even though the residual for row 0 (0 - 4000 = -4000) would otherwise
    # read as heavily "underlimited", a closed/zero-limit account isn't a
    # real credit-line-review candidate - it gets its own category instead.
    assert scored.loc[0, "flag"] == "closed_account"
    assert scored.loc[1, "flag"] == "in_range"


def test_export_scored_accounts_writes_parquet(tmp_path):
    scored = pd.DataFrame({"id_card": [1], "flag": ["in_range"]})
    out_path = tmp_path / "scored.parquet"
    export_scored_accounts(scored, path=out_path)
    assert out_path.exists()
    assert pd.read_parquet(out_path).equals(scored)
