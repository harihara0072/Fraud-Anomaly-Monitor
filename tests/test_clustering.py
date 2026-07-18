import pandas as pd

from credit_line_review.clustering import fit_kmeans_personas, project_to_2d


def test_fit_kmeans_personas_separates_two_obvious_clusters():
    X = pd.DataFrame({
        "a": [0, 0, 0, 10, 10, 10],
        "b": [0, 1, -1, 10, 11, 9],
    })
    _, labels, score = fit_kmeans_personas(X, n_clusters=2)
    assert len(set(labels[:3])) == 1
    assert len(set(labels[3:])) == 1
    assert labels[0] != labels[3]
    assert score > 0.5


def test_project_to_2d_returns_two_columns_same_row_count():
    X = pd.DataFrame({"a": [1, 2, 3], "b": [3, 2, 1], "c": [5, 5, 5]})
    coords = project_to_2d(X)
    assert list(coords.columns) == ["pc1", "pc2"]
    assert len(coords) == 3
