import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler


def fit_kmeans_personas(X: pd.DataFrame, n_clusters: int, random_state: int = 42):
    """Course: Module 4 - Clustering. Segments cards into limit-policy personas -
    a genuinely different model family from the two regressions in models.py."""
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)
    labels = kmeans.fit_predict(X_scaled)
    score = silhouette_score(X_scaled, labels) if n_clusters > 1 else float("nan")
    return kmeans, labels, score


def project_to_2d(X: pd.DataFrame, random_state: int = 42) -> pd.DataFrame:
    """Course: Module 8 - Dimensionality Reduction. Not used to predict the
    target - only to visualize whether flagged accounts cluster sensibly."""
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    pca = PCA(n_components=2, random_state=random_state)
    coords = pca.fit_transform(X_scaled)
    return pd.DataFrame(coords, columns=["pc1", "pc2"], index=X.index)
