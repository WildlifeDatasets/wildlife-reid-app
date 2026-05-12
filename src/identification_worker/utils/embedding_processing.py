from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

import numpy as np
import pandas as pd
from sklearn.manifold import TSNE
from logging import getLogger
logger = getLogger(__name__)



@dataclass
class OutlierConfig:
    method: str = "zscore"  # zscore | percentile | iqr
    threshold: float = 2.5


class EmbeddingProcessing:
    """Analyze embedding quality, similarity, and potential label noise."""

    def __init__(
            self,
            embeddings: np.ndarray,
            metadata: pd.DataFrame,
            label_col: Optional[str] = None,
            normalize: bool = True,
    ) -> None:

        embeddings = np.asarray(embeddings, dtype=np.float32)
        self.embeddings = self._l2_normalize(embeddings) if normalize else embeddings
        self.n_samples, self.n_features = self.embeddings.shape
        self.label_col = label_col
        logger.debug("l2 norm done")
        self.metadata = metadata.copy().reset_index(drop=True)

        self._similarity_cache: Optional[np.ndarray] = None
        logger.debug("init done")

    @staticmethod
    def _l2_normalize(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
        """L2 normalize the embeddings."""
        n = np.linalg.norm(x, axis=1, keepdims=True)
        return x / np.maximum(n, eps)

    def _get_labels(self) -> np.ndarray:
        """Get the labels from the metadata."""
        if self.label_col and self.label_col in self.metadata.columns:
            lbl = self.metadata[self.label_col].to_numpy(dtype=object)
            if len(lbl) != self.n_samples:
                raise ValueError("metadata label column length mismatch")
            return lbl

        raise ValueError("Provide labels or set label_col in metadata")

    def similarity_matrix(self, force_recompute: bool = False) -> np.ndarray:
        """Get the similarity matrix between the embeddings."""
        if self._similarity_cache is None or force_recompute:
            self._similarity_cache = self.embeddings @ self.embeddings.T
        return self._similarity_cache

    def pairwise_similarity(self, idx_a: int, idx_b: int) -> float:
        return float(np.dot(self.embeddings[idx_a], self.embeddings[idx_b]))

    def reduce_embeddings(
        self,
        method: str = "tsne",
        n_components: int = 2,
        random_state: int = 42,
        **kwargs: Any,
    ) -> np.ndarray:
        """Project embeddings with t-SNE or UMAP and return reduced features."""
        method_norm = method.lower().strip()

        if n_components < 1:
            raise ValueError("n_components must be >= 1")

        if method_norm == "tsne":
            tsne_params: dict[str, Any] = {
                "n_components": n_components,
                "random_state": random_state,
                "init": "pca",
                "learning_rate": "auto",
            }
            tsne_params.update(kwargs)
            reducer = TSNE(**tsne_params)
            return reducer.fit_transform(self.embeddings)

        if method_norm == "umap":
            try:
                import umap
            except ImportError as exc:
                raise ImportError(
                    "UMAP is not installed. Install it with: pip install umap-learn"
                ) from exc

            umap_params: dict[str, Any] = {
                "n_components": n_components,
                "random_state": random_state,
            }
            umap_params.update(kwargs)
            reducer = umap.UMAP(**umap_params)
            return reducer.fit_transform(self.embeddings)

        raise ValueError("method must be one of: 'tsne' or 'umap'")

    def get_close_embeddings(self, embeddings, thr=0.9):
        """Get the closest embeddings to the given embeddings."""
        cos_sim_matrix = self.cosine_similarity()
        np.fill_diagonal(cos_sim_matrix, -np.inf)

        # Find the indices of the two closest embeddings (maximum cosine similarity)
        max_sim_idx = np.unravel_index(np.argmax(cos_sim_matrix), cos_sim_matrix.shape)
        idx1, idx2 = max_sim_idx
        max_sim_value = cos_sim_matrix[idx1, idx2]
        similarity_threshold = thr * max_sim_value

        # Find embeddings whose similarity to BOTH idx1 and idx2 is above the threshold
        sim_to_1 = cos_sim_matrix[idx1]
        sim_to_2 = cos_sim_matrix[idx2]
        high_sim_indices = [i for i in range(len(embeddings)) if
                            (sim_to_1[i] >= similarity_threshold and sim_to_2[i] >= similarity_threshold)]

        # Always include the first two most similar (idx1, idx2)
        indices_to_select = set(high_sim_indices)
        indices_to_select.update([idx1, idx2])
        selected_embeddings = embeddings[list(indices_to_select)]
        return selected_embeddings

    def cluster_centers(self) -> pd.DataFrame:
        """Get the cluster centers for the embeddings."""
        lbl = self._get_labels()
        out = []
        outlier_idx: set[int] = set()

        s = pd.Series(lbl).dropna().unique()
        for label in s:
            members = np.where(lbl == label)[0]
            if len(members) == 0:
                continue
            center = self.embeddings[members].mean(axis=0)
            center = center / np.maximum(np.linalg.norm(center), 1e-12)
            out.append(
                {
                    "label": label,
                    "size": int(len(members)),
                    "center": center,
                    "members": members,
                }
            )
        return pd.DataFrame(out).sort_values("size", ascending=False).reset_index(drop=True)

    def distances_to_cluster_center(self) -> pd.DataFrame:
        """Get the distances to the cluster centers for the embeddings."""
        lbl = self._get_labels()
        centers = self.cluster_centers()
        center_map = {r.label: r.center for r in centers.itertuples(index=False)}
        rows = []
        for i in range(self.n_samples):
            c = center_map.get(lbl[i])
            if c is None:
                continue
            sim = float(np.dot(self.embeddings[i], c))
            rows.append(
                {
                    "idx": i,
                    "label": lbl[i],
                    "center_similarity": sim,
                    "distance_to_center": 1.0 - sim,
                }
            )
        d = pd.DataFrame(rows)
        if not d.empty:
            # Rank samples per label; rank 1 is the farthest from its own cluster center.
            d["rank_within_label"] = (
                d.groupby("label")["distance_to_center"].rank(method="dense", ascending=False)
            )
        return d.sort_values("distance_to_center", ascending=False).reset_index(drop=True)

    def filter_outliers(
            self,
            config: Optional[OutlierConfig] = None,
            min_cluster_size: int = 3,
    ) -> pd.DataFrame:
        """Flag per-label embedding outliers by distance to class center.

        Expected `config.threshold` ranges:
        - zscore: typically 2.0-3.5 (higher means fewer outliers)
        - percentile: 80-99.9 (interpreted as percentile in [0, 100])
        - iqr: typically 1.5-3.0 for Tukey-style upper fence
        """
        if config is None:
            config = OutlierConfig()
        d = self.distances_to_cluster_center()
        if d.empty:
            return d

        flags = np.zeros(len(d), dtype=bool)
        for _, grp in d.groupby("label"):
            if len(grp) < min_cluster_size:
                continue
            vals = grp["distance_to_center"].to_numpy()

            # Z-score: mark samples whose distance is > threshold standard deviations above class mean.
            if config.method == "zscore":
                mu, sigma = vals.mean(), vals.std(ddof=0)
                mark = np.zeros_like(vals, dtype=bool) if sigma < 1e-12 else ((vals - mu) / sigma) > config.threshold
            # Percentile: mark samples whose distance is above the chosen percentile cutoff in the class.
            elif config.method == "percentile":
                q = np.percentile(vals, config.threshold)
                mark = vals > q
            # IQR: mark samples above Tukey's upper fence (Q3 + threshold * IQR), robust to skew/outliers.
            elif config.method == "iqr":
                q1, q3 = np.percentile(vals, [25, 75])
                iqr = q3 - q1
                mark = vals > (q3 + config.threshold * iqr)
            else:
                raise ValueError("config.method must be one of: zscore, percentile, iqr")

            flags[d.index.isin(grp.index)] = mark

        out = d.copy()
        out["is_outlier"] = flags
        return out.sort_values(["is_outlier", "distance_to_center"], ascending=[False, False]).reset_index(drop=True)

    def most_similar_clusters(self, top_k: int = 10, min_cluster_size: int = 2) -> pd.DataFrame:
        """Get the most similar clusters for the embeddings."""
        c = self.cluster_centers()
        c = c[c["size"] >= min_cluster_size].reset_index(drop=True)
        if len(c) < 2:
            return pd.DataFrame(columns=["label_a", "label_b", "similarity", "size_a", "size_b"])

        centers = np.stack(c["center"].to_numpy())
        sims = centers @ centers.T
        rows = []
        for i in range(len(c)):
            for j in range(i + 1, len(c)):
                rows.append(
                    {
                        "label_a": c.loc[i, "label"],
                        "label_b": c.loc[j, "label"],
                        "similarity": float(sims[i, j]),
                        "size_a": int(c.loc[i, "size"]),
                        "size_b": int(c.loc[j, "size"]),
                    }
                )
        return pd.DataFrame(rows).sort_values("similarity", ascending=False).head(top_k).reset_index(drop=True)

    def most_similar_embeddings(self, top_k: int = 20, cross_label_only: bool = False, ) -> pd.DataFrame:
        """Get the most similar embeddings for the embeddings."""
        sims = self.similarity_matrix()
        lbl = self._get_labels()

        n = self.n_samples
        iu, ju = np.triu_indices(n, k=1)
        vals = sims[iu, ju]

        if cross_label_only and lbl is not None:
            mask = lbl[iu] != lbl[ju]
            iu = iu[mask]
            ju = ju[mask]
            vals = vals[mask]

        if vals.size == 0:
            return pd.DataFrame(columns=["idx_a", "idx_b", "similarity", "label_a", "label_b"])

        if top_k >= vals.size:
            order = np.argsort(-vals)
        else:
            # Select top-k indices without sorting the full array.
            part = np.argpartition(-vals, top_k - 1)[:top_k]
            order = part[np.argsort(-vals[part])]

        sel_a = iu[order]
        sel_b = ju[order]
        sel_vals = vals[order].astype(float, copy=False)

        out = pd.DataFrame(
            {
                "idx_a": sel_a.astype(int, copy=False),
                "idx_b": sel_b.astype(int, copy=False),
                "similarity": sel_vals,
                "label_a": None if lbl is None else lbl[sel_a],
                "label_b": None if lbl is None else lbl[sel_b],
            }
        )
        return out.reset_index(drop=True)

    def nearest_neighbors(self, query_idx: int, top_k: int = 10, exclude_self: bool = True) -> pd.DataFrame:
        """Get the nearest neighbors for the given embedding."""
        sims = self.similarity_matrix()[query_idx].copy()
        if exclude_self:
            sims[query_idx] = -np.inf

        order = np.argsort(-sims)[:top_k]
        out = pd.DataFrame({"neighbor_idx": order, "similarity": sims[order]})
        if self.label_col and self.label_col in self.metadata.columns:
            out["neighbor_label"] = self.metadata.loc[order, self.label_col].to_numpy()
        return out.reset_index(drop=True)

    def likely_mislabeled(self, margin: float = 0.05, min_cluster_size: int = 3) -> pd.DataFrame:
        """Get the likely mislabeled embeddings."""
        lbl = self._get_labels()
        c = self.cluster_centers()
        c = c[c["size"] >= min_cluster_size].reset_index(drop=True)
        if len(c) < 2:
            return pd.DataFrame(
                columns=[
                    "idx",
                    "label",
                    "own_similarity",
                    "best_other_label",
                    "best_other_similarity",
                    "delta",
                    "is_suspect",
                ]
            )

        centers = np.stack(c["center"].to_numpy())
        center_labels = c["label"].to_numpy()
        label_to_center = {lab: centers[i] for i, lab in enumerate(center_labels)}

        rows = []
        for i in range(self.n_samples):
            sample_label = lbl[i]
            own_center = label_to_center.get(sample_label)

            if own_center is None:
                continue

            # Compute cosine similarity between the sample and its own cluster center.
            own_sim = float(np.dot(self.embeddings[i], own_center))
            sims = centers @ self.embeddings[i]

            # Set the similarity to the own cluster to -inf to avoid selecting it as the best other.
            own_idx = np.where(center_labels == sample_label)[0]
            if len(own_idx) > 0:
                sims[own_idx[0]] = -np.inf

            # Find the index of the embedding with the highest similarity to the sample.
            j = int(np.argmax(sims))
            best_other = float(sims[j])
            delta = best_other - own_sim
            rows.append(
                {
                    "idx": i,
                    "label": sample_label,
                    "own_similarity": own_sim,
                    "best_other_label": center_labels[j],
                    "best_other_similarity": best_other,
                    "delta": delta,
                    "is_suspect": bool(delta > margin),
                }
            )
        return pd.DataFrame(rows).sort_values(["is_suspect", "delta"], ascending=[False, False]).reset_index(drop=True)

    def best_other_member_index(self, query_idx: int, best_other_label: str) -> Optional[int]:
        """Return closest metadata index from candidate's best other cluster.

        Args:
        - query_idx: query embedding index
        - best_other_label: label of the nearest alternative cluster
        """
        
        lbl = self._get_labels()
        other_members = np.where(lbl == best_other_label)[0]
        if len(other_members) == 0:
            return None

        member_sims = self.embeddings[other_members] @ self.embeddings[query_idx]
        return int(other_members[int(np.argmax(member_sims))])
