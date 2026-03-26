"""
UnknownFaceClusterer
--------------------
Collects face embeddings that didn't match any registered person,
clusters them with DBSCAN, and persists cluster sightings to the DB.

Cluster labels are stable letter-based IDs: "Unknown Person A", "B", …
New embeddings are assigned to an existing cluster if their distance to
the cluster centroid is within MERGE_THRESHOLD, otherwise a new cluster
is created.  DBSCAN is re-run periodically to merge drifted clusters.
"""

import threading
import time
import string
import numpy as np
from sklearn.cluster import DBSCAN


# Distance threshold to assign a new embedding to an existing cluster
MERGE_THRESHOLD = 0.75
# How often (seconds) to re-run full DBSCAN to merge stale clusters
RECLUSTER_INTERVAL = 60


def _label_for(index: int) -> str:
    """0→A, 1→B … 25→Z, 26→AA, 27→AB …"""
    letters = string.ascii_uppercase
    result = ""
    index += 1
    while index > 0:
        index, rem = divmod(index - 1, 26)
        result = letters[rem] + result
    return result


class UnknownFaceClusterer:
    def __init__(self, db_manager):
        self.db = db_manager
        self._lock = threading.Lock()

        # cluster_id (int) → {"label": str, "centroid": np.ndarray,
        #                      "embeddings": list[np.ndarray],
        #                      "snap_path": str | None}
        self._clusters: dict = {}
        self._next_id: int = 0

        # Load persisted clusters from DB on startup
        self._load_from_db()

        # Background re-clustering thread
        self._thread = threading.Thread(target=self._recluster_loop, daemon=True)
        self._thread.start()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def add_sighting(self, embedding: np.ndarray, camera_id: str,
                     snap_path: str) -> str:
        """
        Add an unrecognised face embedding.
        Returns the cluster label ("Unknown Person A", …).
        """
        with self._lock:
            cluster_id = self._assign(embedding)
            cluster = self._clusters[cluster_id]
            label = f"Unknown Person {cluster['label']}"

            # Update centroid (running mean)
            cluster["embeddings"].append(embedding)
            cluster["centroid"] = np.mean(cluster["embeddings"], axis=0)
            if cluster["snap_path"] is None:
                cluster["snap_path"] = snap_path

        # Persist sighting
        self.db.add_unknown_sighting(cluster_id, camera_id, snap_path)
        return label

    def get_clusters(self) -> list[dict]:
        """Return all clusters with their sighting history."""
        rows = self.db.get_unknown_clusters_with_sightings()
        # rows: list of (cluster_id, label, snap_path, camera_id, timestamp)
        clusters: dict[int, dict] = {}
        for cid, label, snap, cam, ts in rows:
            if cid not in clusters:
                clusters[cid] = {
                    "cluster_id": cid,
                    "label": f"Unknown Person {label}",
                    "snap_path": snap,
                    "sightings": []
                }
            if cam:
                clusters[cid]["sightings"].append({"camera_id": cam, "timestamp": ts, "snap_path": snap})
        return list(clusters.values())

    def get_cluster_embedding(self, cluster_id: int) -> np.ndarray | None:
        with self._lock:
            c = self._clusters.get(cluster_id)
            return c["centroid"].copy() if c is not None else None

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _assign(self, embedding: np.ndarray) -> int:
        """Find nearest cluster or create a new one. Must hold _lock."""
        best_id, best_dist = None, float("inf")
        for cid, c in self._clusters.items():
            d = float(np.linalg.norm(c["centroid"] - embedding))
            if d < best_dist:
                best_dist = d
                best_id = cid

        if best_id is not None and best_dist < MERGE_THRESHOLD:
            return best_id

        # New cluster
        cid = self._next_id
        self._next_id += 1
        label = _label_for(cid)
        self._clusters[cid] = {
            "label": label,
            "centroid": embedding.copy(),
            "embeddings": [],
            "snap_path": None
        }
        self.db.create_unknown_cluster(cid, label)
        return cid

    def _recluster_loop(self):
        while True:
            time.sleep(RECLUSTER_INTERVAL)
            try:
                self._recluster()
            except Exception as e:
                print(f"[Clusterer] recluster error: {e}")

    def _recluster(self):
        """Re-run DBSCAN over all stored embeddings to merge drifted clusters."""
        with self._lock:
            all_embeddings = []
            all_cids = []
            for cid, c in self._clusters.items():
                for emb in c["embeddings"]:
                    all_embeddings.append(emb)
                    all_cids.append(cid)

        if len(all_embeddings) < 2:
            return

        X = np.array(all_embeddings)
        labels = DBSCAN(eps=MERGE_THRESHOLD, min_samples=2, metric="euclidean").fit_predict(X)

        # Build new cluster map from DBSCAN output
        new_groups: dict[int, list] = {}
        for idx, lbl in enumerate(labels):
            if lbl == -1:
                continue  # noise — keep in original cluster
            new_groups.setdefault(lbl, []).append(all_embeddings[idx])

        with self._lock:
            # Rebuild clusters from DBSCAN groups, preserving lowest original cid
            rebuilt: dict[int, dict] = {}
            used_labels: set[str] = set()
            for _, embs in sorted(new_groups.items()):
                centroid = np.mean(embs, axis=0)
                # Find the existing cluster closest to this centroid
                best_cid = min(
                    self._clusters,
                    key=lambda c: np.linalg.norm(self._clusters[c]["centroid"] - centroid)
                )
                c = self._clusters[best_cid]
                label = c["label"]
                while label in used_labels:
                    label = _label_for(int(label, 36) + 1) if label.isdigit() else label + "'"
                used_labels.add(label)
                rebuilt[best_cid] = {
                    "label": label,
                    "centroid": centroid,
                    "embeddings": embs,
                    "snap_path": c["snap_path"]
                }

            # Keep noise clusters untouched
            for cid, c in self._clusters.items():
                if cid not in rebuilt:
                    rebuilt[cid] = c

            self._clusters = rebuilt

    def _load_from_db(self):
        """Restore cluster metadata from DB (centroids not stored — start fresh)."""
        rows = self.db.get_all_unknown_clusters()
        for cid, label in rows:
            self._clusters[cid] = {
                "label": label,
                "centroid": np.zeros(512, dtype=np.float32),
                "embeddings": [],
                "snap_path": None
            }
            if cid >= self._next_id:
                self._next_id = cid + 1
