"""
High-accuracy face recognizer using MTCNN + InceptionResnetV1 (FaceNet).
Key improvements:
- MTCNN used for BOTH registration AND live recognition (tight face crop)
- Cosine similarity instead of raw L2 for rotation/lighting invariance
- Per-person multi-encoding averaging when multiple photos registered
- Strict threshold tuned for vggface2 pretrained model
"""
import numpy as np
import cv2
import torch
from facenet_pytorch import MTCNN, InceptionResnetV1
import threading
import logging

logger = logging.getLogger(__name__)

# Tuned for InceptionResnetV1 vggface2:
# cosine similarity >= 0.65 = same person (higher = stricter)
RECOGNITION_THRESHOLD = 0.65


def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity between two L2-normalised embeddings."""
    na = np.linalg.norm(a)
    nb = np.linalg.norm(b)
    if na == 0 or nb == 0:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


class FaceRecognizer:
    def __init__(self):
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        logger.info(f"[FaceRecognizer] Device: {self.device}")

        # MTCNN — used for both registration and live recognition
        self.mtcnn = MTCNN(
            keep_all=True,
            min_face_size=40,       # ignore tiny faces
            thresholds=[0.7, 0.8, 0.9],  # stricter cascade
            device=self.device
        )
        # FaceNet embedding model
        self.resnet = InceptionResnetV1(pretrained='vggface2').eval().to(self.device)
        self.ai_lock = threading.Lock()

        # {name: [embedding, ...]}  — supports multiple photos per person
        self.known_encodings: dict[str, list[np.ndarray]] = {}
        # Flat lists for fast vectorised search
        self._enc_matrix: np.ndarray | None = None   # (N, 512)
        self._enc_names: list[str] = []

    # ── Loading ────────────────────────────────────────────────────────────
    def load_known_faces(self, db_manager):
        persons = db_manager.get_registered_persons()
        self.known_encodings = {}
        for p in persons:
            if p[3] is None:
                continue
            enc = np.frombuffer(p[3], dtype=np.float32).copy()
            name = p[1]
            self.known_encodings.setdefault(name, []).append(enc)

        # Build flat matrix for vectorised cosine search
        names, encs = [], []
        for name, enc_list in self.known_encodings.items():
            # Average all encodings for this person → more robust
            avg = np.mean(enc_list, axis=0).astype(np.float32)
            avg /= (np.linalg.norm(avg) + 1e-8)
            names.append(name)
            encs.append(avg)

        if encs:
            self._enc_matrix = np.stack(encs, axis=0)   # (N, 512)
            self._enc_names = names
        else:
            self._enc_matrix = None
            self._enc_names = []

        logger.info(f"[FaceRecognizer] Loaded {len(names)} persons")

    # ── Core embedding ─────────────────────────────────────────────────────
    def _embed(self, face_rgb: np.ndarray) -> np.ndarray | None:
        """Convert a tight RGB face crop (any size) to a 512-d embedding."""
        try:
            face_resized = cv2.resize(face_rgb, (160, 160))
            t = torch.tensor(
                np.transpose(face_resized, (2, 0, 1)), dtype=torch.float32
            ).unsqueeze(0).to(self.device)
            t = (t - 127.5) / 128.0
            with self.ai_lock:
                with torch.no_grad():
                    emb = self.resnet(t).cpu().numpy()[0]
            emb = emb / (np.linalg.norm(emb) + 1e-8)
            return emb.astype(np.float32)
        except Exception as e:
            logger.debug(f"[FaceRecognizer] _embed error: {e}")
            return None

    def _match(self, embedding: np.ndarray):
        """Return (name, cosine_similarity) or ('Unknown', 0.0)."""
        if self._enc_matrix is None or len(self._enc_names) == 0:
            return "Unknown", 0.0
        sims = self._enc_matrix @ embedding          # (N,) cosine sims
        best_idx = int(np.argmax(sims))
        best_sim = float(sims[best_idx])
        if best_sim >= RECOGNITION_THRESHOLD:
            return self._enc_names[best_idx], best_sim
        return "Unknown", 0.0

    # ── MTCNN face detection inside a region ──────────────────────────────
    def _detect_face_in_region(self, frame_rgb: np.ndarray, region_box: list):
        """
        Run MTCNN inside a body bounding box to find the tightest face crop.
        Returns (face_rgb_crop, [fx1,fy1,fx2,fy2] in full-frame coords) or (None, None).
        """
        rx1, ry1, rx2, ry2 = [int(v) for v in region_box]
        h, w = frame_rgb.shape[:2]
        rx1, ry1 = max(0, rx1), max(0, ry1)
        rx2, ry2 = min(w, rx2), min(h, ry2)
        region = frame_rgb[ry1:ry2, rx1:rx2]
        if region.size == 0:
            return None, None
        try:
            with self.ai_lock:
                boxes, probs = self.mtcnn.detect(region)
            if boxes is None or len(boxes) == 0:
                return None, None
            # Pick highest-confidence face
            best = int(np.argmax(probs))
            fx1, fy1, fx2, fy2 = [int(v) for v in boxes[best]]
            fx1, fy1 = max(0, fx1), max(0, fy1)
            fx2, fy2 = min(region.shape[1], fx2), min(region.shape[0], fy2)
            if fx2 - fx1 < 20 or fy2 - fy1 < 20:
                return None, None
            face_crop = region[fy1:fy2, fx1:fx2]
            # Convert back to full-frame coords
            full_box = [rx1 + fx1, ry1 + fy1, rx1 + fx2, ry1 + fy2]
            return face_crop, full_box
        except Exception as e:
            logger.debug(f"[FaceRecognizer] MTCNN error: {e}")
            return None, None

    # ── Public API ─────────────────────────────────────────────────────────
    def recognize_with_encoding(self, frame_bgr: np.ndarray, body_box: list):
        """
        Given a full BGR frame and a body bounding box [x1,y1,x2,y2],
        detect the face with MTCNN, embed it, match against known persons.
        Returns (name, confidence, embedding).
        """
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

        # 1. Use MTCNN to find tight face inside body box
        face_crop, face_box = self._detect_face_in_region(frame_rgb, body_box)

        # 2. Fallback: use top-40% of body box as face region if MTCNN misses
        if face_crop is None:
            bx1, by1, bx2, by2 = [int(v) for v in body_box]
            bh = by2 - by1
            fy2 = min(frame_rgb.shape[0], by1 + int(bh * 0.42))
            face_crop = frame_rgb[by1:fy2, bx1:bx2]
            if face_crop.size == 0:
                return "Unknown", 0.0, None

        # 3. Embed
        embedding = self._embed(face_crop)
        if embedding is None:
            return "Unknown", 0.0, None

        # 4. Match
        name, sim = self._match(embedding)
        return name, sim, embedding

    def recognize(self, frame_bgr: np.ndarray, body_box: list):
        name, conf, _ = self.recognize_with_encoding(frame_bgr, body_box)
        return name, conf

    def get_encoding(self, image_bgr: np.ndarray) -> np.ndarray | None:
        """
        Get embedding for a registration photo.
        Uses MTCNN to find the face, returns 512-d normalised embedding.
        """
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        try:
            with self.ai_lock:
                boxes, probs = self.mtcnn.detect(image_rgb)
        except Exception:
            boxes = None

        if boxes is not None and len(boxes) > 0:
            best = int(np.argmax(probs))
            fx1, fy1, fx2, fy2 = [int(b) for b in boxes[best]]
            fx1, fy1 = max(0, fx1), max(0, fy1)
            fx2, fy2 = min(image_rgb.shape[1], fx2), min(image_rgb.shape[0], fy2)
            face_crop = image_rgb[fy1:fy2, fx1:fx2]
            if face_crop.size > 0:
                return self._embed(face_crop)

        # Fallback: use whole image if no face detected
        logger.warning("[FaceRecognizer] No face detected in registration image — using full image")
        return self._embed(cv2.resize(image_rgb, (160, 160)))
