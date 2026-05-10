import numpy as np
import cv2
from scipy.optimize import linear_sum_assignment

class ObjectTracker:
    def __init__(self, max_age=30, n_init=2, iou_threshold=0.3):
        self.max_age = max_age
        self.n_init = n_init
        self.iou_threshold = iou_threshold
        self.tracks = []
        self.next_id = 1
        self.frame_count = 0

    def _get_hsv_hist(self, frame, bbox):
        """Extract HSV histogram for appearance matching."""
        x1, y1, x2, y2 = [int(v) for v in bbox]
        h, w = frame.shape[:2]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w-1, x2), min(h-1, y2)
        
        if x2 <= x1 or y2 <= y1:
            return None
            
        roi = frame[y1:y2, x1:x2]
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        hist = cv2.calcHist([hsv], [0], None, [32], [0, 180])
        cv2.normalize(hist, hist, 0, 1, cv2.NORM_MINMAX)
        return hist

    def _iou(self, boxA, boxB):
        xA = max(boxA[0], boxB[0])
        yA = max(boxA[1], boxB[1])
        xB = min(boxA[2], boxB[2])
        yB = min(boxA[3], boxB[3])
        interArea = max(0, xB - xA) * max(0, yB - yA)
        boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
        boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
        iou = interArea / float(boxAArea + boxBArea - interArea) if (boxAArea + boxBArea - interArea) > 0 else 0
        return iou

    def update(self, detections, frame):
        self.frame_count += 1
        
        # 1. Predict new positions using velocity
        for track in self.tracks:
            # Simple constant velocity model
            track['bbox'][0] += track['vx']
            track['bbox'][1] += track['vy']
            track['bbox'][2] += track['vx']
            track['bbox'][3] += track['vy']
            track['age'] += 1

        # 2. Convert detections to list of bboxes
        det_bboxes = []
        det_hists = []
        for det in detections:
            bbox, conf, label = det
            x1, y1, w, h = bbox
            box = [x1, y1, x1+w, y1+h]
            det_bboxes.append(box)
            det_hists.append(self._get_hsv_hist(frame, box))

        # 3. Associate detections to tracks using Hungarian Algorithm
        if len(self.tracks) > 0 and len(det_bboxes) > 0:
            cost_matrix = np.zeros((len(self.tracks), len(det_bboxes)), dtype=np.float32)
            for i, track in enumerate(self.tracks):
                for j, det_box in enumerate(det_bboxes):
                    iou_score = self._iou(track['bbox'], det_box)
                    
                    # Appearance score (Bhattacharyya distance)
                    app_score = 0
                    if track['hist'] is not None and det_hists[j] is not None:
                        app_score = cv2.compareHist(track['hist'], det_hists[j], cv2.HISTCMP_BHATTACHARYYA)
                    
                    # Combined cost (Low IoU + High Dist = High Cost)
                    # We use 1 - IOU to turn it into a cost
                    cost_matrix[i, j] = (1.0 - iou_score) + (app_score * 0.5)

            row_indices, col_indices = linear_sum_assignment(cost_matrix)
            
            matched_indices = []
            for r, c in zip(row_indices, col_indices):
                if cost_matrix[r, c] < 0.9: # Threshold for a valid match
                    matched_indices.append((r, c))
                    
            unmatched_tracks = set(range(len(self.tracks))) - set(r for r, c in matched_indices)
            unmatched_dets = set(range(len(det_bboxes))) - set(c for r, c in matched_indices)
            
            # Update matched tracks
            for r, c in matched_indices:
                track = self.tracks[r]
                # Calculate velocity (EMA)
                new_bbox = det_bboxes[c]
                track['vx'] = 0.7 * track['vx'] + 0.3 * (new_bbox[0] - track['bbox'][0])
                track['vy'] = 0.7 * track['vy'] + 0.3 * (new_bbox[1] - track['bbox'][1])
                track['bbox'] = new_bbox
                track['hist'] = det_hists[c]
                track['age'] = 0
                track['hits'] += 1
        else:
            unmatched_tracks = set(range(len(self.tracks)))
            unmatched_dets = set(range(len(det_bboxes)))

        # 4. Handle unmatched tracks (already aged in step 1)
        # We don't need to do anything specific, they just won't be updated

        # 5. Create new tracks for unmatched detections
        for c in unmatched_dets:
            new_track = {
                'id': self.next_id,
                'bbox': det_bboxes[c],
                'hist': det_hists[c],
                'vx': 0,
                'vy': 0,
                'age': 0,
                'hits': 1
            }
            self.tracks.append(new_track)
            self.next_id += 1

        # 6. Cleanup old tracks
        self.tracks = [t for t in self.tracks if t['age'] <= self.max_age]

        # 7. Return confirmed active tracks
        active = []
        for t in self.tracks:
            if t['hits'] >= self.n_init and t['age'] == 0:
                active.append({
                    'id': t['id'],
                    'bbox': t['bbox']
                })
        return active
