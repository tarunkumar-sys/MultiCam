import cv2
import numpy as np

class PersonDetector:
    def __init__(self, model_path='yolov8n.pt'):
        # Try to use YOLO if available, fallback to OpenCV HOG
        self.use_yolo = False
        self.use_opencv_hog = False
        
        try:
            from ultralytics import YOLO
            self.model = YOLO(model_path)
            self.classes = [0]  # COCO class for person is 0
            self.use_yolo = True
            print("[PersonDetector] Using YOLOv8 for person detection")
        except Exception as e:
            print(f"[PersonDetector] YOLO not available: {e}")
            print("[PersonDetector] Falling back to OpenCV HOG+SVM detector")
            self._init_opencv_detector()

    def _init_opencv_detector(self):
        """Initialize OpenCV's HOG+SVM person detector as fallback"""
        self.hog = cv2.HOGDescriptor()
        self.hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
        self.use_opencv_hog = True

    def detect(self, frame):
        detections = []
        
        if self.use_yolo:
            try:
                detections = self._detect_yolo(frame)
            except Exception as e:
                print(f"[PersonDetector] YOLO detection failed: {e}, switching to fallback")
                self.use_yolo = False
                self._init_opencv_detector()
        
        if not self.use_yolo and self.use_opencv_hog:
            detections = self._detect_opencv(frame)
            
        return detections

    def _detect_yolo(self, frame):
        """YOLOv8 detection optimized for distant and small persons."""
        # Use imgsz=800 and conf=0.35 as requested
        results = self.model.predict(frame, classes=self.classes, conf=0.35, imgsz=800, verbose=False)
        detections = []
        
        for result in results:
            boxes = result.boxes
            for box in boxes:
                # [x1, y1, x2, y2], confidence, class
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                conf = float(box.conf[0])
                w, h = x2 - x1, y2 - y1

                # Apply requested filters for small/distant persons
                # height >= 25, width >= 10, aspect_ratio between 0.8 and 5.0
                if h < 25 or w < 10:
                    continue
                
                aspect_ratio = h / max(1.0, w)
                if not (0.8 <= aspect_ratio <= 5.0):
                    continue

                detections.append(([x1, y1, w, h], conf, "person"))
        return detections

    def _detect_opencv(self, frame):
        """OpenCV HOG+SVM detection as fallback"""
        detections = []
        h_orig, w_orig = frame.shape[:2]
        
        # Resize if max dimension > 640 as requested
        scale = 1.0
        if max(h_orig, w_orig) > 640:
            scale = 640 / max(h_orig, w_orig)
            processed_frame = cv2.resize(frame, (int(w_orig * scale), int(h_orig * scale)))
        else:
            processed_frame = frame
        
        # Detect people using HOG+SVM
        rects, weights = self.hog.detectMultiScale(
            processed_frame, 
            winStride=(8, 8),
            padding=(8, 8),
            scale=1.05
        )
        
        for i, (x, y, w, h) in enumerate(rects):
            conf = float(weights[i])
            
            # Scale back bounding box correctly
            if scale != 1.0:
                x = int(x / scale)
                y = int(y / scale)
                w = int(w / scale)
                h = int(h / scale)
            
            # Filter: height >= 40, aspect_ratio between 1.1 and 6.0
            if h < 40:
                continue
            
            aspect_ratio = h / max(1.0, w)
            if not (1.1 <= aspect_ratio <= 6.0):
                continue
            
            detections.append(([x, y, w, h], conf, "person"))
        
        return detections
