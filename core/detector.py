import cv2
import numpy as np
import logging
import os
from core.hw_manager import hw_manager

logger = logging.getLogger(__name__)

class PersonDetector:
    def __init__(self, model_path='yolov8n.pt'):
        self.device = hw_manager.get_device()
        self.classes = [0]  # Person
        self.model_path = model_path
        
        try:
            from ultralytics import YOLO
            
            # Dynamic optimization: If no CUDA, try OpenVINO for Intel or ONNX for others
            best_format = hw_manager.get_optimal_yolo_format()
            
            if best_format == 'openvino':
                # Check if openvino version exists, otherwise export it
                ov_path = model_path.replace('.pt', '_openvino_model')
                if not os.path.exists(ov_path):
                    logger.info(f"PersonDetector: Optimizing model for Intel CPU (OpenVINO)...")
                    model = YOLO(model_path)
                    model.export(format='openvino', imgsz=640)
                self.model = YOLO(ov_path, task='detect')
                logger.info("✓ PersonDetector: Using OpenVINO optimized model")
            elif best_format == 'onnx':
                onnx_path = model_path.replace('.pt', '.onnx')
                if not os.path.exists(onnx_path):
                    logger.info(f"PersonDetector: Optimizing model for Windows (ONNX)...")
                    model = YOLO(model_path)
                    model.export(format='onnx', imgsz=640)
                self.model = YOLO(onnx_path, task='detect')
                logger.info("✓ PersonDetector: Using ONNX optimized model")
            else:
                # Default PyTorch (CUDA or CPU)
                self.model = YOLO(model_path).to(self.device)
                logger.info(f"✓ PersonDetector: Using standard YOLOv8 on {self.device}")
                
            self.use_yolo = True
        except Exception as e:
            logger.error(f"✗ PersonDetector: YOLO initialization failed: {e}")
            self.use_yolo = False
            self._init_opencv_detector()

    def _init_opencv_detector(self):
        self.hog = cv2.HOGDescriptor()
        self.hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
        logger.info("! PersonDetector: Falling back to OpenCV HOG")

    def _normalize_lighting(self, frame):
        """Apply OpenCL-accelerated lighting normalization (Works on Integrated GPUs)."""
        if not hw_manager.use_opencl:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            mean_lume = np.mean(gray)
            gamma = 1.4 if mean_lume < 50 else (0.8 if mean_lume > 200 else 1.0)
            if gamma != 1.0:
                invGamma = 1.0 / gamma
                table = np.array([((i / 255.0) ** invGamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
                frame = cv2.LUT(frame, table)
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
            lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
            l, a, b = cv2.split(lab)
            l = clahe.apply(l)
            return cv2.cvtColor(cv2.merge((l,a,b)), cv2.COLOR_LAB2BGR)
        else:
            u_frame = cv2.UMat(frame)
            u_gray = cv2.cvtColor(u_frame, cv2.COLOR_BGR2GRAY)
            mean_lume = cv2.mean(u_gray)[0]
            gamma = 1.4 if mean_lume < 50 else (0.8 if mean_lume > 200 else 1.0)
            if gamma != 1.0:
                invGamma = 1.0 / gamma
                table = np.array([((i / 255.0) ** invGamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
                u_frame = cv2.LUT(u_frame, table)
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
            u_lab = cv2.cvtColor(u_frame, cv2.COLOR_BGR2LAB)
            u_l, u_a, u_b = cv2.split(u_lab)
            u_l = clahe.apply(u_l)
            u_final = cv2.cvtColor(cv2.merge((u_l, u_a, u_b)), cv2.COLOR_LAB2BGR)
            return u_final.get()

    def detect(self, frame):
        norm_frame = self._normalize_lighting(frame)
        conf_thresh = 0.40 # Slightly lower for optimized models
        
        if self.use_yolo:
            # Use augment=False for maximum speed on non-CUDA systems
            results = self.model.predict(norm_frame, classes=self.classes, conf=conf_thresh, imgsz=640, verbose=False)
            detections = []
            for result in results:
                for box in result.boxes:
                    x1, y1, x2, y2 = box.xyxy[0].tolist()
                    conf = float(box.conf[0])
                    bw, bh = x2-x1, y2-y1
                    if bh > 25 and 0.6 < (bh/bw) < 6.0:
                        detections.append(([x1, y1, bw, bh], conf, 'person'))
            return detections
        else:
            return self._detect_opencv(norm_frame)

    def detect_batch(self, frames):
        if not self.use_yolo or not frames:
            return [self.detect(f) for f in frames]
            
        norm_frames = [self._normalize_lighting(f) for f in frames]
        # Batch size optimization: limit to 2 or 4 on CPU to avoid latency spikes
        results = self.model.predict(norm_frames, classes=self.classes, conf=0.40, imgsz=640, verbose=False)
        
        batch_detections = []
        for result in results:
            detections = []
            for box in result.boxes:
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                bw, bh = x2-x1, y2-y1
                if bh > 25 and 0.6 < (bh/bw) < 6.0:
                    detections.append(([x1, y1, bw, bh], float(box.conf[0]), 'person'))
            batch_detections.append(detections)
        return batch_detections

    def _detect_opencv(self, frame):
        rects, weights = self.hog.detectMultiScale(frame, winStride=(8, 8), padding=(4, 4), scale=1.05)
        return [([x, y, w, h], float(weights[i]), 'person') for i, (x, y, w, h) in enumerate(rects)]
