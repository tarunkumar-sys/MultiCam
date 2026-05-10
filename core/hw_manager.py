import cv2
import os
import logging
import platform
import subprocess

logger = logging.getLogger(__name__)

class HardwareManager:
    """Manages GPU acceleration and hardware-specific optimizations without CUDA."""
    
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(HardwareManager, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        if self._initialized:
            return
            
        self.use_opencl = cv2.ocl.haveOpenCL()
        if self.use_opencl:
            cv2.ocl.setUseOpenCL(True)
            logger.info("✓ HardwareManager: OpenCL is enabled (works on Intel/AMD/NVIDIA).")
        
        self.device = self._detect_best_device()
        self.cpu_info = platform.processor()
        self.is_intel = "Intel" in self.cpu_info
        
        logger.info(f"✓ HardwareManager: Detected {self.cpu_info}")
        self._initialized = True

    def _detect_best_device(self):
        """Dynamic detection of the best available backend."""
        try:
            import torch
            if torch.cuda.is_available():
                return '0'
        except ImportError:
            pass
            
        # Check for DirectML or OpenVINO through available ORT providers if possible
        try:
            import onnxruntime as ort
            providers = ort.get_available_providers()
            if 'DmlExecutionProvider' in providers: return 'dml'
            if 'OpenVINOExecutionProvider' in providers: return 'openvino'
        except ImportError:
            pass
            
        # Standard fallback
        return 'cpu'

    def get_optimal_yolo_format(self):
        """Returns the best YOLO export format for current hardware."""
        if self.device == '0': return None # Use native PyTorch
        if self.is_intel: return 'openvino'
        if platform.system() == "Windows": return 'onnx' # Good for DirectML
        return None

    def get_device(self):
        return self.device

    def is_gpu_available(self):
        # OpenCL or CUDA or DirectML
        return self.use_opencl or self.device != 'cpu'

hw_manager = HardwareManager()
