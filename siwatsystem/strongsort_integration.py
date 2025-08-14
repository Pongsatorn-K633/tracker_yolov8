"""
StrongSORT Integration for MPTA Pipeline
Optimized for YOLOv8/v11 + StrongSORT tracking in pipeline.json
"""
import os
import sys
import logging
from pathlib import Path
import torch
import numpy as np

# Setup paths for tracker imports
CURRENT_DIR = Path(__file__).parent.parent
TRACKERS_DIR = CURRENT_DIR / "trackers"

if str(TRACKERS_DIR / "strongsort") not in sys.path:
    sys.path.append(str(TRACKERS_DIR / "strongsort"))

try:
    # Add the trackers directory to Python path
    if str(TRACKERS_DIR) not in sys.path:
        sys.path.append(str(TRACKERS_DIR))
    
    from strongsort.strong_sort import StrongSORT
    from strongsort.utils.parser import get_config
    STRONGSORT_AVAILABLE = True
except ImportError as e:
    logging.warning(f"StrongSORT not available: {e}")
    STRONGSORT_AVAILABLE = False

logger = logging.getLogger("strongsort_integration")


class PipelineStrongSORT:
    """
    Lean StrongSORT implementation optimized for pipeline.json integration
    """
    
    def __init__(self, config: dict, mpta_dir: str, device: str = "cpu"):
        """
        Initialize StrongSORT tracker for pipeline use
        
        Args:
            config: Tracking configuration from pipeline.json
            mpta_dir: Directory containing model files
            device: Device to run on
        """
        self.config = config
        self.mpta_dir = mpta_dir
        self.device = device
        self.tracker = None
        self.enabled = config.get("enabled", False)
        
        if not self.enabled:
            logger.info("Tracking disabled in configuration")
            return
            
        if not STRONGSORT_AVAILABLE:
            logger.error("StrongSORT not available, tracking disabled")
            self.enabled = False
            return
            
        self._initialize_tracker()
    
    def _initialize_tracker(self):
        """Initialize the StrongSORT tracker"""
        try:
            # Get ReID model path
            reid_model = self.config.get("reidModel", "osnet_x0_25_msmt17.pt")
            reid_path = os.path.join(self.mpta_dir, reid_model)
            
            if not os.path.exists(reid_path):
                logger.error(f"ReID model not found: {reid_path}")
                self.enabled = False
                return
            
            # Get tracker configuration
            tracker_config = self.config.get("config", {})
            
            # Initialize StrongSORT with custom config
            self.tracker = StrongSORT(
                model_weights=reid_path,
                device=self.device,
                fp16=False,  # Disable FP16 for stability
                max_dist=tracker_config.get("max_cos_dist", 0.4),
                max_iou_dist=tracker_config.get("max_iou_dist", 0.7),
                max_age=tracker_config.get("max_age", 30),
                max_unmatched_preds=tracker_config.get("max_unmatched_preds", 7),
                n_init=tracker_config.get("n_init", 3),
                nn_budget=tracker_config.get("nn_budget", 100),
                mc_lambda=tracker_config.get("mc_lambda", 0.995),
                ema_alpha=tracker_config.get("ema_alpha", 0.9)
            )
            
            # Warmup the tracker
            if hasattr(self.tracker, 'model') and hasattr(self.tracker.model, 'warmup'):
                self.tracker.model.warmup()
                
            logger.info(f"StrongSORT tracker initialized with ReID model: {reid_model}")
            
        except Exception as e:
            logger.error(f"Failed to initialize StrongSORT: {e}")
            self.enabled = False
            self.tracker = None
    
    def update(self, detections, original_frame):
        """
        Update tracker with new detections
        
        Args:
            detections: Detection results from YOLO
            original_frame: Original frame for ReID feature extraction
            
        Returns:
            Updated detections with track IDs
        """
        if not self.enabled or self.tracker is None:
            return detections
            
        try:
            # Convert detections to format expected by StrongSORT
            if len(detections) == 0:
                return detections
                
            # Extract bounding boxes and confidences
            boxes = []
            confs = []
            clss = []
            
            for det in detections:
                bbox = det.get("bbox")
                if bbox:
                    x1, y1, x2, y2 = bbox
                    boxes.append([x1, y1, x2, y2])
                    confs.append(det.get("confidence", 0.5))
                    # Map class names to indices (simplified)
                    clss.append(0)  # Default class index
            
            if not boxes:
                return detections
                
            # Convert to tensor format
            det_tensor = torch.tensor(np.column_stack([
                boxes, confs, clss
            ]), dtype=torch.float32)
            
            # Update tracker
            tracks = self.tracker.update(det_tensor, original_frame)
            
            # Update detections with track IDs
            if len(tracks) > 0:
                for i, det in enumerate(detections):
                    if i < len(tracks):
                        track_id = int(tracks[i][4])  # Track ID is at index 4
                        det["track_id"] = track_id
                        det["tracking_confidence"] = float(tracks[i][6])  # Tracking confidence
                        
            return detections
            
        except Exception as e:
            logger.error(f"Error in StrongSORT tracking: {e}")
            return detections
    
    def reset(self):
        """Reset tracker state"""
        if self.tracker and hasattr(self.tracker, 'tracker'):
            if hasattr(self.tracker.tracker, 'tracks'):
                self.tracker.tracker.tracks = []
            logger.debug("Tracker state reset")


def create_pipeline_tracker(tracking_config: dict, mpta_dir: str, device: str = "cpu"):
    """
    Factory function to create a pipeline-optimized tracker
    
    Args:
        tracking_config: Configuration from pipeline.json
        mpta_dir: Directory containing model files  
        device: Device to run on
        
    Returns:
        PipelineStrongSORT instance or None if disabled
    """
    if not tracking_config.get("enabled", False):
        return None
        
    method = tracking_config.get("method", "bytetrack")
    
    if method == "strongsort":
        return PipelineStrongSORT(tracking_config, mpta_dir, device)
    else:
        logger.info(f"Tracking method '{method}' not implemented, using ultralytics default")
        return None


def integrate_tracking_with_ultralytics(model, tracking_config: dict, mpta_dir: str):
    """
    Configure ultralytics model for optimal tracking
    
    Args:
        model: YOLO model instance
        tracking_config: Tracking configuration
        mpta_dir: Model directory
        
    Returns:
        Dictionary with tracking parameters for model.track()
    """
    if not tracking_config.get("enabled", False):
        return {}
        
    method = tracking_config.get("method", "bytetrack")
    
    if method == "strongsort":
        # Configure for StrongSORT
        strongsort_yaml = TRACKERS_DIR / "strongsort" / "configs" / "strongsort.yaml"
        if strongsort_yaml.exists():
            return {
                "tracker": str(strongsort_yaml),
                "persist": True
            }
        else:
            logger.warning("StrongSORT config not found, using default ByteTrack")
            
    return {"persist": True}  # Default tracking parameters