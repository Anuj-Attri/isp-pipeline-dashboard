"""Detection interface: backends return lists of Detection; COCO_NAMES for labels."""
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Tuple

COCO_NAMES = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat",
    "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog",
    "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
    "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite",
    "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle",
    "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich",
    "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch",
    "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote",
    "keyboard", "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator",
    "book", "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush",
]


@dataclass
class Detection:
    """Single detection: class label, confidence, bbox in pixel coords."""
    class_id: int
    class_name: str
    confidence: float
    bbox: Tuple[int, int, int, int]  # x1, y1, x2, y2


class DetectionBackend(ABC):
    """Interface for object detection. IMX500 or host-side backends implement this."""

    @abstractmethod
    def detect(self, frame_bgr) -> Tuple[List[Detection], float]:
        """Run detection on a BGR frame. Returns (detections, inference_time_ms)."""
        pass
