"""PipelineState: single source of truth for ISP toggles and AI mode. Thread-safe singleton."""
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, Optional

import threading


class AIMode(str, Enum):
    NONE = "none"
    DETECTION = "detection"
    SEGMENTATION = "segmentation"
    POSE = "pose"
    DEPTH = "depth"
    EGO_EXO = "ego_exo"


IMX500_DETECTION_MODEL = (
    "/usr/share/imx500-models/imx500_network_ssd_mobilenetv2_fpnlite_320x320_pp.rpk"
)
IMX500_POSE_MODEL = "/usr/share/imx500-models/imx500_network_higherhrnet_coco.rpk"
IMX500_MODES = {AIMode.DETECTION, AIMode.POSE}
CPU_MODES = {AIMode.SEGMENTATION, AIMode.DEPTH, AIMode.EGO_EXO}


@dataclass
class PipelineState:
    """All toggle state for ISP and AI. Use get_instance() for the singleton."""

    sharpening: bool = True
    sharpening_strength: float = 1.0
    denoising: bool = False
    denoising_strength: float = 1.0
    denoising_mode: str = "fast"  # "fast" or "quality"
    demosaicing_quality_fast: bool = True  # True = fast, False = high-quality
    ae_lock: bool = False
    awb_lock: bool = False
    af_enabled: bool = False
    ai_mode: AIMode = AIMode.NONE

    _lock: threading.Lock = field(default_factory=threading.Lock)

    def set_isp(self, key: str, value: Any) -> None:
        with self._lock:
            if key == "sharpening":
                self.sharpening = bool(value)
            elif key == "sharpening_strength":
                self.sharpening_strength = max(0.0, min(2.0, float(value)))
            elif key == "denoising":
                self.denoising = bool(value)
            elif key == "denoising_strength":
                self.denoising_strength = max(0.0, min(2.0, float(value)))
            elif key == "denoising_mode":
                self.denoising_mode = "quality" if str(value).strip().lower() == "quality" else "fast"
            elif key == "demosaicing_quality_fast":
                self.demosaicing_quality_fast = bool(value)
            elif key == "ae_lock":
                self.ae_lock = bool(value)
            elif key == "awb_lock":
                self.awb_lock = bool(value)
            elif key == "af_enabled":
                self.af_enabled = bool(value)

    def set_ai_mode(self, mode: str) -> None:
        with self._lock:
            try:
                self.ai_mode = AIMode(mode)
            except ValueError:
                self.ai_mode = AIMode.NONE

    def get_snapshot(self) -> Dict[str, Any]:
        with self._lock:
            return {
                "sharpening": self.sharpening,
                "sharpening_strength": self.sharpening_strength,
                "denoising": self.denoising,
                "denoising_strength": self.denoising_strength,
                "denoising_mode": self.denoising_mode,
                "demosaicing_quality_fast": self.demosaicing_quality_fast,
                "ae_lock": self.ae_lock,
                "awb_lock": self.awb_lock,
                "af_enabled": self.af_enabled,
                "ai_mode": self.ai_mode.value,
            }


_instance: Optional[PipelineState] = None
_instance_lock = threading.Lock()


def get_pipeline_state() -> PipelineState:
    global _instance
    with _instance_lock:
        if _instance is None:
            _instance = PipelineState()
        return _instance
