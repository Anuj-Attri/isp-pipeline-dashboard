"""ISPController — runtime Picamera2 controls (AE, AWB, AF) without restart."""
from typing import Any


class ISPController:
    """Apply AE/AWB/AF controls via picam2.set_controls() at runtime."""

    @staticmethod
    def set_ae_lock(picam2: Any, locked: bool) -> None:
        """Lock (disable) or unlock (enable) auto exposure."""
        try:
            picam2.set_controls({"AeEnable": 0 if locked else 1})
        except Exception:
            pass

    @staticmethod
    def set_awb_lock(picam2: Any, locked: bool) -> None:
        """Lock (disable) or unlock (enable) auto white balance."""
        try:
            picam2.set_controls({"AwbEnable": 0 if locked else 1})
        except Exception:
            pass

    @staticmethod
    def set_af_mode(picam2: Any, enabled: bool) -> None:
        """Enable (1) or disable (0) auto focus. 0 = manual, 1 = auto."""
        try:
            picam2.set_controls({"AfMode": 1 if enabled else 0})
        except Exception:
            pass
