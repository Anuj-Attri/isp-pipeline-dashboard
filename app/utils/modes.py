from enum import Enum


class Mode(str, Enum):
    BASELINE = "baseline"
    PROCESSED = "processed"
    COMPARE = "compare"
    DETECT = "detect"


_ORDERED_MODES = [Mode.BASELINE, Mode.PROCESSED, Mode.COMPARE, Mode.DETECT]


def cycle_mode(current: Mode) -> Mode:
    try:
        idx = _ORDERED_MODES.index(current)
    except ValueError:
        return Mode.BASELINE
    return _ORDERED_MODES[(idx + 1) % len(_ORDERED_MODES)]

