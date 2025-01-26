from dataclasses import dataclass
from typing import Tuple


@dataclass
class AugmentInfo:
    use_flip: bool = False

    use_offset: bool = False
    offset_factor: float = None

    use_scale: bool = False
    scale_factor: float = None

    use_rotate: bool = False


@dataclass
class CandidateInfo:
    series_uid: str
    center_xyz: Tuple[float, float, float]
    diameter_mm: float
    is_nodule: bool
