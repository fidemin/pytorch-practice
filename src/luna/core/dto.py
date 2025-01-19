from dataclasses import dataclass


@dataclass
class AugmentInfo:
    use_flip: bool = False

    use_offset: bool = False
    offset_factor: float = None

    use_scale: bool = False
    scale_factor: float = None

    use_rotate: bool = False
