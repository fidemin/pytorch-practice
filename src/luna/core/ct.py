import functools
import glob
import logging
from dataclasses import dataclass
from typing import Tuple

import SimpleITK as sitk
import numpy as np

from src.luna.core.utils import xyz2irc

logger = logging.getLogger(__name__)


@dataclass
class CTChunk:
    center_irc: Tuple[int, int, int]
    chunk_arr: np.ndarray


class CT:
    def __init__(self, series_uid: str, CT_files_dir: str):
        # print(f"Loading CT data... with series_uid: {series_uid}")
        mhd_path = glob.glob(f"{CT_files_dir}/subset*/{series_uid}.mhd")[0]
        ct_image = sitk.ReadImage(mhd_path)
        ct_array = np.array(sitk.GetArrayFromImage(ct_image), dtype=np.float32)
        ct_array.clip(-1000, 1000, ct_array)

        self.series_uid = series_uid
        self.ct_image = ct_image

        # NOTE: ct_array.shape: (I, C, R). NOT Z, Y, X!!!
        self.ct_array = ct_array

        self.xyz_origin = ct_image.GetOrigin()
        self.xyz_spacing = ct_image.GetSpacing()
        self.direction = ct_image.GetDirection()

    def extract_chunk(
        self,
        center_xyz: Tuple[float, float, float],
        chunk_shape_irc: Tuple[int, int, int],
    ) -> CTChunk:

        center_irc = xyz2irc(
            center_xyz, self.xyz_origin, self.xyz_spacing, self.direction
        )

        slice_list = []
        for axis, center_val in enumerate(center_irc):
            start_ndx = int(round(center_val - chunk_shape_irc[axis] / 2))
            end_ndx = int(start_ndx + chunk_shape_irc[axis])

            if start_ndx < 0:
                start_ndx = 0
                end_ndx = int(chunk_shape_irc[axis])

            if end_ndx > self.ct_array.shape[axis]:
                end_ndx = self.ct_array.shape[axis]
                start_ndx = int(end_ndx - chunk_shape_irc[axis])

            slice_list.append(slice(start_ndx, end_ndx))
        ct_chunk_arr = self.ct_array[tuple(slice_list)]
        return CTChunk(center_irc, ct_chunk_arr)


@functools.lru_cache(maxsize=1)
def get_ct(series_uid: str, CT_files_dir: str) -> CT:
    return CT(series_uid, CT_files_dir)
