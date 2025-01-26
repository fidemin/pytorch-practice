import functools
import glob
import logging
from dataclasses import dataclass
from typing import Tuple

import SimpleITK as sitk
import numpy as np

from src.luna.core.dto import CandidateInfo
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

    def build_is_nodule_mask(
        self, positive_info_list: list[CandidateInfo], threshold_hu: int = -700
    ) -> np.ndarray:
        """
        return a mask of annotation for one CT file
        """

        bounding_box_array = np.zeros(self.ct_array.shape, dtype=np.bool)

        for candidate_info in positive_info_list:
            if not candidate_info.is_nodule:
                raise ValueError("CandidateInfo is not positive")

            center_irc = xyz2irc(
                candidate_info.center_xyz,
                self.xyz_origin,
                self.xyz_spacing,
                self.direction,
            )

            center_index = center_irc[0]  # index
            center_row = center_irc[1]  # row
            center_column = center_irc[2]  # column

            # find proper index radius for nodule
            index_radius = 2

            try:
                while self._is_voxel_nodule(
                    center_index + index_radius,
                    center_row,
                    center_column,
                    threshold_hu,
                ) and self._is_voxel_nodule(
                    center_index - index_radius,
                    center_row,
                    center_column,
                    threshold_hu,
                ):
                    index_radius += 1
            except IndexError:
                logger.warning(f"Index out of bounds for {self.series_uid}")
                index_radius -= 1

            # find proper row radius for nodule
            row_radius = 2
            try:
                while self._is_voxel_nodule(
                    center_index,
                    center_row + row_radius,
                    center_column,
                    threshold_hu,
                ) and self._is_voxel_nodule(
                    center_index,
                    center_row - row_radius,
                    center_column,
                    threshold_hu,
                ):
                    row_radius += 1
            except IndexError:
                logger.warning(f"Index out of bounds for {self.series_uid}")
                row_radius -= 1

            # find proper column radius for nodule
            column_radius = 2
            try:
                while self._is_voxel_nodule(
                    center_index,
                    center_row,
                    center_column + column_radius,
                    threshold_hu,
                ) and self._is_voxel_nodule(
                    center_index,
                    center_row,
                    center_column - column_radius,
                    threshold_hu,
                ):
                    column_radius += 1
            except IndexError:
                logger.warning(f"Index out of bounds for {self.series_uid}")
                column_radius -= 1

            print(
                f"index_radius: {index_radius}, row_radius: {row_radius}, column_radius: {column_radius}"
            )

            # set the bounding box to True
            bounding_box_array[
                center_index - index_radius : center_index + index_radius + 1,
                center_row - row_radius : center_row + row_radius + 1,
                center_column - column_radius : center_column + column_radius + 1,
            ] = True

        # bounding_box_array is from positive_info_list and the shape of True is rectangle.
        # To remove voxels which is lower than threshold_hu, we need to use & operator using whole ct_array.
        mask_array = bounding_box_array & (self.ct_array > threshold_hu)
        return mask_array

    def _is_voxel_nodule(self, index, row, column, threshold_hu: int) -> bool:
        return self.ct_array[index, row, column] > threshold_hu


@functools.lru_cache(maxsize=1, typed=True)
def get_ct(series_uid: str, CT_files_dir: str) -> CT:
    return CT(series_uid, CT_files_dir)
