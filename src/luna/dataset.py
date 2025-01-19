import copy
import csv
import functools
import glob
import logging
import math
import random
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple, Dict, Set

import torch
import torch.nn.functional as F
from toolz import curried as tc
from torch.utils.data import Dataset

from src.luna.core.ct import get_ct
from src.luna.core.dto import AugmentInfo

logger = logging.getLogger(__name__)


def _get_diameter_dict(
    annotation_file_path: str,
) -> Dict[str, List[Tuple[Tuple[float, float, float], float]]]:
    diameter_dict = defaultdict(list)

    with open(annotation_file_path) as f:
        reader = csv.DictReader(f)
        for row_dict in reader:
            # row_dict:
            # {
            #   'seriesuid': '1.3.6.1.4.1.14519.5.2.1.6279.6001.100225287222365663678666836860',
            #   'coordX': '-128.6994211',
            #   'coordY': '-175.3192718',
            #   'coordZ': '-298.3875064',
            #   'diameter_mm': '5.651470635'
            #   }

            seriesuid = row_dict["seriesuid"]
            center = tuple(
                map(float, [row_dict["coordX"], row_dict["coordY"], row_dict["coordZ"]])
            )
            diameter = float(row_dict["diameter_mm"])

            diameter_dict[seriesuid].append((center, diameter))

    return diameter_dict


def _get_candidate_dict(candidate_file_path: str):
    candidate_dict = defaultdict(list)

    with open(candidate_file_path) as f:
        reader = csv.DictReader(f)
        for row_dict in reader:
            # row_dict:
            # {
            #   'seriesuid': '1.3.6.1.4.1.14519.5.2.1.6279.6001.100225287222365663678666836860',
            #   'coordX': '-56.08',
            #   'coordY': '-67.85',
            #   'coordZ': '-311.92',
            #   'class': '0'
            # }
            seriesuid = row_dict["seriesuid"]
            center = tuple(
                map(float, [row_dict["coordX"], row_dict["coordY"], row_dict["coordZ"]])
            )
            class_label = int(row_dict["class"])

            candidate_dict[seriesuid].append((center, class_label))

    return candidate_dict


def _get_series_uid_set_from_ct_files(dir_of_CT_files: str) -> Set[str]:
    series_uid_set_from_ct_files = tc.pipe(
        glob.glob(f"{dir_of_CT_files}/subset*/*.mhd"),
        tc.map(Path),
        tc.map(lambda p: p.stem),
        set,
    )
    return series_uid_set_from_ct_files


@dataclass
class CandidateInfo:
    series_uid: str
    center_xyz: Tuple[float, float, float]
    diameter_mm: float
    is_nodule: bool


@functools.lru_cache(maxsize=1)
def get_candidate_info_list(
    *,
    candidate_file_path: str,
    annotation_file_path: str,
    require_CT_files: bool = False,
    CT_files_dir: str = None,
) -> List[CandidateInfo]:

    series_uid_set_from_ct_files = None
    if require_CT_files:
        if CT_files_dir is None:
            raise ValueError(
                "CT_files_dir must be provided if require_CT_files is True"
            )

        series_uid_set_from_ct_files = _get_series_uid_set_from_ct_files(CT_files_dir)

    diameter_dict = _get_diameter_dict(annotation_file_path)
    candidate_dict = _get_candidate_dict(candidate_file_path)

    candidate_info_list = []

    for serial_uid, candidate_list in candidate_dict.items():
        if require_CT_files and serial_uid not in series_uid_set_from_ct_files:
            logger.warning(
                f"series_uid: {serial_uid} does not have a corresponding CT file"
            )
            continue

        diameter_list = diameter_dict.get(serial_uid, [])

        if not diameter_list:
            logger.info(f"series_uid: {serial_uid} has no diameter info")

        for candidate_xyz, is_nodule in candidate_list:
            candidate_diameter = 0
            for diameter_xyz, diameter_mm in diameter_list:
                threshold = diameter_mm / 4  # half of the radius

                # Check if the candidate_xyz and diameter_xyz is within the threshold of the diameter
                # If it is, it is considered as a same point and the diameter is assigned to the candidate
                if (
                    abs(candidate_xyz[0] - diameter_xyz[0]) <= threshold
                    and abs(candidate_xyz[1] - diameter_xyz[1]) <= threshold
                    and abs(candidate_xyz[2] - diameter_xyz[2]) <= threshold
                ):
                    candidate_diameter = diameter_mm
                    break

            candidate_info = CandidateInfo(
                series_uid=serial_uid,
                center_xyz=candidate_xyz,
                diameter_mm=candidate_diameter,
                is_nodule=bool(is_nodule),
            )
            candidate_info_list.append(candidate_info)

    # Sort by series_uid, is_nodule, diameter_mm -> data with diameter_mm = 0 will be at the end per series_uid
    # Sort by main series_uid is important for the caching of get_ct
    candidate_info_list.sort(
        key=lambda x: (x.series_uid, x.is_nodule, x.diameter_mm, x.center_xyz),
        reverse=True,
    )

    return candidate_info_list


class LunaDataset(Dataset):
    def __init__(
        self,
        candidate_file_path: str,
        annotation_file_path: str,
        CT_files_dir: str,
        /,
        *,
        is_validation: bool = False,
        validation_ratio: float = None,
        negative_data_ratio: float = None,
        augment_info: AugmentInfo = None,
    ):
        self.candidate_info_list = copy.copy(
            get_candidate_info_list(
                candidate_file_path=candidate_file_path,
                annotation_file_path=annotation_file_path,
                require_CT_files=True,
                CT_files_dir=CT_files_dir,
            )
        )

        self.CT_files_dir = CT_files_dir
        self.augment_info = augment_info

        self.validation_ratio = validation_ratio
        self.is_validation = is_validation

        self.negative_data_ratio = negative_data_ratio

        if is_validation:
            assert (
                validation_ratio is not None
            ), "validation_ratio must be provided for is_validation=True"
            val_stride = int(1 / validation_ratio)
            self.candidate_info_list = self.candidate_info_list[::val_stride]
        else:
            if validation_ratio is not None:
                val_stride = int(1 / validation_ratio)
                del self.candidate_info_list[::val_stride]

        self.positive_list = [c for c in self.candidate_info_list if c.is_nodule]
        self.negatives_list = [c for c in self.candidate_info_list if not c.is_nodule]
        logger.info(
            f"original: positive_data_count={len(self.positive_list)}, negative_data_count={len(self.negatives_list)}"
        )

    def shuffle_samples(self):
        if not self.negative_data_ratio:
            return

        random.shuffle(self.positive_list)
        random.shuffle(self.negatives_list)

    def __len__(self):
        if self.negative_data_ratio is not None:
            total_size = 20000
            if not self.is_validation:
                return int(total_size * (1.0 - self.validation_ratio))
            else:
                return int(total_size * self.validation_ratio)
        else:
            return len(self.candidate_info_list)

    def __getitem__(self, idx):
        # if self.negative_data_ratio is 1, then we will have 1 positive and 1 negative
        if self.negative_data_ratio:
            positive_idx = idx // (self.negative_data_ratio + 1)
            negative_idx = idx - positive_idx - 1

            if idx % (self.negative_data_ratio + 1) == 0:
                candidate_info = self.positive_list[
                    positive_idx % len(self.positive_list)
                ]
            else:
                candidate_info = self.negatives_list[
                    negative_idx % len(self.negatives_list)
                ]
        else:
            candidate_info = self.candidate_info_list[idx]

        chunk_shape_irc = (32, 48, 48)
        if self.augment_info and not self.is_validation:
            candidate_tensor = self._get_augmented_ct_candidate(
                self.augment_info,
                candidate_info.series_uid,
                candidate_info.center_xyz,
                chunk_shape_irc,
            )
        else:
            candidate_tensor = self._get_ct_candidate(
                candidate_info.series_uid,
                candidate_info.center_xyz,
                chunk_shape_irc,
            )

        label_tensor = torch.tensor(
            [not candidate_info.is_nodule, candidate_info.is_nodule],
            dtype=torch.float,
        )

        return candidate_tensor, label_tensor

    def _get_ct_candidate(
        self,
        series_uid: str,
        center_xyz: Tuple[float, float, float],
        chunk_shape_irc: Tuple[int, int, int],
    ) -> torch.Tensor:
        ct = get_ct(series_uid, self.CT_files_dir)
        ct_chunk = ct.extract_chunk(center_xyz, chunk_shape_irc)
        # unsqueeze to add channel dimension
        return torch.from_numpy(ct_chunk.chunk_arr).to(torch.float32).unsqueeze(0)

    def _get_augmented_ct_candidate(
        self,
        aug_info: AugmentInfo,
        series_uid: str,
        center_xyz: Tuple[float, float, float],
        chunk_shape_irc: Tuple[int, int, int],
    ):
        ct_original_candidate = self._get_ct_candidate(
            series_uid, center_xyz, chunk_shape_irc
        )

        # NOTE: N x C x D x H x W. N (batch size) is required for affine_grid
        ct_t = ct_original_candidate.unsqueeze(0).to(torch.float32)

        # 4x4 matrix -> for rotation matrix compatibility
        transform_t = torch.eye(4)

        for i in range(3):
            if aug_info.use_flip:
                if random.random() > 0.5:
                    transform_t[i, i] *= -1

            if aug_info.use_offset:
                offset_value = aug_info.offset_factor
                random_float = random.random() * 2 - 1  # random float between -1 and 1
                transform_t[i, 3] *= offset_value * random_float

            if aug_info.use_scale:
                scale_value = aug_info.scale_factor
                random_float = random.random() * 2 - 1
                transform_t[i, i] *= 1.0 + scale_value * random_float

            if aug_info.use_rotate:
                rotate_angle_rad = random.random() * math.pi * 2  # 0 to 2pi
                sin_val = math.sin(rotate_angle_rad)
                cos_val = math.cos(rotate_angle_rad)

                rotation_t = torch.tensor(
                    [
                        [cos_val, -sin_val, 0, 0],
                        [sin_val, cos_val, 0, 0],
                        [0, 0, 1, 0],
                        [0, 0, 0, 1],
                    ]
                )

                transform_t @= rotation_t

        affine_t = F.affine_grid(
            transform_t[:3].unsqueeze(0).to(torch.float32),
            list(ct_t.size()),
            align_corners=False,
        )

        augmented_chunk = F.grid_sample(
            ct_t, affine_t, padding_mode="border", align_corners=False
        ).to("cpu")

        # remove batch dimension: N x C x D x H x W -> C x D x H x W
        return augmented_chunk[0]
