import csv
import glob
import logging
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple, Dict, Set

from toolz import curried as tc

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


def get_candidate_info_list(
    *,
    candidate_file_path: str,
    annotation_file_path: str,
    require_CT_files: bool = False,
    dir_of_CT_files: str = None,
) -> List[CandidateInfo]:

    series_uid_set_from_ct_files = None
    if require_CT_files:
        if dir_of_CT_files is None:
            raise ValueError(
                "dir_of_CT_files must be provided if require_CT_files is True"
            )

        series_uid_set_from_ct_files = _get_series_uid_set_from_ct_files(
            dir_of_CT_files
        )

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

    return candidate_info_list