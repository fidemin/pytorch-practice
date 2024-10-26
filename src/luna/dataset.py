import csv
import logging
from collections import defaultdict
from dataclasses import dataclass
from typing import List, Tuple, Dict

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


@dataclass
class CandidateInfo:
    series_uid: str
    center_xyz: Tuple[float, float, float]
    diameter_mm: float
    is_nodule: bool


def get_candidate_info_list(
    candidate_file_path: str, annotation_file_path: str
) -> List[CandidateInfo]:

    diameter_dict = _get_diameter_dict(annotation_file_path)
    candidate_dict = _get_candidate_dict(candidate_file_path)

    diameter_serial_uids = set(diameter_dict.keys())
    candidate_serial_uids = set(candidate_dict.keys())
    intersected_series_uids = diameter_serial_uids & candidate_serial_uids
    logger.info(
        f"series_uids size: {len(intersected_series_uids)}, diameter_serial_uids size: {len(diameter_serial_uids)}, candidate_serial_uids size: {len(candidate_serial_uids)}"
    )

    candidate_info_list = []

    for serial_uid in candidate_serial_uids:
        diameter_list = diameter_dict.get(serial_uid, [])
        candidate_list = candidate_dict[serial_uid]

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
