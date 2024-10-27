"""
LUNA data can be downloaded from https://luna16.grand-challenge.org/Download/
"""

from src.luna.core.ct import CT
from src.luna.core.dataset import get_candidate_info_list
from src.luna.core.utils import xyz2irc, irc2xyz

if __name__ == "__main__":
    CT_files_dir = "../../resources/data/luna/subsets"
    candidate_info_list = get_candidate_info_list(
        candidate_file_path="../../resources/data/luna/candidates.csv",
        annotation_file_path="../../resources/data/luna/annotations.csv",
        require_CT_files=True,
        CT_files_dir=CT_files_dir,
    )

    total_data_count = len(candidate_info_list)
    wrong_data_count = 0
    positive_data_count = 0
    negative_data_count = 0
    for candidate_info in candidate_info_list:
        if candidate_info.is_nodule and candidate_info.diameter_mm == 0:
            wrong_data_count += 1

        elif candidate_info.is_nodule:
            positive_data_count += 1

        elif not candidate_info.is_nodule:
            negative_data_count += 1

    print(f"Total data count: {total_data_count}")
    print(f"Positive data count: {positive_data_count}")
    print(f"Negative data count: {negative_data_count}")
    print(f"Wrong data count: {wrong_data_count}")
    assert (
        total_data_count == positive_data_count + negative_data_count + wrong_data_count
    )

    candidate_info = candidate_info_list[100]
    ct1 = CT(candidate_info.series_uid, CT_files_dir)
    xyz = (2.7, 123.3, 100.1)
    irc = xyz2irc(xyz, ct1.xyz_origin, ct1.xyz_spacing, ct1.direction)
    xyz_recovered = irc2xyz(irc, ct1.xyz_origin, ct1.xyz_spacing, ct1.direction)
    print(f"IRC from xyz2irc: {irc}")
    print(f"Original XYZ: {xyz}")
    print(f"Recovered XYZ: {xyz_recovered}")

    irc, chunk_arr = ct1.extract_chunk(xyz, (32, 48, 48))
