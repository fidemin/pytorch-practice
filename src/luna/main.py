"""
LUNA data can be downloaded from https://luna16.grand-challenge.org/Download/
"""

from src.luna.core.dataset import get_candidate_info_list


if __name__ == "__main__":
    candidate_info_list = get_candidate_info_list(
        candidate_file_path="../../resources/data/luna/candidates.csv",
        annotation_file_path="../../resources/data/luna/annotations.csv",
        require_CT_files=True,
        dir_of_CT_files="../../resources/data/luna/subsets",
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
