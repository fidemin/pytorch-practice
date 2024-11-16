"""
LUNA data can be downloaded from https://luna16.grand-challenge.org/Download/
"""

from src.luna.core.ct import get_ct
from src.luna.core.dataset import get_candidate_info_list, LunaDataset
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
    ct1 = get_ct(candidate_info.series_uid, CT_files_dir)

    xyz = (2.7, 123.3, 100.1)
    irc = xyz2irc(xyz, ct1.xyz_origin, ct1.xyz_spacing, ct1.direction)
    xyz_recovered = irc2xyz(irc, ct1.xyz_origin, ct1.xyz_spacing, ct1.direction)
    print(f"IRC from xyz2irc: {irc}")
    print(f"Original XYZ: {xyz}")
    print(f"Recovered XYZ: {xyz_recovered}")

    ct_chunk = ct1.extract_chunk(xyz, (32, 48, 48))
    print(f"Center IRC: {irc}, Chunk shape: {ct_chunk.chunk_arr.shape}")

    dataset = LunaDataset(candidate_info_list, CT_files_dir, validation_ratio=0.1)
    print(f"Dataset length: {len(dataset)}")
    candidate, label = dataset[1]
    print(f"Candidate shape: {candidate.shape}, Label: {label}")

    # check time
    import time

    start = time.time()
    last_time = start
    for i, data in enumerate(dataset):
        count = i + 1
        if count % 1000 == 0:
            print(f"Processed {count} data")
            print(
                f"Time taken to iterate through the dataset: {time.time() - last_time:.2f} seconds"
            )
            last_time = time.time()
    print(
        f"Time taken to iterate through the dataset: {time.time() - start:.2f} seconds"
    )
