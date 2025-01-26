from collections import defaultdict

from matplotlib import pyplot as plt

from src.luna.core.ct import get_ct
from src.luna.dataset import get_candidate_info_list

if __name__ == "__main__":
    CT_files_dir = "../../resources/data/luna/subsets"
    candidate_file_path = "../../resources/data/luna/candidates.csv"
    annotation_file_path = "../../resources/data/luna/annotations.csv"
    candidate_info_list = get_candidate_info_list(
        candidate_file_path=candidate_file_path,
        annotation_file_path=annotation_file_path,
        require_CT_files=True,
        CT_files_dir=CT_files_dir,
    )

    info_list = [x for x in candidate_info_list if x.is_nodule]

    print(f"Positive info count: {len(info_list)}")

    positive_info_list_by_series_uid = defaultdict(list)

    for positive_info in info_list:
        positive_info_list_by_series_uid[positive_info.series_uid].append(positive_info)

    for series_uid, info_list in positive_info_list_by_series_uid.items():
        print(f"Series UID: {series_uid}, Positive info count: {len(info_list)}")

        ct = get_ct(series_uid, CT_files_dir)

        for positive_info in info_list:
            center_xyz = positive_info.center_xyz
            ct_chunk = ct.extract_chunk(center_xyz, (32, 48, 48))
            ct_chunk_arr = ct_chunk.chunk_arr

            # Define grid dimensions
            rows, cols = 4, 8
            num_slices = ct_chunk_arr.shape[0]

            # Create a figure with subplots arranged in a 4x8 grid
            fig, axes = plt.subplots(rows, cols, figsize=(cols * 2, rows * 2))
            axes = axes.flatten()  # Flatten to easily index subplots

            # Plot each slice into the grid
            for i in range(rows * cols):
                if i < num_slices:
                    axes[i].imshow(ct_chunk_arr[i], cmap="gray")
                    axes[i].axis("off")  # Hide axes
                else:
                    axes[i].axis("off")  # Hide unused subplots

            # Add a title to the entire figure
            fig.suptitle(f"Positive Info: {center_xyz}", fontsize=16)
            plt.tight_layout()
            plt.show()
