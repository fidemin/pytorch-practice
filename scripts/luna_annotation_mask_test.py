import copy
from collections import defaultdict

import numpy as np
from matplotlib import pyplot as plt

from src.luna.core.ct import get_ct
from src.luna.core.utils import xyz2irc
from src.luna.dataset import get_candidate_info_list


def transparent_cmap(cmap, N=255):
    "Copy colormap and set alpha values"
    mycmap = copy.deepcopy(cmap)
    mycmap._init()
    mycmap._lut[:, -1] = np.linspace(0, 0.75, N + 4)
    return mycmap


tgreen = transparent_cmap(plt.cm.Greens)

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

    positive_candidate_info_list = [x for x in candidate_info_list if x.is_nodule]

    positive_candidate_info_list_by_series_uid = defaultdict(list)

    for candidate_info in positive_candidate_info_list:
        positive_candidate_info_list_by_series_uid[candidate_info.series_uid].append(
            candidate_info
        )

    for (
        series_uid,
        candidate_info_list,
    ) in positive_candidate_info_list_by_series_uid.items():
        print(
            f"Series UID: {series_uid}, Positive candidate count: {len(candidate_info_list)}"
        )

        ct = get_ct(series_uid, CT_files_dir)

        is_nodule_mask_array = ct.build_is_nodule_mask(
            candidate_info_list, threshold_hu=0
        )
        ct_array = ct.ct_array

        for candidate_info in candidate_info_list:
            center_xyz = candidate_info.center_xyz
            center_irc = xyz2irc(
                center_xyz, ct.xyz_origin, ct.xyz_spacing, ct.direction
            )
            ct_chunk = ct.extract_chunk(center_xyz, (32, 48, 48))

            # center_index = int(center_irc[0])
            # center_row = int(center_irc[1])
            # center_column = int(center_irc[2])

            # Extract the axial slice containing the candidate nodule
            slice_idx = int(center_irc[0])
            try:
                # ct_slice = ct_array[slice_idx]
                ct_slice = ct_chunk.chunk_arr[0]
                mask_slice = is_nodule_mask_array[slice_idx]
            except IndexError:
                print(f"Index out of bounds for {series_uid}")
                continue

            # Plot the CT image and overlay the mask
            plt.figure(figsize=(8, 8))
            plt.imshow(
                ct_slice,
                clim=(-1000, 3000),
                cmap="gray",
            )  # CT image
            # plt.imshow(mask_slice, alpha=0.1, cmap="Greens")  # Mask overlay
            plt.title(f"Series UID: {series_uid}, Slice: {slice_idx}")
            plt.axis("off")
            plt.show()
