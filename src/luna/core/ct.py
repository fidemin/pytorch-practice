import glob

import SimpleITK as sitk
import numpy as np


class CT:
    def __init__(self, series_uid: str, CT_files_dir: str):
        mhd_path = glob.glob(f"{CT_files_dir}/subset*/{series_uid}.mhd")[0]
        ct_image = sitk.ReadImage(mhd_path)
        ct_array = np.array(sitk.GetArrayFromImage(ct_image), dtype=np.float32)
        ct_array.clip(-1000, 1000, ct_array)

        self.series_uid = series_uid
        self.ct_image = ct_image
        self.series_uid = series_uid
        self.ct_array = ct_array

        self.xyz_origin = ct_image.GetOrigin()
        self.xyz_spacing = ct_image.GetSpacing()
        self.direction = ct_image.GetDirection()
