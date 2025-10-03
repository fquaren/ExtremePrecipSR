from train_val_test_split import get_train_val_test_dates
from tqdm import tqdm
import glob
import os


# OPERA raw data downloaded from MCH
raw_data_path = "/work/FAC/FGSE/IDYST/tbeucler/downscaling/raw_data/OPERA"

train_dates_list, val_dates_list, test_dates_list = get_train_val_test_dates(
    raw_data_path
)

print(train_dates_list)

patches_path = (
    "/work/FAC/FGSE/IDYST/tbeucler/downscaling/fquareng/data/OPERA/patches_v2"
)
precip_path = "precip"
dem_path = "dem"

train_files = []
for date in tqdm(train_dates_list):
    train_files.extend(
        glob.glob(
            os.path.join(
                patches_path,
                precip_path,
                date,
                "*/patch_y[0-9][0-9][0-9][0-9]_x[0-9][0-9][0-9][0-9].npy",
            )
        )
    )

print(train_files)

val_files = []
for date in tqdm(val_dates_list):
    val_files.extend(
        glob.glob(
            os.path.join(
                patches_path,
                precip_path,
                date,
                "*/patch_y[0-9][0-9][0-9][0-9]_x[0-9][0-9][0-9][0-9].npy",
            )
        )
    )

test_files = []
for date in tqdm(test_dates_list):
    test_files.extend(
        glob.glob(
            os.path.join(
                patches_path,
                precip_path,
                date,
                "*/patch_y[0-9][0-9][0-9][0-9]_x[0-9][0-9][0-9][0-9].npy",
            )
        )
    )


# Save to file
def save_to_txt(save_path):
    train_files_path = os.path.join(save_path, "train_files.txt")
    with open(train_files_path, "w") as f:
        for path in train_files:
            f.write(path + "\n")
    val_files_path = os.path.join(save_path, "val_files.txt")
    with open(val_files_path, "w") as f:
        for path in val_files:
            f.write(path + "\n")
    test_files_path = os.path.join(save_path, "test_files.txt")
    with open(test_files_path, "w") as f:
        for path in test_files:
            f.write(path + "\n")
    print(f"Saved at {save_path}.")
    return


save_path = "/work/FAC/FGSE/IDYST/tbeucler/downscaling/fquareng/ExtremePrecipSR/data/"
save_to_txt(save_path)
