import os
import numpy as np
import pandas as pd
import torch
from concurrent.futures import ProcessPoolExecutor
from tqdm import tqdm
from datetime import datetime

# === Configuration ===
RAW_ROOT = "/scratch/users/dtran/croptype/dataset/TimeSen2Crop"
SAVE_DIR = "./"
N_BANDS = 9
N_MONTHS = 12
BAND_INDICES = list(range(9))  # B1-B9 → B2-B12
FLAG_COL = 'Flag'
TRAIN_TILES = ['32TNT', '32TPT', '32TQT', '33TUM', '33TUN', '33TVM', '33TVN',
               '33TWM', '33TXN', '33UUP', '33UWP', '33UWQ', '33UXP']
VAL_TILE = ['33TWN']
TEST_TILE = ['33UVP']

# === Helper Functions ===
def extract_monthly_median(data: pd.DataFrame, dates: list) -> np.ndarray:
    reflectance = data.iloc[:, BAND_INDICES].values
    flags = data[FLAG_COL].values
    months = [datetime.strptime(str(d), "%Y%m%d").month for d in dates]
    monthly = np.zeros((N_MONTHS, N_BANDS), dtype=np.float32)

    for m in range(1, 13):
        idx = [i for i, mo in enumerate(months) if mo == m and flags[i] == 0]
        if idx:
            monthly[m - 1] = np.median(reflectance[idx], axis=0)
        else:
            monthly[m - 1] = 0.0
    return monthly

def build_label_map(root: str) -> dict:
    label_map = {}
    for tile in sorted(os.listdir(root)):
        tile_path = os.path.join(root, tile)
        if not os.path.isdir(tile_path): continue
        for cls in sorted(os.listdir(tile_path)):
            if cls.isdigit() and cls not in label_map:
                label_map[cls] = int(cls)
    return label_map

def process_sample(cls_path, file, date_list, label):
    try:
        df = pd.read_csv(os.path.join(cls_path, file))
        if len(df) != len(date_list):
            return None
        result = extract_monthly_median(df, date_list)
        return result, label
    except Exception as e:
        print(f"Error in {cls_path}/{file}: {e}")
        return None

def process_sample_wrapper(args):
    return process_sample(*args)

def process_tile_parallel(tile: str, label_map: dict) -> list:
    tile_data = []
    metadata = []
    tile_path = os.path.join(RAW_ROOT, tile)
    if not os.path.isdir(tile_path): return tile_data

    dates_path = os.path.join(tile_path, "dates.csv")
    if not os.path.exists(dates_path):
        print(f"Missing {dates_path}, skipping...")
        return tile_data
    try:
        date_list = pd.read_csv(dates_path)["acquisition_date"].tolist()
    except Exception as e:
        print(f"Could not read dates.csv in {tile}: {e}")
        return tile_data

    tasks = []
    for cls in sorted(os.listdir(tile_path)):
        cls_path = os.path.join(tile_path, cls)
        if not os.path.isdir(cls_path) or not cls.isdigit():
            continue
        label = label_map[cls]
        for file in sorted(os.listdir(cls_path)):
            if file.endswith(".csv"):
                tasks.append((cls_path, file, date_list, label))
                metadata.append((tile, label, file))

    with ProcessPoolExecutor(max_workers=16) as executor:
        for result in tqdm(executor.map(process_sample_wrapper, tasks),
                           total=len(tasks), desc=f"{tile} samples"):
            if result is not None:
                tile_data.append(result)
    return tile_data, metadata

def process_split(split_name, tiles, label_map):
    all_data, all_labels = [], []
    all_metadata = []
    if len(tiles) > 1:
        # Parallelize over tiles
        with ProcessPoolExecutor(max_workers=16) as executor:
            for tile_data, metadata in tqdm(
                executor.map(process_tile_parallel, tiles, [label_map]*len(tiles)),
                desc=f"Processing {split_name} tiles"
            ):
                for result, label in tile_data:
                    all_data.append(result)
                    all_labels.append(label)
                all_metadata.append(metadata)
    else:
        # Single tile — still use ProcessPoolExecutor but show sample progress
        tile_data, metadata = process_tile_parallel(tiles[0], label_map)
        for result, label in tile_data:
            all_data.append(result)
            all_labels.append(label)
        all_metadata.append(metadata)

    X = torch.tensor(np.array(all_data), dtype=torch.float32)
    y = torch.tensor(np.array(all_labels), dtype=torch.long)
    torch.save(X, os.path.join(SAVE_DIR, f"{split_name}_X.pt"))
    torch.save(y, os.path.join(SAVE_DIR, f"{split_name}_y.pt"))
    print(f"Saved {split_name}: {X.shape[0]} samples to {split_name}_X.pt / {split_name}_y.pt")

    metadata_flat = [item for sublist in all_metadata for item in sublist]  # Flatten the list of lists
    metadata_df = pd.DataFrame(metadata_flat, columns=["tile", "label", "file"])
    metadata_df.to_csv(os.path.join(SAVE_DIR, f"{split_name}_metadata.csv"), index=False)
    print(f"Saved {split_name} metadata to {split_name}_metadata.csv")


# === Run All ===
if __name__ == "__main__":
    label_map = build_label_map(RAW_ROOT)
    process_split("train", TRAIN_TILES, label_map)
    process_split("val", VAL_TILE, label_map)
    process_split("test", TEST_TILE, label_map)
    
X_train = torch.load("train_X.pt")
y_train = torch.load("train_y.pt")
metadata_df = pd.read_csv("train_metadata.csv")

# Shape of train dataset: [N, n_months, n_bands], there are N samples, each sample is a n_months x n_bands matrix showing the median values of each band across each month acquired
print("Shape:", X_train.shape, y_train.shape, metadata_df.shape)
#Show example of data
print("Sample data:", X_train[0])#Note: months in the output (1/2018, 2/2018, ..., 7/2018, 9/2017, ..., 12/2017) is arranged in different order from dates.csv file
# Show example of labels
print("Labels:", y_train[:10])
print("Example metadata:", metadata_df.iloc[0])
