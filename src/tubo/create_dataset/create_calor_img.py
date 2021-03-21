import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cv2
from pathlib import Path
from tqdm import tqdm

#yukiさんのpalette2img
def generate_img(df: pd.DataFrame, save_dir:str, img_width=224, img_height=224):
    df = df.sort_values("ratio", ascending=False).reset_index(drop=True)
    img = np.zeros((img_width, img_height, 3), dtype=int)

    total_ratio = 0
    for row in df.itertuples():
        width_start = int(img_width * (total_ratio))
        width_end = int(img_width * (total_ratio + row.ratio))
        img[:, width_start:width_end, 0] = row.color_b
        img[:, width_start:width_end, 1] = row.color_g
        img[:, width_start:width_end, 2] = row.color_r
        total_ratio += row.ratio
    
    save_path = os.path.join(save_dir, f"{row.object_id}.jpg")
    cv2.imwrite(save_path, img)

# dataが入っているディレクトリを指定してください
data_dir = Path('../../../data/')

# 保存するディレクトリをしていしてください
features_dir = Path('../../../features/tubo/')
images_dir = features_dir.joinpath('images')

def main():
    palette_df = pd.read_csv(data_dir.joinpath('palette.csv'))
    for object_id, df in tqdm(palette_df.groupby('object_id')):
        generate_img(df, str(images_dir))

if __name__ == '__main__':
    main()