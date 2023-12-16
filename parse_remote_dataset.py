import numpy as np
import os
import sys
from PIL import Image
from tqdm import tqdm
import re

IMG_FOLDER = 'rgb'
SEG_FOLDER = 'seg'
VIS_FOLDER = 'vis'


def process_sample(jpg_sample_name: str, output_folder: str, source_folder: str):
    
    pattern = r'frame_(\d+)\.jpg'
    match = re.match(pattern, jpg_sample_name)
    if match:
        id_number = int(match.group(1))
    else:
        print('n id: ', jpg_sample_name)
        pass

    jpg_path = os.path.join(source_folder, IMG_FOLDER, jpg_sample_name)
    vis_path = os.path.join(source_folder, VIS_FOLDER, f'frame_{id_number}.jpg')
    seg_path = os.path.join(source_folder, SEG_FOLDER, f'frame_{id_number}.npy')

    out_path = os.path.join(output_folder, f'sample_{id_number}.npz')

    try:
        img = np.array(Image.open(jpg_path))
        img_vis = np.array(Image.open(vis_path))
        seg = np.load(seg_path)

        print(f'{id_number}: {img.shape} / {img_vis.shape} / {seg.shape}')
        np.savez(out_path, X=img, y=seg, vis=img_vis)
    except Exception as e:
        print(e)

if __name__ == '__main__':

    mount_path = sys.argv[1]
    output_folder = sys.argv[2]
    
    pth = os.path.join(mount_path, 'vis/')
    files = os.listdir(pth)
    pbar = tqdm(total=len(files))
    for filename in os.listdir(os.path.join(mount_path, IMG_FOLDER)):
        process_sample(jpg_sample_name=filename, output_folder=output_folder, source_folder=mount_path)
        pbar.update(1)
    pbar.close()
    print('done')

    
