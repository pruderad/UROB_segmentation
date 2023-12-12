import numpy as np
import os
import sys
from PIL import Image
from tqdm import tqdm

IMG_FOLDER = 'rgb'
SEG_FOLDER = 'seg'
VIS_FOLDER = 'vis'


def process_sample(mount_path: str, output_path: str, sample_id: int):
    img_sample_name = f'frame_{sample_id}.jpg'
    seg_sample_name = f'frame_{sample_id}.npy'
    out_sample_name = f'sample_{sample_id}.npz'

    img_path = os.path.join(mount_path, IMG_FOLDER, img_sample_name)
    seg_path = os.path.join(mount_path, SEG_FOLDER, seg_sample_name)
    vis_path = os.path.join(mount_path, VIS_FOLDER, img_sample_name)
    out_path = os.path.join(output_path, out_sample_name)
    
    try:
        img = np.array(Image.open(img_path))
        img_vis = np.array(Image.open(vis_path))
        seg = np.load(seg_path)
        np.savez(out_path, X=img, y=seg, vis=img_vis)
    except Exception as e:
        print(e)

if __name__ == '__main__':

    mount_path = sys.argv[1]
    output_folder = sys.argv[2]
    
    pth = os.path.join(mount_path, 'vis/')
    files = os.listdir(pth)
    pbar = tqdm(total=len(files))
    for i, _ in enumerate(files):
        process_sample(mount_path=mount_path, output_path=output_folder, sample_id=i)
        pbar.update(1)
    pbar.close()
    print('done')

    
