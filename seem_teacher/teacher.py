import os
import sys

import PIL
from PIL import Image

import gradio as gr
import torch
import argparse
import whisper
import numpy as np
import cv2

from gradio import processing_utils
from modeling.BaseModel import BaseModel
from modeling import build_model
from utils.distributed import init_distributed
from utils.arguments import load_opt_from_config_files
from utils.constants import COCO_PANOPTIC_CLASSES
from detectron2.data import MetadataCatalog

def parse_option():
    parser = argparse.ArgumentParser('SEEM Demo', add_help=False)
    parser.add_argument('--conf_files', default="./configs/seem/focall_unicl_lang_v1.yaml", metavar="FILE", help='path to config file', )
    cfg = parser.parse_args()
    return cfg

'''
build args
'''
cfg = parse_option()
opt = load_opt_from_config_files([cfg.conf_files])
opt = init_distributed(opt)

pretrained_pth = './'

# META DATA
cur_model = 'None'
print('downloading model')
pretrained_pth = os.path.join("seem_focall_v1.pt")
if not os.path.exists(pretrained_pth):
    os.system("wget {}".format("https://huggingface.co/xdecoder/SEEM/resolve/main/seem_focall_v1.pt"))
cur_model = 'Focal-L'
print('downloaded model')
'''
build model
'''
model = BaseModel(opt, build_model(opt)).from_pretrained(pretrained_pth).eval().cuda()
with torch.no_grad():
    model.model.sem_seg_head.predictor.lang_encoder.get_text_embeddings(COCO_PANOPTIC_CLASSES + ["background"], is_eval=True)

'''
audio
'''
print('build model')
audio = whisper.load_model("base")

@torch.no_grad()
def inference(image, task, *args, **kwargs):
    with torch.autocast(device_type='cuda', dtype=torch.float16):
        if 'Video' in task:
            return interactive_infer_video(model, audio, image, task, *args, **kwargs)
        else:
            return interactive_infer_image(model, audio, image, task, *args, **kwargs)

class ImageMask(gr.components.Image):
    """
    Sets: source="canvas", tool="sketch"
    """

    is_template = True

    def __init__(self, **kwargs):
        super().__init__(source="upload", tool="sketch", interactive=True, **kwargs)

    def preprocess(self, x):
        return super().preprocess(x)


def resize_jpeg(jpeg_file: str, size: list = None) -> np.ndarray:
    img = Image.open(jpeg_file)

    if size is not None:
        resized_img = img.resize(tuple(size))
    else:
        resized_img = img
    resized_img_arr = np.array(resized_img)

    return resized_img_arr

def show_image(img: np.ndarray):
    cv2.imshow("demo", cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
    cv2.waitKey(0)
    cv2.destroyWindow("demo")

metadata = MetadataCatalog.get('coco_2017_train_panoptic')
model.model.metadata = metadata
model.model.task_switch['spatial'] = False
model.model.task_switch['visual'] = True
model.model.task_switch['grounding'] = False
model.model.task_switch['audio'] = False




img = resize_jpeg('./../dataset/tests/img04.jpg')
height, width, _ = img.shape

print(img.shape)
images = torch.from_numpy(img.copy()).permute(2,0,1).cuda()
print(images.shape)

data = {"image": images, "height": height, "width": width}
batch_inputs = [data]

results = model.model.evaluate(batch_inputs)
pano_seg = results[-1]['panoptic_seg'][0]
pano_seg_info = results[-1]['panoptic_seg'][1]

print(pano_seg_info)

pano_mask = pano_seg.cpu().numpy()
for id in np.unique(pano_mask):
    mask = pano_mask == id
    image = img.copy()
    image[~mask, :] = 0
    show_image(image)

for info in pano_seg_info:
    cat_info = info['category_id']
    print(f'{cat_info}, {COCO_PANOPTIC_CLASSES[cat_info]}')