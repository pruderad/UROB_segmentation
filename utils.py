import numpy as np
from PIL import Image
import cv2

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

