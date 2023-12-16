from torch.utils.data import DataLoader, Dataset
import pickle   
import numpy as np
import torch
from PIL import ImageOps, Image
class UROBDataset(Dataset):

    def __init__(self, filenames_file: str, target_img_shape: list,  label_mapping: dict = None) -> None:
        with open(filenames_file, 'rb') as file:
            self.filenames = pickle.load(file)

        if label_mapping is not None:
            self.label_mapping = label_mapping
        else:
            self.label_mapping = {
                2 : 1 # car
            }

        self.target_shape = target_img_shape

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, index):
        filepath = self.filenames[index]
        with np.load(filepath, allow_pickle=True) as file:
            img = file['X']
            sem_seg = file['y']

            print(img.shape, sem_seg.shape)

        labels = np.zeros_like(sem_seg, dtype=float)
        for label_mapping_key in self.label_mapping.keys():
            mask = sem_seg == label_mapping_key
            labels[mask] = self.label_mapping[label_mapping_key]

        img = 2 * img / 255 - 1

        # resize the image
        img = self.padd(img, labels).transpose((2,1,0))
        

        # TODO() delete
        labels = np.ones(self.target_shape[::-1] + [1])
        #print(img.shape, labels.shape)
        return img, labels


    def padd(self, image: np.ndarray, labels: np.ndarray, ignore_label = 30):
        

        h, w, c = image.shape
        
        if w > h:
            image = np.transpose(image, (1, 0, 2))
            labels = labels.T
            h, w, c = image.shape

        result_labels = np.full(self.target_shape + [1], ignore_label, dtype=image.dtype)
        result = np.zeros(self.target_shape + [c],dtype=image.dtype)
        # Calculate the starting position to place the image in the center
        start_h = (self.target_shape[0] - h) // 2
        start_w = (self.target_shape[1] - w) // 2

        # Copy the image to the center of the result array
        print(image.shape, labels.shape)
        print(labels.reshape((h,w,1)).shape, [h,w,c])
        result[start_h:start_h + h, start_w:start_w + w, :] = image
        result_labels[start_h:start_h + h, start_w:start_w + w, :] = labels.reshape((h,w,1))

        return result
            





