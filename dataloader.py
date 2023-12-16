from torch.utils.data import DataLoader, Dataset
import pickle   
import numpy as np
import torch
from PIL import ImageOps, Image
from torchvision import transforms
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

        self.transform = transforms.Compose([transforms.Resize(512, interpolation=Image.BICUBIC, antialias=True)])

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, index):
        filepath = self.filenames[index]
        with np.load(filepath, allow_pickle=True) as file:
            img = file['X']
            sem_seg = file['y']


        labels = np.zeros_like(sem_seg, dtype=float)
        for label_mapping_key in self.label_mapping.keys():
            mask = sem_seg == label_mapping_key
            labels[mask] = self.label_mapping[label_mapping_key]
        
        img, labels = self.fix_sizes(img_orig=img, labels_orig=labels)
        img = 2 * (img / 255) - 1 
        img = np.transpose(img, (2, 0, 1))

        return img, labels


    def fix_sizes(self, img_orig: np.ndarray, labels_orig: np.ndarray):

        #! assuming the labels are resized correctly
        img_pil = Image.fromarray(img_orig)
        transformed_img = np.asarray(self.transform(img_pil))

        img = np.zeros((*self.target_shape, 3), dtype=transformed_img.dtype)
        labels = np.zeros(self.target_shape, dtype=labels_orig.dtype) # TODO() add ignore labels
        start_idx = (self.target_shape[1] - transformed_img.shape[1]) // 2 
        assert start_idx >= 0

        img[:, start_idx: start_idx + transformed_img.shape[1], :] = transformed_img
        labels[:, start_idx: start_idx + transformed_img.shape[1]] = labels_orig

        #print(transformed_img.shape, self.target_shape)
        #print(img.shape, labels.shape)

        return img, labels





