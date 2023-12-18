from torch.utils.data import DataLoader, Dataset
import pickle   
import numpy as np
import torch
from PIL import ImageOps, Image
from torchvision import transforms
import random


class UROBDataset(Dataset):

    def __init__(self, filenames_file: str, target_img_shape: list,  label_mapping: dict = None, ignore_label: int = 10, p_cutmix: float = 0.5) -> None:
        with open(filenames_file, 'rb') as file:
            self.filenames = pickle.load(file)

        self.label_mapping = label_mapping
        self.p_cutmix = 0.5
        self.target_shape = target_img_shape
        self.ignore_label = ignore_label
        self.transform = transforms.Compose([transforms.Resize(512, interpolation=Image.BICUBIC, antialias=True)])
        
        # get unique labels
        unique_labels_list = [0] + [label_mapping[key] for key in self.label_mapping.keys()]
        self.unique_labels = torch.unique(torch.tensor(unique_labels_list, dtype=int))


    def __len__(self):
        return len(self.filenames)
    
    def get_sample(self, filepath: str):
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

    def __getitem__(self, index):
        filepath = self.filenames[index]

        img, labels = self.get_sample(filepath=filepath)

        if torch.rand(1).item() < self.p_cutmix:
            # do cutmix
            pass
        
        return img, labels


    def fix_sizes(self, img_orig: np.ndarray, labels_orig: np.ndarray):

        #! assuming the labels are resized correctly
        img_pil = Image.fromarray(img_orig)
        transformed_img = np.asarray(self.transform(img_pil))

        img = np.zeros((*self.target_shape, 3), dtype=transformed_img.dtype)
        labels = self.ignore_label * np.ones(self.target_shape, dtype=labels_orig.dtype) 
        start_idx = (self.target_shape[1] - transformed_img.shape[1]) // 2 
        assert start_idx >= 0

        img[:, start_idx: start_idx + transformed_img.shape[1], :] = transformed_img
        labels[:, start_idx: start_idx + transformed_img.shape[1]] = labels_orig

        #print(transformed_img.shape, self.target_shape)
        #print(img.shape, labels.shape)

        return img, labels
    
    def my_cutmix(self, sample_x: torch.tensor, sample_y: torch.tensor):

        # choose random sample to augment with
        aug_sample_path = random.choice(self.filenames)
        aug_sample_x, aug_sample_y = self.get_sample(aug_sample_path)

        # select the random label to augment
        target_label = 




