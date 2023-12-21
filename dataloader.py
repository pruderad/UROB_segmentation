from torch.utils.data import DataLoader, Dataset
import pickle   
import numpy as np
import torch
from PIL import ImageOps, Image
from torchvision import transforms
import random
import os


class UROBDataset(Dataset):

    def __init__(self, filenames_file: str, target_img_shape: list,  label_mapping: dict = None, ignore_label: int = 10, p_cutmix: float = 0.5, background_transform = None) -> None:
        with open(filenames_file, 'rb') as file:
            self.filenames = pickle.load(file)

        self.label_mapping = label_mapping
        self.p_cutmix = p_cutmix
        self.target_shape = target_img_shape
        self.ignore_label = ignore_label
        self.transform = transforms.Compose([transforms.Resize(512, interpolation=Image.BICUBIC, antialias=True)])
        self.backround_transform = background_transform
        
        # get unique labels
        unique_labels_list = [0] + [label_mapping[key] for key in self.label_mapping.keys()]
        self.unique_labels = torch.unique(torch.tensor(unique_labels_list, dtype=int))


    def __len__(self):
        return len(self.filenames)
    
    def get_sample(self, filepath: str):
        src_dirname = os.path.dirname(os.path.dirname(filepath))
        src_name = os.path.basename(filepath).split('.')[-2]
        img_path = os.path.join(src_dirname, 'rgb', f'{src_name}.jpg')
        if not os.path.isfile(img_path):
            img_path = os.path.join(src_dirname, 'rgb', f'{src_name}.png')
        if not os.path.isfile(img_path):
            print(f'unknown image type: {filepath}')
        seg_path = os.path.join(src_dirname,'seg', f'{src_name}.npy')

        img = np.asanyarray(Image.open(img_path))
        sem_seg = np.load(seg_path)
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

        if torch.rand(1).item() < self.p_cutmix and self.p_cutmix != 0:
            # do cutmix
            img, labels = self.my_cutmix(sample_x=img, sample_y=labels)

        if torch.rand(1).item() > 0.5 and self.backround_transform is not None:
            img = self.augment_background(img=img, labels=labels, background_labels=[0])
        
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
    
    def my_cutmix(self, sample_x: np.ndarray, sample_y: np.ndarray):

        # choose random sample to augment with
        aug_sample_path = random.choice(self.filenames)
        aug_sample_x, aug_sample_y = self.get_sample(aug_sample_path)

        feasible_targets = np.unique(aug_sample_y).tolist()
        feasible_targets.remove(0)
        if self.ignore_label in feasible_targets:
            feasible_targets.remove(self.ignore_label)
        if len(feasible_targets) == 0:
            return sample_x, sample_y

        # select the random label to augment
        target_label = random.choice(feasible_targets)
        target_mask = aug_sample_y == target_label

        # just a test
        sample_x[:, target_mask] = aug_sample_x[:, target_mask]
        sample_y[target_mask] = aug_sample_y[target_mask]

        return sample_x, sample_y
    
    def shift_binary_mask(self, mask, shift_x, shift_y):
    # Get the shape of the original mask
        original_shape = mask.shape

        # Create a new array with zeros of shape (N + abs(shift_y), M + abs(shift_x))
        shifted_mask = np.zeros((original_shape[0], original_shape[1]), dtype=mask.dtype)

        # Determine the region to copy from the original mask
        y_start = max(0, shift_y)
        y_end = min(original_shape[0] + shift_y, shifted_mask.shape[0])

        x_start = max(0, shift_x)
        x_end = min(original_shape[1] + shift_x, shifted_mask.shape[1])

        # Copy the relevant region from the original mask to the shifted mask
        shifted_mask[y_start:y_end, x_start:x_end] = mask[max(0, -shift_y):min(original_shape[0], shifted_mask.shape[0] - shift_y),
                                                        max(0, -shift_x):min(original_shape[1], shifted_mask.shape[1] - shift_x)]

        return shifted_mask

    def augment_background(self, img: np.ndarray, labels: np.ndarray, background_labels: list = [0]):

        img_orig = torch.from_numpy(img.copy())
        mask = torch.ones_like(img_aug[0, :, :], dtype=int) # mask -- 1 where foreground
        for label in background_labels:
            mask[labels == label] = 0
        
        # transform the image
        img_aug = self.background_transform(img_orig)
        
        # bring back the original foreground pixels
        img_aug[:, :, mask] = img_orig[:, :, mask]
        img_aug = img_aug.numpy()

        return img_aug
        


