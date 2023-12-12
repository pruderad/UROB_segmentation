from torch.utils.data import DataLoader, Dataset
import pickle   
import numpy as np
import torch

class UROBDataset(Dataset):

    def __init__(self, filenames_file: str, label_mapping: dict = None) -> None:
        with open(filenames_file, 'rb') as file:
            self.filenames = pickle.load(file)

        if label_mapping is not None:
            self.label_mapping = label_mapping
        else:
            label_mapping = {
                2 : 1 # car
            }

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

        labels = torch.from_numpy(labels)
        img = torch.from_numpy(img).permute(3, 1, 2)
        img = 2 * img / 255 - 1

        return img, labels

        







