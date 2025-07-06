import torchvision.transforms as transforms
from torch.utils.data import Dataset
import torch
from torch.utils.data import DataLoader, Subset,random_split, TensorDataset
import os
from PIL import Image
import numpy as np
from utils import *

transform = transforms.Compose(
    [transforms.ToTensor(),
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

transform_tinyimagenet = transforms.Compose(
    [transforms.ToTensor(),
    transforms.RandomCrop(64, padding=4), 
    transforms.RandomHorizontalFlip(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])


cifar20_structure = {
    0: [4, 30, 55, 72, 95],   1: [1, 32, 67, 73, 91],
    2: [54, 62, 70, 82, 92],  3: [9, 10, 16, 28, 61],
    4: [0, 51, 53, 57, 83],   5: [22, 39, 40, 86, 87],
    6: [5, 20, 25, 84, 94],   7: [6, 7, 14, 18, 24],
    8: [3, 42, 43, 88, 97],   9: [12, 17, 37, 68, 76],
    10: [23, 33, 49, 60, 71], 11: [15, 19, 21, 31, 38],
    12: [34, 63, 64, 66, 75], 13: [26, 45, 77, 79, 99],
    14: [2, 11, 35, 46, 98],  15: [27, 29, 44, 78, 93],
    16: [36, 50, 65, 74, 80], 17: [47, 52, 56, 59, 96],
    18: [8, 13, 48, 58, 90],  19: [41, 69, 81, 85, 89],
}

class CIFAR20Dataset(Dataset):
    def __init__(self, original_dataset):
        self.original_dataset = original_dataset
        self.original_targets = original_dataset.targets
        self.targets = self.get_superclass_labels(original_dataset.targets)

    def get_superclass_labels(self, labels):
        targets = []
        for label in labels:
            for superclass, subclasses in cifar20_structure.items():
                if label in subclasses:
                    targets.append(superclass)
                    break
        return targets

    def __getitem__(self, index):
        image, _ = self.original_dataset[index]
        superclass_label = self.targets[index]
        return image, superclass_label

    def __len__(self):
        return len(self.original_dataset)
    
class TinyImageNetDataset(Dataset):
    def __init__(self, root_dir, split='train', transform=None, class_to_idx=None):
        self.root_dir = root_dir
        self.split = split
        self.transform = transform
        self.image_paths = []
        self.targets = []

        if split == 'train':
            self.class_to_idx = {}
            self._load_train_data()
        elif split == 'val':
            if class_to_idx is None:
                raise ValueError("class_to_idx must be provided for val split.")
            self.class_to_idx = class_to_idx
            self._load_val_data()
        else:
            raise ValueError("split must be 'train' or 'val'")

    def _load_train_data(self):
        train_dir = os.path.join(self.root_dir, 'train')
        classes = sorted(os.listdir(train_dir))
        self.class_to_idx = {cls_name: idx for idx, cls_name in enumerate(classes)}
        
        for cls_name in classes:
            class_dir = os.path.join(train_dir, cls_name, 'images')
            for img_file in os.listdir(class_dir):
                self.image_paths.append(os.path.join(class_dir, img_file))
                self.targets.append(self.class_to_idx[cls_name])

    def _load_val_data(self):
        val_dir = os.path.join(self.root_dir, 'val', 'images')
        val_annotations_path = os.path.join(self.root_dir, 'val', 'val_annotations.txt')

        with open(val_annotations_path, 'r') as f:
            for line in f:
                parts = line.strip().split('\t')
                img_file = parts[0]
                class_name = parts[1]
                if class_name in self.class_to_idx:
                    self.image_paths.append(os.path.join(val_dir, img_file))
                    self.targets.append(self.class_to_idx[class_name])
                else:
                    raise ValueError(f"Class '{class_name}' in val not found in train data.")

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        label = self.targets[idx]
        
        image = Image.open(img_path).convert('RGB') 
        if self.transform:
            image = self.transform(image)
        
        return image, label

def create_teacher_dataloader(trainset, indices_k_unlearn, batch_size):
    train_labels = torch.tensor(trainset.targets)
    
    unlearn_labels = torch.ones(len(indices_k_unlearn))
    unlearn_data = [trainset[i] for i in indices_k_unlearn]
    
    retain_size = int(0.3 * len(trainset))
    retain_indices = torch.randperm(len(trainset))[:retain_size]
    retain_labels = torch.zeros(len(retain_indices))
    retain_data = [trainset[i] for i in retain_indices]

    combined_data = unlearn_data + retain_data
    combined_labels = torch.cat([unlearn_labels, retain_labels])

    combined_x = torch.stack([data[0] for data in combined_data])  
    combined_y = combined_labels  

    combined_dataset = TensorDataset(combined_x, combined_y)
    combined_loader = DataLoader(combined_dataset, batch_size=batch_size, shuffle=True)
    
    return combined_loader

