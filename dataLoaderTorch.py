import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
import os

class GaussianDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.image_files = sorted([f for f in os.listdir(root_dir) if f.endswith(".png")])
        self.transform = transform or transforms.Compose([
            transforms.ToTensor()
        ])

    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, index):
        img_path = os.path.join(self.root_dir, self.image_files[index])
        image = Image.open(img_path).convert("L")
        if self.transform:
            image = self.transform(image)
        
        return image