import lightning as L
from torch.utils.data import DataLoader, Dataset
from typing import Optional
from PIL import Image
import torch
from torch.utils.data import Dataset
import os
from torchvision import transforms


class DefectDataModule(L.LightningDataModule):
    def __init__(
        self,
        train_data_dir: str,
        val_data_dir: str,
        test_data_dir: str,
        batch_size: int = 8,
        num_workers: int = 4,
    ):
        super().__init__()
        self.train_data_dir = train_data_dir
        self.val_data_dir = val_data_dir
        self.test_data_dir = test_data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers

    def setup(self, stage: Optional[str] = None):
        if stage == "fit" or stage is None:
            self.train_dataset = ImageDataset(self.train_data_dir)
            self.val_dataset = ImageDataset(self.val_data_dir)
        
        if stage == "test" or stage is None:
            self.test_dataset = ImageDataset(self.test_data_dir)

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=True
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers
        ) 


class ImageDataset(Dataset):
    def __init__(self, data_dir):
        self.data_dir = data_dir
        # Get all jpg files in the directory
        self.image_files = [f for f in os.listdir(data_dir) if f.endswith('.jpg')]
        
        # Define image transformations
        self.transform = transforms.Compose([
            transforms.ToTensor(),  # Converts PIL Image to tensor and scales to [0, 1]
            # transforms.Resize((256, 256)),  # Resize all images to same dimensions
            transforms.Resize((512, 512)),  # Resize all images to same dimensions
            # Add any other transformations you need
        ])

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        # Get image path
        img_path = os.path.join(self.data_dir, self.image_files[idx])
        
        # Load image using PIL
        image = Image.open(img_path).convert('RGB')
        
        # Apply transformations
        image = self.transform(image)

        return image
