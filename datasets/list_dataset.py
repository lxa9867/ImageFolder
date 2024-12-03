import torch
from PIL import Image


class ImageListDataset(torch.utils.data.Dataset):
    def __init__(self, image_list, transform=None, return_path=True):
        
        self.transform = transform
        self.image_list = image_list
        self.return_path = return_path

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, idx):
        path = self.image_list[idx]
        img = Image.open(path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        if self.return_path:
            return path, img
        else:
            return img