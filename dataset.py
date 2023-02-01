import os
from typing import Optional

import pandas as pd
import torch
from PIL import Image
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader

import pytorch_lightning as pl
from torchvision.transforms import RandomHorizontalFlip, RandomRotation, GaussianBlur, \
    Compose, Resize, RandomApply, ToTensor, Normalize


class FashionDataset(Dataset):
    """
    Class to facilitate generation of individual data samples for training and validation.
    """
    def __init__(self, data, transforms, gender_dict, master_dict, sub_dict, color_dict, path='data/fashion-dataset/images'):
        self.data = data
        self.transforms = transforms
        self.path = path
        self.gender_dict = gender_dict
        self.master_dict = master_dict
        self.sub_dict = sub_dict
        self.color_dict = color_dict

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, index):
        row = self.data.iloc[index]
        image_id = str(row['id']) + '.jpg'
        master_category = self.master_dict[row['masterCategory']]
        gender = self.gender_dict[row['gender']]
        sub_category = self.sub_dict[row['subCategory']]
        base_colour = self.color_dict[row['baseColour']]

        the_image = Image.open(os.path.join(self.path, image_id))
        transformed_image = self.transforms(the_image)
        return {
            'transformed_image': transformed_image,
            'master_category': torch.tensor(master_category),
            'gender': torch.tensor(gender),
            'sub_category': torch.tensor(sub_category),
            'base_color': torch.tensor(base_colour),
        }


class FashionClassificationDataModule(pl.LightningDataModule):
    """
    Class to dictate how to create data train and validation batches for
    training and assign data augmentation strategy.
    """
    def __init__(self, hparams, gender_dict, master_dict, sub_dict, color_dict, full_data):
        super().__init__()
        self.path = hparams['path']
        self.full_data = full_data
        self.augmentations = [
            RandomHorizontalFlip(0.5),
            RandomRotation(30),
            GaussianBlur(kernel_size=(5, 9), sigma=(0.1, 5)),
        ]
        self.train_transforms = Compose([
            Resize((232, 232)),
            RandomApply(self.augmentations, p=0.6),
            ToTensor(),
            Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        self.gender_dict = gender_dict
        self.master_dict = master_dict
        self.sub_dict = sub_dict
        self.color_dict = color_dict
        self.transforms = Compose([Resize((232, 232)), ToTensor(), Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
        self.train_batch = hparams['batch_size']
        self.val_batch = self.train_batch * 4

    def setup(self, stage: Optional[str] = None):
        if stage == 'fit' or stage is None:
            train_data, self.test_data = train_test_split(self.full_data, test_size=0.05,
                                                     stratify=self.full_data['masterCategory'])
            self.train, self.val = train_test_split(train_data, test_size=0.1, stratify=train_data['masterCategory'])
            self.train_data = FashionDataset(self.train, self.train_transforms, self.gender_dict,
                                             self.master_dict, self.sub_dict, self.color_dict, self.path)
            self.val_data = FashionDataset(self.val, self.transforms, self.gender_dict, self.master_dict,
                                           self.sub_dict, self.color_dict, self.path)

    def train_dataloader(self):
        return DataLoader(self.train_data, batch_size=self.train_batch, shuffle=True, pin_memory=True,
                          num_workers=16)

    def val_dataloader(self):
        return DataLoader(self.val_data, batch_size=self.val_batch, shuffle=False, pin_memory=True,
                          num_workers=16)


def test_fashion_dataset():
    pl.seed_everything(12487)
    hparams = {'path': 'data/fashion-dataset/images/', 'batch_size': 64}
    gender_dict = {'Boys': 0, 'Girls': 1, 'Men': 2, 'Unisex': 3, 'Women': 4}
    master_dict = {'Accessories': 0, 'Apparel': 1, 'Footwear': 2, 'Personal Care': 3}
    sub_dict = {'Accessories': 0, 'Apparel Set': 1, 'Bags': 2, 'Belts': 3, 'Bottomwear': 4, 'Cufflinks': 5, 'Dress': 6,
  'Eyes': 7, 'Eyewear': 8, 'Flip Flops': 9, 'Fragrance': 10, 'Headwear': 11, 'Innerwear': 12, 'Jewellery': 13, 'Lips': 14,
  'Loungewear and Nightwear': 15, 'Makeup': 16, 'Mufflers': 17, 'Nails': 18, 'Sandal': 19, 'Saree': 20, 'Scarves': 21,
  'Shoe Accessories': 22, 'Shoes': 23, 'Skin': 24, 'Skin Care': 25, 'Socks': 26, 'Stoles': 27, 'Ties': 28, 'Topwear': 29,
  'Wallets': 30, 'Watches': 31}
    color_dict = {'Beige': 0, 'Black': 1, 'Blue': 2, 'Bronze': 3, 'Brown': 4, 'Burgundy': 5, 'Charcoal': 6, 'Coffee Brown': 7,
  'Copper': 8, 'Cream': 9, 'Gold': 10, 'Green': 11, 'Grey': 12, 'Grey Melange': 13, 'Khaki': 14, 'Lavender': 15,
  'Magenta': 16, 'Maroon': 17, 'Mauve': 18, 'Metallic': 19, 'Multi': 20, 'Mushroom Brown': 21, 'Mustard': 22, 'Navy Blue': 23,
  'Nude': 24, 'Off White': 25, 'Olive': 26, 'Orange': 27, 'Peach': 28, 'Pink': 29, 'Purple': 30, 'Red': 31, 'Rose': 32,
  'Rust': 33, 'Sea Green': 34, 'Silver': 35, 'Skin': 36, 'Steel': 37, 'Tan': 38, 'Taupe': 39, 'Teal': 40, 'Turquoise Blue': 41,
 'White': 42, 'Yellow': 43}
    print(gender_dict.keys())
    print(sub_dict.keys())
    print(color_dict.keys())
    print(master_dict.keys())
    full_data = pd.read_csv('data/fashion-dataset/final-styles_df.csv')
    sample_module = FashionClassificationDataModule(hparams, gender_dict, master_dict, sub_dict, color_dict, full_data)
    sample_module.setup()
    train_data = sample_module.train_data
    val_data = sample_module.val_data
    print(len(train_data))
    print(len(val_data))
    print(train_data[0])
    print(val_data[0])
    print(val_data[0]['transformed_image'].shape)

if __name__ == '__main__':
    test_fashion_dataset()
