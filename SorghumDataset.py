import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms
import pandas as pd
import numpy as np
import os
from PIL import Image

PATH = 'data/'
TRAIN_PATH = PATH + 'train_images/'
TEST_PATH = PATH + 'test/'

from utils import apply_clahe


def init_train_table():
    train_data_path = PATH + 'train_cultivar_mapping.csv'
    train_table = pd.read_csv(train_data_path)
    train_table = train_table.dropna()
    train_table['index_cultivar'] = train_table['cultivar'].astype('category')
    train_table['index_cultivar'] = train_table['index_cultivar'].cat.codes

    index_to_cultivar = {index: cultivar for cultivar, index in
                         zip(train_table['cultivar'], train_table['index_cultivar'])
                         }
    return train_table, index_to_cultivar


def create_loaders_mapping(batch_size, training_percentage, train_transformations=[]):
    train_table, index_to_cultivar = init_train_table()
    idx = np.arange(len(train_table))
    np.random.shuffle(idx)
    cut = int(len(train_table) * training_percentage)
    idx_train = idx[0:cut]
    idx_val = idx[cut:]
    print(f"Train size {len(idx_train)}, test size {len(idx_val)} ")

    training_data = SorghumDataset(df=train_table.iloc[idx_train],
                                   transform=train_transformations,
                                   mode='train')
    val_data = SorghumDataset(train_table.iloc[idx_val])
    # Num workers 8 was found to be the fastest
    trainloader = torch.utils.data.DataLoader(training_data, batch_size=batch_size, shuffle=True, num_workers=8)
    valloader = torch.utils.data.DataLoader(val_data, batch_size=batch_size, shuffle=True, num_workers=8)
    return trainloader, valloader, index_to_cultivar


# This is a rather heavy dataset, thus we need a special class to read the images as we encounter them.
# We took the class structure from https://www.kaggle.com/code/pegasos/sorghum-pytorch-lightning-starter-training

class SorghumDataset(Dataset):
    def __init__(self, df=None, transform=[], mode='train'):
        # mode test is not for validation is to use the test data
        if mode == 'test':
            if df is not None:
                raise ValueError("df must be none for test")
            df = pd.DataFrame(os.listdir(TEST_PATH))
            self.labels = df.values
            self.image_path = TEST_PATH + df.values
        elif mode == 'train':
            self.labels = df["index_cultivar"].values
            self.image_path = TRAIN_PATH + df['image'].values
        else:
            raise ValueError("Mode most be train or test")

        self.mode = mode
        if len(transform) == 0:
            trans = [
                transforms.Resize(size=(512, 512)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225]),
            ]
        else:
            trans = transform

        self.transform = transforms.Compose(trans)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        if self.mode == 'test':
            image_name = self.labels[idx][0]
            image_path = self.image_path[idx][0]
        elif self.mode == 'train':
            label = torch.tensor(self.labels[idx], dtype=torch.float32)
            image_path = self.image_path[idx]

        image = Image.open(image_path)
        image = apply_clahe(image)
        image = self.transform(image)

        if self.mode == 'test':
            return {'image': image, 'filename': image_name}
        elif self.mode == 'train':
            return {'image': image, 'target': label}
