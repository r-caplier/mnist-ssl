import os
import pathlib
import pandas as pd
import numpy as np
import torch.utils.data as data

from PIL import Image
from torch.utils.data.dataloader import DataLoader
from torch.utils.data.sampler import RandomSampler, SequentialSampler

ROOT_PATH = os.path.join(pathlib.Path(__file__).parent.absolute(), 'dataset')
DATA_PATH = os.path.join(ROOT_PATH, 'raw')
INPUT_PATH = os.path.join(ROOT_PATH, 'clean')


def load_dataset(fpath, test):

    df_imgs = pd.read_csv(fpath)

    if not test:
        df_imgs = df_imgs.loc[~df_imgs['Test']][['Name', 'Label']].reset_index(drop=True)
    else:
        df_imgs = df_imgs.loc[df_imgs['Test']][['Name']].reset_index(drop=True)

    return df_imgs


def get_info(df_imgs, idx):

    if len(df_imgs.columns) == 2:
        return os.path.join(DATA_PATH, df_imgs.iloc[idx]['Name']), df_imgs.iloc[idx]['Label']
    else:
        return os.path.join(DATA_PATH, df_imgs.iloc[idx]['Name']), None


def pil_loader(path, mode):
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        img = Image.open(f)
        if mode == 'L':
            return img.convert('L')  # convert image to grey
        elif mode == 'RGB':
            return img.convert('RGB')  # convert image to rgb image


def show_image(dataset_image):
    dataset_image[0].show()


class DatasetMNIST(data.Dataset):

    """
    Dataset class containing images and corresponding target (the digit, or no label)
    ------
    Args:
        - args: from the argparse, used for dataset_name and img_mode
        - test: True or False, wheter or not the dataset is test or train
        - transform: transforms to be applied to all imgs
        - target_transform: same thing for all targets
        - loader: loading method for the images
    """

    def __init__(self, args, test, transform=None, target_transform=None, loader=pil_loader):

        fpath = os.path.join(INPUT_PATH, args.dataset_name)

        self.df_imgs = load_dataset(fpath, test)

        self.img_mode = args.img_mode
        self.transform = transform
        self.target_transform = target_transform
        self.loader = loader

    def __len__(self):
        return len(self.df_imgs)

    def __getitem__(self, idx):

        if idx > len(self.df_imgs):
            raise ValueError(f'Index out of bounds: {idx}')

        path, target = get_info(self.df_imgs, idx)

        img = self.loader(path, self.img_mode)
        print(type(img))

        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target
