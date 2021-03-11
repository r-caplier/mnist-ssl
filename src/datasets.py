import os
import pathlib
import pandas as pd
import numpy as np
import torch.utils.data as data

from PIL import Image
from torch.utils.data.dataloader import DataLoader

ROOT_PATH = pathlib.Path(__file__).resolve().parents[1].absolute()

DATASETS_PATH = os.path.join(ROOT_PATH, 'datasets')  # dataset.csv files path
if not os.path.exists(DATASETS_PATH):
    raise RuntimeError('No data found, please create datasets folder')


class DatasetCIFAR10(data.Dataset):

    """
    Dataset class containing images and corresponding target (the class index, or no label)
    ------
    Args:
        - args: from the argparse, used for data, dataset_name and img_mode
        - test: True or False, wheter or not the dataset is test or train
        - transform: transforms to be applied to all imgs
        - target_transform: same thing for all targets
    Methods:
        len and getitem
    """

    def load_dataset(self, fpath, test):
        # Loads the dataframe containing image names and labels

        df_imgs = pd.read_csv(fpath)

        if not test:
            df_imgs = df_imgs.loc[~df_imgs['Test']][['Name', 'Label']].reset_index(drop=True)
        else:
            df_imgs = df_imgs.loc[df_imgs['Test']][['Name', 'Label']].reset_index(drop=True)

        return df_imgs

    def get_info(self, df_imgs, idx):
        # Grabs the name and label of the given idx

        return os.path.join(self.raw_path, df_imgs.iloc[idx]['Name']), int(df_imgs.iloc[idx]['Label'])

    def img_loader(self, path, mode):
        # Loads the image from its path to a PIL object

        with open(path, 'rb') as f:
            img = Image.open(f)
            if mode == 'L':
                return img.convert('L')  # convert image to grey
            elif mode == 'RGB':
                return img.convert('RGB')  # convert image to rgb image

    def __init__(self, args, test, transform=None, target_transform=None):

        self.clean_path = os.path.join(DATASETS_PATH, args.data, 'clean')
        if not os.path.exists(self.clean_path):
            raise RuntimeError('Please create clean folder and populate it')

        self.raw_path = os.path.join(DATASETS_PATH, args.data, 'raw')
        if not os.path.exists(self.raw_path):
            raise RuntimeError('Please create raw folder and populate it')

        fpath = os.path.join(self.clean_path, args.dataset_name + '.csv')
        if not os.path.exists(fpath):
            raise RuntimeError('Dataset not found')

        self.df_imgs = self.load_dataset(fpath, test)

        labels = self.df_imgs['Label'].unique()
        if not test:
            if -1 in labels:
                self.nb_classes = len(labels) - 1
            else:
                self.nb_classes = len(labels)
            self.percent_labeled = 1 - len(self.df_imgs.loc[self.df_imgs['Label'] == -1]) / len(self.df_imgs)
        else:
            self.nb_classes = len(labels)
            self.percent_labeled = 1.

        self.img_mode = args.img_mode
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.df_imgs)

    def __getitem__(self, idx):

        if idx > len(self.df_imgs):
            raise ValueError(f'Index out of bounds: {idx}')

        path, target = self.get_info(self.df_imgs, idx)

        img = self.img_loader(path, self.img_mode)

        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target


class DatasetMNIST(data.Dataset):

    """
    Dataset class containing images and corresponding target (the digit, or no label)
    ------
    Args:
        - args: from the argparse, used for data, dataset_name and img_mode
        - test: True or False, wheter or not the dataset is test or train
        - transform: transforms to be applied to all imgs
        - target_transform: same thing for all targets
    Methods:
        len and getitem
    """

    def load_dataset(self, fpath, test):
        # Loads the dataframe containing image names and labels

        df_imgs = pd.read_csv(fpath)

        if not test:
            df_imgs = df_imgs.loc[~df_imgs['Test']][['Name', 'Label']].reset_index(drop=True)
        else:
            df_imgs = df_imgs.loc[df_imgs['Test']][['Name', 'Label']].reset_index(drop=True)

        return df_imgs

    def get_info(self, df_imgs, idx):
        # Grabs the name and label of the given idx

        return os.path.join(self.raw_path, df_imgs.iloc[idx]['Name']), int(df_imgs.iloc[idx]['Label'])

    def img_loader(self, path, mode):
        # Loads the image from its path to a PIL object

        with open(path, 'rb') as f:
            img = Image.open(f)
            if mode == 'L':
                return img.convert('L')  # convert image to grey
            elif mode == 'RGB':
                return img.convert('RGB')  # convert image to rgb image

    def __init__(self, args, test, transform=None, target_transform=None):

        self.clean_path = os.path.join(DATASETS_PATH, args.data, 'clean')
        if not os.path.exists(self.clean_path):
            raise RuntimeError('Please create clean folder and populate it')

        self.raw_path = os.path.join(DATASETS_PATH, args.data, 'raw')
        if not os.path.exists(self.raw_path):
            raise RuntimeError('Please create raw folder and populate it')

        fpath = os.path.join(self.clean_path, args.dataset_name + '.csv')
        if not os.path.exists(fpath):
            raise RuntimeError('Dataset not found')

        self.df_imgs = self.load_dataset(fpath, test)

        labels = self.df_imgs['Label'].unique()
        if not test:
            if -1 in labels:
                self.nb_classes = len(labels) - 1
            else:
                self.nb_classes = len(labels)
            self.percent_labeled = 1 - len(self.df_imgs.loc[self.df_imgs['Label'] == -1]) / len(self.df_imgs)
        else:
            self.nb_classes = len(labels)
            self.percent_labeled = 1.

        self.img_mode = args.img_mode
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.df_imgs)

    def __getitem__(self, idx):

        if idx > len(self.df_imgs):
            raise ValueError(f'Index out of bounds: {idx}')

        path, target = self.get_info(self.df_imgs, idx)

        img = self.img_loader(path, self.img_mode)

        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target
