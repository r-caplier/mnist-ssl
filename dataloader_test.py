import os
import pathlib
import pandas as pd
import numpy as np

import torch
from torch.utils.data import TensorDataset, DataLoader

from PIL import Image

ROOT_PATH = os.path.join(pathlib.Path(__file__).parent.absolute(), 'dataset')
DATA_PATH = os.path.join(ROOT_PATH, 'raw')
INPUT_PATH = os.path.join(ROOT_PATH, 'clean')


def load_dataset(fpath, mode):

    df_imgs = pd.read_csv(fpath)

    if mode == 'train':
        df_imgs = df_imgs.loc[~df_imgs['Test']][['Name', 'Label']].reset_index(drop=True)
    if mode == 'test':
        df_imgs = df_imgs.loc[df_imgs['Test']][['Name']].reset_index(drop=True)

    return df_imgs


def get_info(df_imgs, idx):
    if len(df_imgs.columns) == 2:
        return df_imgs.iloc[idx]['Name'], df_imgs.iloc[idx]['Label']
    else:
        return df_imgs.iloc[idx]['Name'], None


def pil_loader(path, mode='L'):
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        img = Image.open(f)
        if mode == 'L':
            return np.asarray(img.convert('L'))  # convert image to grey
        elif mode == 'RGB':
            return img.convert('RGB')  # convert image to rgb image
        elif mode == 'HSV':
            return img.convert('HSV')
        # elif mode == 'LAB':
        #     return RGB2Lab(img)


def accimage_loader(path):
    import accimage
    try:
        return accimage.Image(path)
    except IOError:
        # Potentially a decoding problem, fall back to PIL.Image
        return pil_loader(path)


def default_loader(path, mode='L'):
    from torchvision import get_image_backend
    if get_image_backend() == 'accimage':
        return accimage_loader(path)
    else:
        return pil_loader(path, mode)


def get_loader(args, mode, img_transform=None, target_transform=None, loader=default_loader, batch_size=32, shuffle=True):

    fpath = os.path.join(INPUT_PATH, args.dataset_name)
    df_imgs = load_dataset(fpath,  mode)

    if len(df_imgs) == 0:
        raise(RuntimeError("No images in the dataset"))

    list_imgs = []
    list_targets = []

    for index, row in df_imgs.iterrows():

        img = loader(os.path.join(DATA_PATH, row['Name']))
        target = row['Label']

        if len(img.shape) == 2:
            img = np.expand_dims(img, axis=0)
        if img_transform is not None:
            img = img_transform(img)
        if target_transform is not None:
            target = target_transform(target)

        list_imgs.append(np.array([img, img]))
        list_targets.append(target)

    dataset = TensorDataset(torch.FloatTensor(list_imgs), torch.LongTensor(list_targets))

    return DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=8, pin_memory=True)
