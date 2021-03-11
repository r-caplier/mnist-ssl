import os
import pathlib
import pandas as pd
import random
import re
import argparse

from sklearn.model_selection import train_test_split

import warnings
warnings.filterwarnings('ignore')

ROOT_PATH = pathlib.Path(__file__).resolve().parents[1].absolute()

NB_CLASSES = 10

parser = argparse.ArgumentParser(description='Dataset maker')

parser.add_argument('--data', type=str, help='data to use')
parser.add_argument('--dataset_name', type=str, help='name of the saved dataset to use')

parser.add_argument('--dataset_size', type=int, default=30000, help='desired number of images in dataset (train and test) (default: 30000)')
parser.add_argument('--test_size', type=float, default=0.2, help='percent of samples to be used for testing (default: 0.2)')
parser.add_argument('--nb_labels', type=float, default=0.1, help='percent of test samples to be labelized (defautl: 0.1)')

args = parser.parse_args()

DATA_PATH = os.path.join(ROOT_PATH, 'datasets', args.data)
if not os.path.exists(DATA_PATH):
    raise RuntimeError('Please create datasets folder and add data to it')

CLEAN_PATH = os.path.join(DATA_PATH, 'clean')
if not os.path.exists(CLEAN_PATH):
    os.mkdir(CLEAN_PATH)

RAW_PATH = os.path.join(DATA_PATH, 'raw')
if not os.path.exists(RAW_PATH):
    raise RuntimeError('Please create raw folder and populate it')


def good_parameters(nb_imgs, test_size, nb_labels):

    assert (nb_imgs * test_size).is_integer()
    assert (nb_imgs * (1 - test_size) * nb_labels).is_integer()


def mask_labels(list_labels, nb_labels, nb_class):

    id_kept = list(range(len(list_labels)))
    labels_kept = random.sample(id_kept, int(nb_labels * len(list_labels)))

    cnt = 0
    for i in range(len(list_labels)):
        if i not in labels_kept:
            list_labels[i] = -1
            cnt += 1

    return list_labels


def make_dataset(name=None, size=30000, test_size=0.2, nb_labels=0.1, nb_class=10):
    """
    Grabs images names and creates a list of training samples and testing
    samples, and saves it in a .csv file

    Args:
        - size (int, default 30000): number of images for the dataset
        - test_size (float, default 0.1): percent of the train size to be used
        as test size
        - nb_labels (float, default 0.1): share of the training samples to save
        with label (for semi-supervised learning), set to 1. for supervised
        training
        - name (str, default None): name override for the final file
    """

    good_parameters(size, test_size, nb_labels)

    if name == None:
        # default naming convention
        dataset_name = f"mnist_{str(size)}_{str(test_size)}_{str(nb_labels)}.csv"
        dataset_path = os.path.join(CLEAN_PATH, dataset_name)
    else:
        dataset_path = os.path.join(CLEAN_PATH, name + '.csv')

    print('Creating dataset...')

    if os.path.exists(dataset_path):
        raise NameError('Dataset already exists')
    else:
        with open(dataset_path, 'w+') as f:
            f.write('Name,Label,Test\n')

    df_imgs = pd.read_csv(os.path.join(RAW_PATH, 'name_labels.csv')).iloc[:size]

    train_imgs, test_imgs = train_test_split(df_imgs, test_size=test_size, shuffle=True)

    train_imgs.reset_index(drop=True, inplace=True)
    test_imgs.reset_index(drop=True, inplace=True)

    train_imgs.loc[:, 'class'] = mask_labels(train_imgs.loc[:, 'class'], nb_labels, nb_class)

    df_data = pd.concat([train_imgs, test_imgs]).reset_index(drop=True)
    df_data.insert(2, "Test", [False for x in range(len(train_imgs))] + [True for x in range(len(test_imgs))])

    with open(dataset_path, 'a') as f:
        f.write(df_data.to_csv(header=False))

    print('Done!')


if __name__ == '__main__':
    make_dataset(args.dataset_name, args.dataset_size, args.test_size, args.nb_labels, NB_CLASSES)
