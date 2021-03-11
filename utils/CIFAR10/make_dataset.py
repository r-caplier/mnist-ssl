import os
import pathlib
import pandas as pd
import random
import re
import argparse

import warnings
warnings.filterwarnings('ignore')

ROOT_PATH = pathlib.Path(__file__).resolve().parents[2].absolute()

parser = argparse.ArgumentParser(description='Dataset maker')

parser.add_argument('--dataset_name', type=str, help='name of the saved dataset to use')

parser.add_argument('--nb_labels', type=float, default=0.1, help='percent of test samples to be labelized (defautl: 0.1)')

args = parser.parse_args()

DATA_PATH = os.path.join(ROOT_PATH, 'datasets', 'CIFAR10')
if not os.path.exists(DATA_PATH):
    raise RuntimeError('Please create datasets folder and add data to it')

CLEAN_PATH = os.path.join(DATA_PATH, 'clean')
if not os.path.exists(CLEAN_PATH):
    os.makedirs(CLEAN_PATH)

RAW_PATH = os.path.join(DATA_PATH, 'raw')
if not os.path.exists(RAW_PATH):
    raise RuntimeError('Please create raw folder and populate it')


def good_parameters(nb_imgs_train, nb_labels):

    assert (nb_imgs_train * nb_labels).is_integer()


def mask_labels(list_labels, nb_labels):

    id_kept = list(range(len(list_labels)))
    labels_kept = random.sample(id_kept, int(nb_labels * len(list_labels)))

    cnt = 0
    for i in range(len(list_labels)):
        if i not in labels_kept:
            list_labels[i] = -1
            cnt += 1
        else:
            list_labels[i] = list_labels[i]

    return list_labels


def make_dataset(nb_labels, name=None):
    """
    Grabs images names and creates a list of training samples and testing
    samples, and saves it in a .csv file

    Args:
        - nb_labels (float, default 0.1): share of the training samples to save
        with label (for semi-supervised learning), set to 1. for supervised
        training
        - name (str, default None): name override for the final file
    """

    if name == None:
        # default naming convention
        dataset_name = f"cifar10_{str(nb_labels)}.csv"
        dataset_path = os.path.join(CLEAN_PATH, dataset_name)
    else:
        dataset_path = os.path.join(CLEAN_PATH, name + '.csv')

    print('Creating dataset...')

    df_imgs = pd.read_csv(os.path.join(RAW_PATH, 'name_labels.csv'))
    df_imgs['Label'] = df_imgs['Label'].apply(lambda x: int(x[1]))

    dataset_size = len(df_imgs)
    good_parameters(dataset_size, nb_labels)

    if os.path.exists(dataset_path):
        raise NameError('Dataset already exists')
    else:
        with open(dataset_path, 'w+') as f:
            f.write('Name,Label,Test\n')

    train_imgs = df_imgs.loc[~df_imgs['Test']]
    test_imgs = df_imgs.loc[df_imgs['Test']]

    train_imgs.reset_index(drop=True, inplace=True)
    test_imgs.reset_index(drop=True, inplace=True)


    train_imgs.loc[:, 'Label'] = mask_labels(train_imgs.loc[:, 'Label'], nb_labels)

    df_data = pd.concat([train_imgs, test_imgs]).reset_index(drop=True)
    df_data['Label'] = df_data['Label'].astype(int)

    with open(dataset_path, 'a') as f:
        f.write(df_data.to_csv(header=False))

    print('Done!')


if __name__ == '__main__':
    make_dataset(args.nb_labels, args.dataset_name)
