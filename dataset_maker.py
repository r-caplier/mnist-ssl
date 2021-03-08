import os
import pathlib
import pandas as pd
import random
import re

from sklearn.model_selection import train_test_split

ROOT_PATH = '/home/romainc/code/mnist-ssl/dataset'
DATA_PATH = os.path.join(ROOT_PATH, 'raw')
OUTPUT_PATH = os.path.join(ROOT_PATH, 'clean')


def good_parameters(nb_imgs, test_size, nb_labels):

    assert (nb_imgs * test_size).is_integer()
    assert (nb_imgs * (1 - test_size) * nb_labels).is_integer()


def mask_labels(list_labels, nb_labels, nb_class):

    print(len(list_labels))

    id_kept = list(range(len(list_labels)))
    labels_kept = random.sample(id_kept, int(nb_labels * len(list_labels)))

    print(len(labels_kept))

    cnt = 0
    for i in range(len(list_labels)):
        if i not in labels_kept:
            list_labels[i] = -1
            cnt += 1

    print(cnt)

    return list_labels


def make_dataset(size=4000, test_size=0.2, nb_labels=0.1, nb_class=10, name=None):
    """
    Grabs images names and creates a list of training samples and testing
    samples, and saves it in a .csv file

    Args:
        - size (int, default 4000): number of images for the dataset
        - test_size (float, default 0.1): percent of the train size to be used
        as test size
        - nb_labels (float, default 1.): share of the training samples to save
        with label (for semi-supervised learning), set to 1. for supervised
        training
        - name (str, default None): name override for the final file
    """

    good_parameters(size, test_size, nb_labels)

    if name == None:
        # default naming convention
        dataset_name = f"mnist_{str(size)}_{str(test_size)}_{str(nb_labels)}.csv"
        dataset_path = os.path.join(OUTPUT_PATH, dataset_name)
    else:
        dataset_path = os.path.join(OUTPUT_PATH, name)

    if os.path.exists(dataset_path):
        raise NameError('Dataset already exists')
    else:
        with open(dataset_path, 'w') as f:
            f.write('Name,Label,Test\n')

    df_imgs = pd.read_csv(os.path.join(DATA_PATH, 'name_labels.csv')).iloc[:size]

    train_imgs, test_imgs = train_test_split(df_imgs, test_size=test_size, shuffle=True)

    train_imgs.reset_index(drop=True, inplace=True)
    test_imgs.reset_index(drop=True, inplace=True)

    print(train_imgs)
    print(test_imgs)

    train_imgs.loc[:, 'class'] = mask_labels(train_imgs.loc[:, 'class'], nb_labels, nb_class)

    df_data = pd.concat([train_imgs, test_imgs]).reset_index(drop=True)
    df_data.insert(2, "Test", [False for x in range(len(train_imgs))] + [True for x in range(len(test_imgs))])

    with open(dataset_path, 'a') as f:
        f.write(df_data.to_csv(header=False))


make_dataset()
