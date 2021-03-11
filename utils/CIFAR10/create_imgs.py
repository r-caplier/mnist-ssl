import os
import pathlib
import pickle
import pandas as pd

from PIL import Image
from tensorflow.keras.datasets import cifar10

from tqdm import tqdm

import warnings
warnings.filterwarnings('ignore')

ROOT_PATH = pathlib.Path(__file__).resolve().parents[2].absolute()

RAW_PATH = os.path.join(ROOT_PATH, 'datasets', 'CIFAR10', 'raw')
if not os.path.exists(RAW_PATH):
    os.makedirs(RAW_PATH)

print('Downloading CIFAR...')
(X_train, y_train), (X_test, y_test) = cifar10.load_data()

print(X_train[0].shape)

list_names = []
list_labels = []

print('Creating images...')

nb_imgs_train = len(X_train)
nb_imgs_test = len(X_test)

for i in tqdm(range(nb_imgs_train)):
    im = Image.fromarray(X_train[i])
    im = im.convert("RGB")
    im.save(os.path.join(RAW_PATH, f'cifar10_{i}.jpeg'))
    list_names.append(f'cifar10_{i}.jpeg')
    list_labels.append(y_train[i])

for i in tqdm(range(nb_imgs_test)):
    im = Image.fromarray(X_test[i])
    im = im.convert("RGB")
    im.save(os.path.join(RAW_PATH, f'cifar10_{i + nb_imgs_train}.jpeg'))
    list_names.append(f'cifar10_{i + nb_imgs_train}.jpeg')
    list_labels.append(y_test[i])

print('Creating dataframe...')
df_imgs = pd.DataFrame(columns=['Name', 'Label', 'Test'])
df_imgs['Name'] = list_names
df_imgs['Label'] = list_labels
df_imgs['Test'] = [False for i in range(nb_imgs_train)] + [True for i in range(nb_imgs_test)]

with open(os.path.join(RAW_PATH, 'name_labels.csv'), 'w+') as f:
    f.write(df_imgs.to_csv(index=None))
