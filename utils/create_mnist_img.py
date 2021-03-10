import pandas as pd
from PIL import Image
from sklearn.datasets import fetch_openml

from tqdm import tqdm

print('Downloading MNIST...')
mnist = fetch_openml('mnist_784', as_frame=True)
X, y = mnist["data"], mnist["target"]
y = pd.DataFrame(y)

list_names = []

for i in tqdm(range(len(X))):
    im = Image.fromarray(X.iloc[i].values.reshape(28, 28))
    im = im.convert("L")
    im.save(f"dataset/raw/mnist_784_{i}.jpeg")
    list_names.append(f"mnist_784_{i}.jpeg")

y.insert(0, "Name", list_names, True)

with open("dataset/raw/name_labels.csv", 'w') as f:
    f.write(y.to_csv(index=None))
