import pandas as pd
from PIL import Image
from sklearn.datasets import fetch_openml

mnist = fetch_openml('mnist_784', as_frame=True)
X, y = mnist["data"], mnist["target"]



for i in range(len(X)):
    im = Image.fromarray(X.iloc[i].values.reshape(28,28))
    im = im.convert("L")
    im.save(f"data/raw/mnist_784_{i}.jpeg")

with open("data/raw/labels.csv", 'wb') as f:
    f.write(y.to_csv(header=False))
