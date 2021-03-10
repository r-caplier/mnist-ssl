import os
import pathlib
import pickle
import matplotlib.pyplot as plt

ROOT_PATH = os.path.join(pathlib.Path(__file__).parent.absolute())  # Src files path
INPUT_PATH = os.path.join(ROOT_PATH, 'graph')  # dataset.csv files path

with open(os.path.join(INPUT_PATH, 'loss_dataset1.csv.pkl'), 'rb') as f:
    losses = pickle.load(f)

with open(os.path.join(INPUT_PATH, 'unsup_loss_dataset1.csv.pkl'), 'rb') as f:
    sup_losses = pickle.load(f)

with open(os.path.join(INPUT_PATH, 'sup_loss_dataset1.csv.pkl'), 'rb') as f:
    unsup_losses = pickle.load(f)

number_epochs = len(losses)

fig = plt.figure(figsize=(12, 24))

ax1 = fig.add_subplot(311)
ax1.set_title('Loss')
ax1.plot(range(number_epochs), losses)

ax2 = fig.add_subplot(312)
ax2.set_title('Supervised Loss')
ax2.plot(range(number_epochs), sup_losses)

ax3 = fig.add_subplot(313)
ax3.set_title('Unsupervised Loss')
ax3.plot(range(number_epochs), unsup_losses)

plt.show()
