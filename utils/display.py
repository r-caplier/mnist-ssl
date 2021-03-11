import os
import pathlib
import pickle
import matplotlib.pyplot as plt

ROOT_PATH = os.path.join(pathlib.Path(__file__).parent.absolute())  # Src files path

def show_loss(args):

    with open(os.path.join(args.graphs_path, 'loss.pkl'), 'rb') as f:
        losses = pickle.load(f)

    with open(os.path.join(args.graphs_path, 'sup_loss.pkl'), 'rb') as f:
        sup_losses = pickle.load(f)

    with open(os.path.join(args.graphs_path, 'unsup_loss.pkl'), 'rb') as f:
        unsup_losses = pickle.load(f)

    fig = plt.figure(figsize=(12, 24))

    ax1 = fig.add_subplot(311)
    ax1.set_title('Loss')
    ax1.plot(range(args.epochs), losses)

    ax2 = fig.add_subplot(312)
    ax2.set_title('Supervised Loss')
    ax2.plot(range(args.epochs), sup_losses)

    ax3 = fig.add_subplot(313)
    ax3.set_title('Unsupervised Loss')
    ax3.plot(range(args.epochs), unsup_losses)

    plt.show()
