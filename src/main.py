import os
import pathlib
import argparse

import training
import models

from torch.optim import Adam
from torch.utils.data import DataLoader


################################################################################
#   Paths and variables                                                        #
################################################################################

TRAIN_STEP = 10  # used for snapshot, and adjust learning rate
RAMP_MULT = 5

ROOT_PATH = pathlib.Path(__file__).resolve().parents[1].absolute()

DATASETS_PATH = os.path.join(ROOT_PATH, 'datasets')  # dataset.csv files path
if not os.path.exists(DATASETS_PATH):
    os.makedirs(CLEAN_PATH)

LOGS_PATH = os.path.join(ROOT_PATH, 'logs')  # Logs path
if not os.path.exists(LOGS_PATH):
    os.makedirs(LOGS_PATH)

################################################################################
#   Argparse                                                                   #
################################################################################

parser = argparse.ArgumentParser(description='Semi-supervised MNIST training')

parser.add_argument('--data', type='str', help='data to use')
parser.add_argument('--dataset_name', type=str, help='name of the saved dataset to use')
parser.add_argument('--img_mode', type=str, default='L', help='loading method (RGB or L)')

parser.add_argument('--method', type='str', default='TemporalEnsembling', help='training method')
parser.add_argument('--opitimizer', type='str', default='Adam', help='optimizer to use')

parser.add_argument('--batch_size', type=int, default=100, help='input batch size for training (default: 32)')
parser.add_argument('--shuffle', type=bool, default=True, help='shuffle bool for train dataset (default: True)')
parser.add_argument('--epochs', type=int, default=30, help='number of epochs to train (default: 300)')
parser.add_argument('--ramp_epochs', type=int, default=10, help='number of epochs before unsupervised weight reaches its maximum (default: 50)')
parser.add_argument('--max_weight', type=float, default=20., help='maximum weight for the unsupervised loss (default: 30.)')
parser.add_argument('--alpha', type=float, default=0.6, help='variable for the moving average part (default: 0.7)')

parser.add_argument('--log_interval', type=int, default=10, help='how many batches to wait before logging training status')
parser.add_argument('--no_cuda', default=False, help='disables CUDA training')

args = parser.parse_args()

################################################################################
#   Cuda                                                                       #
################################################################################

args.cuda = not args.no_cuda and torch.cuda.is_available()
if args.cuda:
    cudnn.benchmark = True

if args.cuda:
    kwargs = {'batch_size': args.batch_size, 'shuffle': args.shuffle, 'num_workers': 8, 'pin_memory': True}
else:
    kwargs = {'batch_size': args.batch_size, 'shuffle': args.shuffle}

################################################################################
#   Training                                                                   #
################################################################################

def main():

    print('Creating dataset...')
    if args.data == 'MNIST':
        train_dataset_transforms = transforms.Compose([transforms.ToTensor(),
                                                       transforms.Normalize((0.5), (0.5))]))
        train_dataset = datasets.DatasetMNIST(args,
                                              False,
                                              transform=train_dataset_transforms)
        model = models.MNISTModel()
    if args.data == 'CGvsNI':
        train_dataset_transforms = None  #TODO!

    train_dataloder = DataLoader(train_dataset, **kwargs)

    if args.optimizer == 'Adam':
        optimizer = Adam(model.parameters(), lr=0.002, betas=(0.9, 0.99))
    if args.optimizer == 'SGD':
        opitimizer = None  # TODO!

    print('Starting training...')
    if args.method == 'TemporalEnsembling':
        training.temporal_ensembling_training(train_dataloder, model, optimizer, args)

    print('Training done!')


if __name__ == '__main__':
    main()
