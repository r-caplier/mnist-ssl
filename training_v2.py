import os
import pathlib
import argparse

import networks

import torch
import torch.backends.cudnn as cudnn
import torchvision.transforms as transforms
import torch.nn as nn
import torch.optim as optim

from tqdm import tqdm
from torch.utils.data import DataLoader

import dataset
import mnist_model
import criterions

ROOT_PATH = os.path.join(pathlib.Path(__file__).parent.absolute())  # Src files path
INPUT_PATH = os.path.join(ROOT_PATH, 'datasets', 'clean')  # dataset.csv files path

TRAIN_STEP = 100  # used for snapshot, and adjust learning rate

# Training settings
parser = argparse.ArgumentParser(description='Semi-supervised MNIST')

parser.add_argument('--dataset_name', type=str, help='name of the saved dataset to use')
parser.add_argument('--img_mode', type=str, default='L', help='loading method (RGB or L)')

parser.add_argument('--batch-size', type=int, default=4, metavar='N', help='input batch size for training (default: 32)')
parser.add_argument('--test-batch-size', type=int, default=10, metavar='N', help='input batch size for testing (default: 10)')
parser.add_argument('--epochs', type=int, default=TRAIN_STEP * 3, metavar='N', help='number of epochs to train (default: 300)')

parser.add_argument('--log-dir', default='/logs', help='folder to output model checkpoints')
parser.add_argument('--log-interval', type=int, default=20, metavar='N', help='how many batches to wait before logging training status')
parser.add_argument('--no-cuda', action='store_true', default=False, help='disables CUDA training')

args = parser.parse_args()

args.cuda = not args.no_cuda and torch.cuda.is_available()
if args.cuda:
    cudnn.benchmark = True
kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}

LOG_DIR = ROOT_PATH + args.log_dir  # Logs path
if not os.path.exists(LOG_DIR):
    os.makedirs(LOG_DIR)

# Dataloader

train_dataset = dataset.DatasetMNIST(args,
                                     False,
                                     transform=transforms.Compose([
                                         transforms.ToTensor(),
                                         transforms.Normalize((0.5), (0.5))]))
train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, **kwargs)

print('The number of train data: {}'.format(len(train_loader.dataset)))


def main():

    model = MNISTModel()
    networks.print_network(model)
    networks.init_weights(model, init_type='normal')

    if args.cuda:
        model.cuda()

    torch.save({'epoch': 0,
                'state_dict': model.state_dict()},
               '{}/checkpoint_{}.pth'.format(LOG_DIR, 0))

    criterion = criterions.TemporalLoss()

    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9, weight_decay=0)

    for epoch in range(1, args.epochs + 1):

        train(train_loader, model, optimizer, criterion, epoch)


def train(train_loader, model, optimizer, criterion, epoch):

    model.train()

    pbar = tqdm(enumerate(train_loader))

    for batch_idx, (data, target) in pbar:

        if args.cuda:
            data, target = data.cuda(), target.cuda()

        optimizer.zero_grad()

        prediction = model.forward(data)

        loss = cr
