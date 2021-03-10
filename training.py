import os
import pathlib
import argparse
import pickle

import networks

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torchvision.transforms as transforms
import torch.nn as nn
import torch.optim as optim

from tqdm import tqdm
from torch.utils.data import DataLoader
from torch.autograd import Variable

import dataset
import mnist_model
import criterions

ROOT_PATH = os.path.join(pathlib.Path(__file__).parent.absolute())  # Src files path
INPUT_PATH = os.path.join(ROOT_PATH, 'datasets', 'clean')  # dataset.csv files path

TRAIN_STEP = 20  # used for snapshot, and adjust learning rate
RAMP_MULT = 5

# Training settings
parser = argparse.ArgumentParser(description='Semi-supervised MNIST')

parser.add_argument('--dataset_name', type=str, help='name of the saved dataset to use')
parser.add_argument('--img_mode', type=str, default='L', help='loading method (RGB or L)')

parser.add_argument('--batch_size', type=int, default=100, metavar='N', help='input batch size for training (default: 32)')
parser.add_argument('--epochs', type=int, default=TRAIN_STEP * 3, metavar='N', help='number of epochs to train (default: 300)')
parser.add_argument('--ramp_epochs', type=int, default=int(TRAIN_STEP * 0.5), metavar='N', help='number of epochs before unsupervised weight reaches its maximum (default: 50)')
parser.add_argument('--max_weight', type=float, default=20., metavar='N', help='maximum weight for the unsupervised loss (default: 30.)')
parser.add_argument('--alpha', type=float, default=0.6, metavar='N', help='variable for the moving average part (default: 0.7)')

parser.add_argument('--log_dir', default='/logs', help='folder to output model checkpoints')
parser.add_argument('--log_interval', type=int, default=10, metavar='N', help='how many batches to wait before logging training status')
parser.add_argument('--no_cuda', action='store_true', default=False, help='disables CUDA training')

args = parser.parse_args()

args.cuda = not args.no_cuda and torch.cuda.is_available()
if args.cuda:
    cudnn.benchmark = True
kwargs = {'num_workers': 8, 'pin_memory': True} if args.cuda else {}

LOG_DIR = ROOT_PATH + args.log_dir  # Logs path
if not os.path.exists(LOG_DIR):
    os.makedirs(LOG_DIR)


def main():

    train_dataset = dataset.DatasetMNIST(args,
                                         False,
                                         transform=transforms.Compose([
                                             transforms.ToTensor(),
                                             transforms.Normalize((0.5), (0.5))]))
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, **kwargs)

    print('The number of train data: {}'.format(len(train_loader.dataset)))

    nb_train = len(train_dataset)
    nb_classes = train_dataset.nb_classes
    percent_labeled = train_dataset.percent_labeled

    model = mnist_model.MNISTModel()
    networks.print_network(model)
    networks.init_weights(model, init_type='normal')
    y_ema = torch.zeros(nb_train, nb_classes).float()  # Temporal moving average

    if args.cuda:
        model.cuda()
        y_ema = y_ema.cuda()

    torch.save({'epoch': 0,
                'state_dict': model.state_dict()},
               '{}/checkpoint_{}.pth'.format(LOG_DIR, 0))

    criterion = criterions.TemporalLoss()

    losses = []
    sup_losses = []
    unsup_losses = []

    optimizer = torch.optim.Adam(model.parameters(), lr=0.002, betas=(0.9, 0.99))

    for epoch in range(1, args.epochs + 1):

        weight_unsupervised_loss = get_weight(epoch, args.ramp_epochs, args.max_weight, percent_labeled)
        output, loss, sup_loss, unsup_loss = train(train_loader, model, y_ema, optimizer, criterion, weight_unsupervised_loss, epoch, args.cuda)

        losses.append(loss / int(nb_train / args.batch_size))
        sup_losses.append(sup_loss / int(nb_train / args.batch_size))
        unsup_losses.append(unsup_loss / int(nb_train / args.batch_size))

        print('Updataing moving average...')
        y_ema = update_moving_average(output, y_ema, epoch, args.alpha, args.cuda)

    with open(os.path.join(ROOT_PATH, f'graph/loss_{args.dataset_name}.pkl'), 'wb') as f:
        pickle.dump(losses, f)
    with open(os.path.join(ROOT_PATH, f'graph/sup_loss_{args.dataset_name}.pkl'), 'wb') as f:
        pickle.dump(sup_losses, f)
    with open(os.path.join(ROOT_PATH, f'graph/unsup_loss_{args.dataset_name}.pkl'), 'wb') as f:
        pickle.dump(unsup_losses, f)


def train(train_loader, model, y_ema, optimizer, criterion, weight_unsupervised_loss, epoch, cuda):

    model.train()

    loss_epoch = 0.
    sup_loss_epoch = 0.
    unsup_loss_epoch = 0.

    outputs = torch.zeros(nb_train, nb_classes).float()
    w = torch.autograd.Variable(torch.FloatTensor([weight_unsupervised_loss]), requires_grad=False)

    if cuda:
        outputs = outputs.cuda()
        w = w.cuda()

    pbar = tqdm(enumerate(train_loader))

    for batch_idx, (data, target) in pbar:

        if cuda:
            data, target = data.cuda(), target.cuda()

        optimizer.zero_grad()

        prediction = model.forward(data)
        y_ema_batch = Variable(y_ema[batch_idx * args.batch_size: (batch_idx + 1) * args.batch_size], requires_grad=False)
        loss, sup_loss, unsup_loss = criterion(prediction, y_ema_batch, target, w)

        outputs[batch_idx * args.batch_size: (batch_idx + 1) * args.batch_size] = prediction.data.clone()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        loss_epoch += loss
        sup_loss_epoch += sup_loss
        unsup_loss_epoch += unsup_loss

        if batch_idx % args.log_interval == 0:
            pbar.set_description(
                'Train Epoch: {} [{}/{} ({:.0f}%)]. Loss: {:.8f}'.format(
                    epoch, batch_idx * len(data), len(train_loader.dataset),
                    100. * batch_idx / len(train_loader),
                    loss.item()))

        if batch_idx + 1 >= len(train_loader.dataset) / args.batch_size:
            pbar.set_description(
                'Train Epoch: {} [{}/{} ({:.0f}%)]. Loss: {:.8f}'.format(
                    epoch, len(train_loader.dataset), len(train_loader.dataset),
                    100.,
                    loss.item()))

    if epoch % TRAIN_STEP == 0:
        torch.save({'epoch': epoch,
                    'state_dict': model.state_dict()},
                   '{}/{}/checkpoint_{}.pth'.format(LOG_DIR, args.dataset_name, epoch))

    return outputs, loss_epoch, sup_loss_epoch, unsup_loss_epoch


def get_weight(epoch, ramp_epochs, max_weight, percent_labeled):  # TODO

    max_weight_corr = max_weight * percent_labeled

    if epoch == 1:
        return 0
    elif epoch >= ramp_epochs:
        return max_weight_corr
    else:
        return max_weight_corr * np.exp(-RAMP_MULT * (1 - epoch / ramp_epochs) ** 2)


def update_moving_average(output, y_ema, epoch, alpha, cuda):

    new_y_ema = torch.zeros(y_ema.shape).float()

    if cuda:
        new_y_ema = new_y_ema.cuda()

    for idx in range(len(y_ema)):
        new_y_ema[idx] = (alpha * y_ema[idx] + (1 - alpha) * output[idx]) / (1 - alpha ** epoch)

    return new_y_ema


if __name__ == '__main__':
    main()
