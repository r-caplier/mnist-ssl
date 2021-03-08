import os
import pathlib
import argparse
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torchvision.transforms as transforms
import torch.nn as nn
import torch.optim as optim
import networks
from tqdm import tqdm

import dataloader_v2

ROOT_PATH = os.path.join(pathlib.Path(__file__).parent.absolute())  # Src files path
INPUT_PATH = os.path.join(ROOT_PATH, 'datasets', 'clean')  # dataset.csv files path

TRAIN_STEP = 100  # used for snapshot, and adjust learning rate

# Training settings
parser = argparse.ArgumentParser(description='Semi-supervised MNIST')

parser.add_argument('--dataset_name', type=str, help='name of the saved dataset to use')
parser.add_argument('--img_mode', type=str, help='loading method (RGB or L)')

parser.add_argument('--batch-size', type=int, default=32, metavar='N', help='input batch size for training (default: 32)')
parser.add_argument('--test-batch-size', type=int, default=10, metavar='N', help='input batch size for testing (default: 10)')
parser.add_argument('--epochs', type=int, default=TRAIN_STEP * 3, metavar='N', help='number of epochs to train (default: 300)')

parser.add_argument('--log-dir', default='/logs', help='folder to output model checkpoints')
parser.add_argument('--log-interval', type=int, default=20, metavar='N', help='how many batches to wait before logging training status')
parser.add_argument('--no-cuda', action='store_true', default=False, help='disables CUDA training')

args = parser.parse_args()

args.cuda = not args.no_cuda and torch.cuda.is_available()
if args.cuda:
    cudnn.benchmark = True
kwargs = {'num_workers': 8, 'pin_memory': True} if args.cuda else {}

LOG_DIR = ROOT_PATH + args.log_dir  # Logs path
if not os.path.exists(LOG_DIR):
    os.makedirs(LOG_DIR)

# Dataloader
train_loader = dataloader_v2.DataloaderMNIST(args,
                                             'train',
                                             img_transform=lambda x: x / np.linalg.norm(x))

print('The number of train data: {}'.format(len(train_loader.dataset)))


def main():

    model = mnist_model.MNISTModel()  # Mettre un modèle là
    networks.print_network(model)
    networks.init_weights(model, init_type='normal')

    if args.cuda:
        model.cuda()

    torch.save({'epoch': 0,
                'state_dict': model.state_dict()},
               '{}/checkpoint_{}.pth'.format(LOG_DIR, 0))

    criterion = criterions.pi_loss()  # Définir ça

    optimizer = create_optimizer(model.parameters(), args.lr)

    for epoch in range(1, args.epochs + 1):
        # update the optimizer learning rate
        adjust_learning_rate(optimizer, epoch)

        train(train_loader, model, optimizer, criterion, epoch)


def train(train_loader, model, optimizer, criterion, epoch):
    # switch to train mode
    model.train()

    pbar = tqdm(enumerate(train_loader))

    for batch_idx, (data, target) in pbar:

        if args.cuda:
            data_var, target_var = data.cuda(), target.cuda()

        data_var = torch.transpose(data_var, 1, 0)

        optimizer.zero_grad()

        # compute output
        prediction_target = model.forward(data_var[0])
        prediction_eval = model.forward(data_var[1])

        loss = criterion.forward(prediction_target, prediction_eval, target_var)

        # compute gradient and update weights
        loss.backward(retain_graph=True)
        optimizer.step()

        if batch_idx % args.log_interval == 0:
            pbar.set_description(
                'Train Epoch: {} [{}/{} ({:.0f}%)]. Loss: {:.8f}'.format(
                    epoch, batch_idx * len(data), len(train_loader.dataset),
                    100. * batch_idx / len(train_loader),
                    loss.item()))

    if epoch % TRAIN_STEP == 0:
        torch.save({'epoch': epoch,
                    'state_dict': model.state_dict()},
                   '{}/checkpoint_{}.pth'.format(LOG_DIR, epoch))


def adjust_learning_rate(optimizer, epoch):
    """Sets the learning rate"""
    lr = args.lr * (0.1 ** ((epoch - 1) // TRAIN_STEP))
    print('lr: {}'.format(lr))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def create_optimizer(params, new_lr):
    # setup optimizer
    if args.optimizer == 'sgd':
        optimizer = optim.SGD(params, lr=new_lr,
                              momentum=0.9,
                              weight_decay=0)
    elif args.optimizer == 'adam':
        optimizer = optim.Adam(params, lr=new_lr,
                               weight_decay=args.wd, betas=(args.beta1, 0.999))
    elif args.optimizer == 'adagrad':
        optimizer = optim.Adagrad(params,
                                  lr=new_lr,
                                  lr_decay=args.lr_decay,
                                  weight_decay=args.wd)
    return optimizer


if __name__ == '__main__':
    main()
