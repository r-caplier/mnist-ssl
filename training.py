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

import dataloader
import mnist_model
import criterions

TRAIN_STEP = 100  # used for snapshot, and adjust learning rate
normalize = transforms.Normalize((0.5), (0.5))

# Training settings
parser = argparse.ArgumentParser(description='PyTorch Classifier CGI vs NI')

parser.add_argument('--dataset_name', type=str,
                    help='name of the saved dataset to use')
parser.add_argument('--input_nc', type=int, default=3,
                    help='# of input image channels')
parser.add_argument('--img_mode', type=str, default='RGB',
                    help='chooses how image are loaded')

parser.add_argument('--batch-size', type=int, default=32, metavar='N',
                    help='input batch size for training (default: 32)')
parser.add_argument('--test-batch-size', type=int, default=10, metavar='N',
                    help='input batch size for testing (default: 10)')
parser.add_argument('--epochs', type=int, default=TRAIN_STEP * 3, metavar='N',
                    help='number of epochs to train (default: 300)')
parser.add_argument('--lr', type=float, default=0.001, metavar='LR',
                    help='learning rate (default: 0.001)')
parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                    help='SGD momentum (default: 0.9)')
parser.add_argument('--wd', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)')
parser.add_argument('--optimizer', default='sgd', type=str,
                    metavar='OPT', help='The optimizer to use (default: sgd)')

parser.add_argument('--log-dir', default='/logs',
                    help='folder to output model checkpoints')
parser.add_argument('--log-interval', type=int, default=20, metavar='N',
                    help='how many batches to wait before logging training status')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')

args = parser.parse_args()

args.cuda = not args.no_cuda and torch.cuda.is_available()
if args.cuda:
    cudnn.benchmark = True
kwargs = {'num_workers': 8, 'pin_memory': True} if args.cuda else {}

ROOT_PATH = os.path.join(pathlib.Path(__file__).parent.absolute())
INPUT_PATH = os.path.join(ROOT_PATH, 'datasets', 'clean')

LOG_DIR = ROOT_PATH + args.log_dir
if not os.path.exists(LOG_DIR):
    os.makedirs(LOG_DIR)

# dataset_path = os.path.join(INPUT_PATH, dataset_name)
train_loader = dataloader.DataLoaderImages(
    dataloader.DatasetImages(args,
                             'train',
                             transforms.Compose([
                                 transforms.RandomHorizontalFlip(),
                                 transforms.ToTensor(),
                                 normalize
                             ])),
    batch_size=args.batch_size, shuffle=True, **kwargs
)

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
        data = torch.stack(data)
        data = torch.transpose(data, 1, 0)

        if args.cuda:
            data_var, target_var = data.cuda(), target.cuda()

        print(data_var[0].shape)
        print(target)

        # compute output
        prediction = model(data_var)

        loss = criterion(prediction, target_var)

        # compute gradient and update weights
        optimizer.zero_grad()
        loss.backward()
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
