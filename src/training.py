################################################################################
#   Libraries                                                                  #
################################################################################

import sys
sys.path.append('../')

from vars import *
import criterions
import models
import datasets
from utils import networks
from torch.autograd import Variable
from torch.utils.data import DataLoader
from tqdm import tqdm
import torch.optim as optim
import torch.nn as nn
import torchvision.transforms as transforms
import torch.backends.cudnn as cudnn
import torch
import numpy as np
import pickle
import argparse
import pathlib
import os

################################################################################
#   Paths                                                                      #
################################################################################

ROOT_PATH = pathlib.Path(__file__).resolve().parents[1].absolute()

CLEAN_PATH = os.path.join(ROOT_PATH, 'datasets', 'clean')  # dataset.csv files path
if not os.path.exists(CLEAN_PATH):
    os.makedirs(CLEAN_PATH)

LOGS_PATH = os.path.join(ROOT_PATH, 'logs')  # Logs path
if not os.path.exists(LOGS_PATH):
    os.makedirs(LOGS_PATH)


def training(args):

    def train(train_loader, model, y_ema, optimizer, criterion, weight_unsupervised_loss, epoch, cuda):

        model.train()

        loss_epoch = 0.
        sup_loss_epoch = 0.
        unsup_loss_epoch = 0.

        outputs = torch.zeros(args.nb_img_train, args.nb_classes).float()
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

    # Creating the dataset and dataloader objects needed for training
    train_dataset = datasets.DatasetMNIST(args,
                                          False,
                                          transform=transforms.Compose([
                                              transforms.ToTensor(),
                                              transforms.Normalize((0.5), (0.5))]))
    train_loader = DataLoader(train_dataset, **kwargs)

    args.nb_img_train = len(train_dataset)
    args.nb_classes = train_dataset.nb_classes
    args.percent_labeled = train_dataset.percent_labeled

    print('The number of train data: {}'.format(len(train_loader.dataset)))

    # Initialize the model and its weights
    model = models.MNISTModel()
    networks.print_network(model)
    networks.init_weights(model, init_type='normal')

    # Initialize the temporal moving average for each target
    y_ema = torch.zeros(args.nb_img_train, args.nb_classes).float()

    if args.cuda:
        model.cuda()
        y_ema = y_ema.cuda()

    # First model checkpoint
    torch.save({'epoch': 0,
                'state_dict': model.state_dict()},
               os.path.join(LOGS_PATH, 'checkpoint_0.pth'))

    # Criterion for calculating the loss of our model
    criterion = criterions.TemporalLoss(args.cuda)

    losses = []
    sup_losses = []
    unsup_losses = []

    optimizer = torch.optim.Adam(model.parameters(), lr=0.002, betas=(0.9, 0.99))

    for epoch in range(1, args.epochs + 1):

        weight_unsupervised_loss = get_weight(epoch, args)
        output, loss, sup_loss, unsup_loss = train(train_loader, model, y_ema, optimizer, criterion, weight_unsupervised_loss, epoch, args.cuda)

        losses.append(loss / int(args.nb_img_train / args.batch_size))
        sup_losses.append(sup_loss / int(args.nb_img_train / args.batch_size))
        unsup_losses.append(unsup_loss / int(args.nb_img_train / args.batch_size))

        print('Updataing moving average...')
        y_ema = update_moving_average(output, y_ema, epoch, args.alpha, args.cuda)

    with open(os.path.join(ROOT_PATH, f'graph/loss_{args.dataset_name}.pkl'), 'wb') as f:
        pickle.dump(losses, f)
    with open(os.path.join(ROOT_PATH, f'graph/sup_loss_{args.dataset_name}.pkl'), 'wb') as f:
        pickle.dump(sup_losses, f)
    with open(os.path.join(ROOT_PATH, f'graph/unsup_loss_{args.dataset_name}.pkl'), 'wb') as f:
        pickle.dump(unsup_losses, f)


def get_weight(epoch, args):  # TODO

    ramp_epochs = args.ramp_epochs
    max_weight = args.max_weight
    percent_labeled = args.percent_labeled

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
