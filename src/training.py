################################################################################
#   Libraries                                                                  #
################################################################################

import sys
sys.path.append('../')

import os
import pathlib
import argparse
import pickle
import numpy as np

import criterions
from utils import networks

import torch

from torch.autograd import Variable

from tqdm import tqdm


################################################################################
#   Training                                                                   #
################################################################################


def temporal_ensembling_training(train_dataloader, model, optimizer, args):

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
            return max_weight_corr * np.exp(-args.RAMP_MULT * (1 - epoch / ramp_epochs) ** 2)

    def update_moving_average(output, y_ema, epoch, alpha, cuda):

        new_y_ema = torch.zeros(y_ema.shape).float()

        if cuda:
            new_y_ema = new_y_ema.cuda()

        for idx in range(len(y_ema)):
            new_y_ema[idx] = (alpha * y_ema[idx] + (1 - alpha) * output[idx]) / (1 - alpha ** epoch)

        return new_y_ema

    def train(train_dataloader, model, y_ema, optimizer, criterion, weight_unsupervised_loss, epoch, cuda):

        model.train()

        loss_epoch = torch.tensor([0.], requires_grad=False)
        sup_loss_epoch = torch.tensor([0.], requires_grad=False)
        unsup_loss_epoch = torch.tensor([0.], requires_grad=False)

        if args.cuda:
            loss_epoch = loss_epoch.cuda()
            sup_loss_epoch = sup_loss_epoch.cuda()
            unsup_loss_epoch = unsup_loss_epoch.cuda()

        outputs = torch.zeros(args.nb_img_train, args.nb_classes).float()
        w = torch.autograd.Variable(torch.FloatTensor([weight_unsupervised_loss]), requires_grad=False)

        if cuda:
            outputs = outputs.cuda()
            w = w.cuda()

        pbar = tqdm(enumerate(train_dataloader))

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

            loss_epoch += loss.detach()
            sup_loss_epoch += sup_loss
            unsup_loss_epoch += unsup_loss

            if batch_idx % args.log_interval == 0:
                pbar.set_description('Train Epoch: {}/{} [{}/{} ({:.0f}%)]. Loss: {:.8f}'.format(epoch,
                                                                                                args.epochs,
                                                                                                batch_idx * len(data),
                                                                                                args.nb_img_train,
                                                                                                100. * batch_idx / args.nb_batches,
                                                                                                (loss_epoch / (batch_idx + 1)).item()))

            if batch_idx + 1 >= args.nb_batches:
                pbar.set_description('Train Epoch: {}/{} [{}/{} ({:.0f}%)]. Loss: {:.8f}'.format(epoch,
                                                                                                 args.epochs,
                                                                                                 args.nb_img_train,
                                                                                                 args.nb_img_train,
                                                                                                 100.,
                                                                                                 (loss_epoch / args.nb_batches).item()))

        if epoch % args.TRAIN_STEP == 0:
            torch.save({'epoch': epoch,
                        'state_dict': model.state_dict()},
                       os.path.join(args.logs_path, f'checkpoint_{epoch}.pth'))

        return outputs, loss_epoch, sup_loss_epoch, unsup_loss_epoch

    # Initialize the model weights and print its layout
    networks.init_weights(model, init_type='normal')
    networks.print_network(model)

    # Initialize the temporal moving average for each target
    y_ema = torch.zeros(args.nb_img_train, args.nb_classes).float()

    if args.cuda:
        model.cuda()
        y_ema = y_ema.cuda()

    # First model checkpoint
    torch.save({'epoch': 0,
                'state_dict': model.state_dict()},
               os.path.join(args.logs_path, 'checkpoint_0.pth'))

    # Criterion for calculating the loss of our model
    criterion = criterions.TemporalLoss(args.cuda)

    # Keeping track of each epoch losses
    losses = []
    sup_losses = []
    unsup_losses = []

    for epoch in range(1, args.epochs + 1):

        weight_unsupervised_loss = get_weight(epoch, args)
        output, loss, sup_loss, unsup_loss = train(train_dataloader, model, y_ema, optimizer, criterion, weight_unsupervised_loss, epoch, args.cuda)

        losses.append(loss / int(args.nb_img_train / args.batch_size))
        sup_losses.append(sup_loss / int(args.nb_img_train / args.batch_size))
        unsup_losses.append(unsup_loss / int(args.nb_img_train / args.batch_size))

        y_ema = update_moving_average(output, y_ema, epoch, args.alpha, args.cuda)

    with open(os.path.join(args.graphs_path, 'loss.pkl'), 'wb') as f:
        pickle.dump(losses, f)
    with open(os.path.join(args.graphs_path, 'sup_loss.pkl'), 'wb') as f:
        pickle.dump(sup_losses, f)
    with open(os.path.join(args.graphs_path, 'unsup_loss.pkl'), 'wb') as f:
        pickle.dump(unsup_losses, f)
