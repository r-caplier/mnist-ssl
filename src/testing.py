import os
import pathlib
import argparse
import pickle
import random

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torchvision.transforms as transforms
import torch.nn.functional as F
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

from torch.utils.data import DataLoader
from tqdm import tqdm
from sklearn import metrics


ROOT_PATH = pathlib.Path(__file__).resolve().parents[1].absolute()


def testing_metrics(test_dataloader, model, args):

    if args.cuda:
        model.cuda()

    model.eval()

    oriImageLabel = []
    oriTestLabel = []

    pbar = tqdm(enumerate(test_dataloader))

    with torch.no_grad():
        for batch_idx, (data, target) in pbar:

            if args.cuda:
                data, target = data.cuda(), target.cuda()

            bs, c, h, w = data.size()
            result = model(data.view(-1, c, h, w))
            result = F.softmax(result, dim=1)
            pred = result.data.max(1, keepdim=True)[1]
            oriTestLabel.extend(pred.squeeze().cpu().numpy())
            oriImageLabel.extend(target.data.cpu().numpy())

            if batch_idx % args.log_interval == 0:
                pbar.set_description('Test Batch: {}/{} [{}/{} ({:.0f}%)]'.format(batch_idx + 1,
                                                                                  args.nb_batches_test,
                                                                                  batch_idx * len(data),
                                                                                  args.nb_img_test,
                                                                                  100. * batch_idx / args.nb_batches_test))
            if batch_idx + 1 >= args.nb_batches_test:
                pbar.set_description('Test Batch: {}/{} [{}/{} ({:.0f}%)]'.format(args.nb_batches_test,
                                                                          args.nb_batches_test,
                                                                          args.nb_img_test,
                                                                          args.nb_img_test,
                                                                          100.))


    print(metrics.classification_report(oriImageLabel, oriTestLabel, digits=3))

def testing_display(test_dataloader, model, args):

    nb_imgs_to_check = 9
    fig = plt.figure(figsize=(12, 12))

    id_to_check = random.sample(range(args.nb_img_test), nb_imgs_to_check)
    subplot_id = 1

    for i in id_to_check:

        img, target = test_dataloader.dataset[i]
        if args.cuda:
            img = img.cuda()

        result = model(torch.unsqueeze(img, 0))
        result = F.softmax(result, dim=1)
        pred_label = result.data.max(1, keepdim=True)[1]

        img = img.cpu().numpy()
        if img.shape[1] == img.shape[2]:
            img = np.transpose(img, (1, 2, 0))  # Edge case can be annoying
        if np.amin(img) < 0:
            img = ((img / 2 + 0.5) * 255).astype(np.uint8)

        ax = fig.add_subplot(3, 3, subplot_id)
        if args.img_mode == 'L':
            ax.imshow(img, cmap='gray_r')
        elif args.img_mode == 'RGB':
            ax.imshow(img)
        ax.set_title(f'Prediction/True label: {pred_label.squeeze().cpu().numpy()}/{target}')
        ax.axis('off')

        subplot_id += 1

    plt.show()
