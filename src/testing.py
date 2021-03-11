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

from torch.utils.data import DataLoader
from tqdm import tqdm


ROOT_PATH = pathlib.Path(__file__).resolve().parents[1].absolute()

def testing(test_dataloader, model, args):

    if args.cuda:
        model.cuda()

    model.eval()

    oriImageLabel = []
    oriTestLabel = []

    pbar = tqdm(enumerate(test_loader))

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

    result = np.array(oriImageLabel) == np.array(oriTestLabel)
    print('Accuracy: ', sum(result) / len(test_loader.dataset))

    nb_imgs_to_check = 9
    fig = plt.figure(figsize=(12, 12))

    id_to_check = random.sample(range(len(test_dataset)), nb_imgs_to_check)
    subplot_id = 1

    for i in id_to_check:

        img, target = test_dataset[i]
        if args.cuda:
            img = img.cuda()

        result = model(torch.unsqueeze(img, 0))
        result = F.softmax(result, dim=1)
        pred_label = result.data.max(1, keepdim=True)[1]

        img = img.squeeze().cpu().numpy()
        ax = fig.add_subplot(3, 3, subplot_id)
        ax.imshow(img, cmap='gray_r')
        ax.set_title(f'Prediction/True label: {pred_label.squeeze().cpu().numpy()}/{target}')
        ax.axis('off')

        subplot_id += 1

    plt.show()


def get_latest_log(logs_path):

    list_logs = os.listdir(logs_path)

    latest_log_id = 0
    latest_log_epoch = 0
    latest_log = list_logs[0]

    for i in range(len(list_logs)):
        log_epoch = list_logs[i].split('_')[-1].split('.')[0]
        if int(log_epoch) > latest_log_epoch:
            latest_log = list_logs[i]
            latest_log_epoch = int(log_epoch)
            latest_log_id = i

    return latest_log


if __name__ == '__main__':
    main()
