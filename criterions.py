import torch


class pi_loss(object):

    def __init__(self):

        self.loss = 0

    def __call__(self, prediction, target):

        loss_supervised = torch.nn.CrossEntropyLoss()
        loss_unsupervised = torch.nn.MSELoss()

        if target != -1:
            self.loss += loss_supervised(prediction[1], target)
        self.loss += loss_unsupervised(prediction[0], prediction[1])

        return self.loss
