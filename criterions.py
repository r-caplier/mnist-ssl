import torch


class pi_loss(object):

    def __init__(self):

        self.loss = 0

    def __call__(self, prediction_target, prediction_eval, target):

        loss_supervised = torch.nn.CrossEntropyLoss()
        loss_unsupervised = torch.nn.MSELoss()

        if target != -1:
            self.loss += loss_supervised(prediction_eval, target)
        self.loss += loss_unsupervised(prediction_target, prediction_eval)

        return self.loss
