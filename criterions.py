import torch
import torch.nn.functional as F
from torch.autograd import Variable


class MaskedCrossEntropy(object):

    def __init__(self):

        self.loss = 0

    def __call__(self, out, labels):

        cond = (labels >= 0)
        non_zero = torch.nonzero(cond)
        nb_labels = len(non_zero)
        # check if labeled samples in batch, return 0 if none*

        if nb_labels > 0:
            masked_outputs = torch.index_select(out, 0, non_zero.view(nb_labels))
            masked_labels = labels[cond]
            self.loss = F.cross_entropy(masked_outputs, masked_labels)
            return self.loss

        return Variable(torch.FloatTensor([0.]).cuda(), requires_grad=False)


class MSELoss(object):

    def __init__(self):
        pass

    def __call__(self, out1, out2):
        quad_diff = torch.sum((F.softmax(out1, dim=1) - F.softmax(out2, dim=1)) ** 2)
        return quad_diff / out1.data.nelement()


class TemporalLoss(object):

    def __init__(self):

        self.loss_supervised = MaskedCrossEntropy()
        self.loss_unsupervised = MSELoss()

    def __call__(self, out1, out2, labels, w):

        sup_loss = self.loss_supervised(out1, labels)
        unsup_loss = self.loss_unsupervised(out1, out2)
        return sup_loss + w * unsup_loss, sup_loss, unsup_loss
