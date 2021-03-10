import torch
import torch.nn.functional as F
from torch.autograd import Variable


class TemporalLoss(object):

    def __init__(self, cuda):

        self.cuda = cuda

        if self.cuda:
            self.supervised_loss = Variable(torch.FloatTensor([0.]).cuda(), requires_grad=False)
            self.unsupervised_loss = Variable(torch.FloatTensor([0.]).cuda(), requires_grad=False)
        else:
            self.supervised_loss = Variable(torch.FloatTensor([0.]), requires_grad=False)
            self.unsupervised_loss = Variable(torch.FloatTensor([0.]), requires_grad=False)

    def masked_cross_entropy(self, out, labels):

        # Count number of labeled samples
        true_labels = (labels >= 0)
        non_zero = torch.nonzero(true_labels)
        nb_labels = len(non_zero)

        if nb_labels > 0:
            masked_outputs = torch.index_select(out, 0, non_zero.view(nb_labels))
            masked_labels = labels[true_labels]
            sup_loss = F.cross_entropy(masked_outputs, masked_labels)
        else:
            if self.cuda:
                sup_loss = Variable(torch.FloatTensor([0.]).cuda(), requires_grad=False)
            else:
                sup_loss = Variable(torch.FloatTensor([0.]), requires_grad=False)

        return sup_loss

    def mse_loss(self, out1, out2):

        quad_diff = torch.sum((F.softmax(out1, dim=1) - F.softmax(out2, dim=1)) ** 2)
        unsup_loss = quad_diff / out1.data.nelement()

        return unsup_loss

    def __call__(self, pred, target, labels, w):

        self.supervised_loss = self.masked_cross_entropy(pred, labels)
        self.unsupervised_loss = self.mse_loss(pred, target)

        return self.supervised_loss + w * self.unsupervised_loss, self.supervised_loss, self.unsupervised_loss
