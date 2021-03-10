import argparse
import training

TRAIN_STEP = 10  # used for snapshot, and adjust learning rate
RAMP_MULT = 5

################################################################################
#   Argparse                                                                   #
################################################################################

parser = argparse.ArgumentParser(description='Semi-supervised MNIST training')

parser.add_argument('--dataset_name', type=str, help='name of the saved dataset to use')
parser.add_argument('--img_mode', type=str, default='L', help='loading method (RGB or L)')

parser.add_argument('--batch_size', type=int, default=100, help='input batch size for training (default: 32)')
parser.add_argument('--shuffle', type=bool, default=True, help='shuffle bool for train dataset (default: True)')
parser.add_argument('--epochs', type=int, default=30, help='number of epochs to train (default: 300)')
parser.add_argument('--ramp_epochs', type=int, default=10, help='number of epochs before unsupervised weight reaches its maximum (default: 50)')
parser.add_argument('--max_weight', type=float, default=20., help='maximum weight for the unsupervised loss (default: 30.)')
parser.add_argument('--alpha', type=float, default=0.6, help='variable for the moving average part (default: 0.7)')

parser.add_argument('--log_interval', type=int, default=10, help='how many batches to wait before logging training status')
parser.add_argument('--no_cuda', default=False, help='disables CUDA training')

args = parser.parse_args()

################################################################################
#   Cuda                                                                       #
################################################################################

args.cuda = not args.no_cuda and torch.cuda.is_available()
if args.cuda:
    cudnn.benchmark = True

if args.cuda:
    kwargs = {'batch_size': args.batch_size, 'shuffle': args.shuffle, 'num_workers': 8, 'pin_memory': True}
else:
    kwargs = {'batch_size': args.batch_size, 'shuffle': args.shuffle}

training.training(args)
