# mnist-ssl

This project aims to compare multiple semi-supervised learning methods on the well-known MNIST dataset (http://yann.lecun.com/exdb/mnist/)

## Setup

1. To install all required libraries for this project, run the following command at the root of the project:
```
pip install -r requirements.txt
```

2. You now need to download the MNIST images, and create a dataset (a split of images in train and test set, while associating all images with their respective label). To do so, run the following commands, still at the root:
```
python utils/MNIST/create_imgs.py
python utils/MNIST/make_dataset.py --data MNIST --dataset_name dataset_full
```

3. You now have all the required tools to train a model!

## Training

1. To run the training, you first need to choose one training method. The available methods are the following:
- TemporalEnsembling

2. The training can be started by running the following commands (don't forget to replace the training method in the command, without the <>):
```
cd src
python main --data MNIST --dataset_name dataset_full --method <Insert your chosen method here> --test False
```
If your computer doesn't support cuda training, run the following command instead:
```
python src/main --data MNIST --dataset_name dataset_full --method <Insert your chosen method here> --no_cuda True
```

3. After a while, your model will be trained. The model checkpoints are saved in the checkpoint folder, and the loss evolution over all epochs is saved in .pkl format in the graphs folder.

## Testing

WIP
