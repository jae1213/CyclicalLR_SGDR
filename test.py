# pytorch
import torch
import torchvision
import torchvision.transforms as transforms
import torchvision.models as models
import torch.nn as nn
from torch import optim
# end of pytorch import

# visualization
import matplotlib.pyplot as plt
# progress bar
from tqdm import tqdm
# argument parser
import argparse
# math.ceil
import math
# learning rate schedulers
from LRscheduler import  CyclicalLR, SGDRLR, FinderLR

# argument parser
parser = argparse.ArgumentParser()
parser.add_argument('--gpu', type=bool, default=True, required=False)
parser.add_argument('--gpu_id', type=int, default=0, required=False)
parser.add_argument('--batch_size', type=int, default=100, required=False)
parser.add_argument('--num_workers', type=int, default=2, required=False)
parser.add_argument('--epochs', type=int, default=1, required=False)
parser.add_argument('--init_lr', type=float, default=1e-5, required=False)
parser.add_argument('--max_lr', type=float, default=1e-3, required=False)
parser.add_argument('--min_lr', type=float, default=1e-5, required=False)
parser.add_argument('--epochs_per_cycle', type=float, default=0.2, required=False)
parser.add_argument('--lr_scheduler', type=str, default='SGDR', choices=['Cyclical', 'SGDR', 'LRFinder'],required=False)
args = parser.parse_args()

is_gpu = args.gpu
gpu_id = args.gpu_id
batch_size = args.batch_size
num_workers= args.num_workers
num_epochs = args.epochs
init_lr = args.init_lr
max_lr = args.max_lr
min_lr = args.min_lr
lr_scheduler = args.lr_scheduler
# end of argument parser


# load CIFAR-10 dataset
def load_dataset():

    transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))]
    )

    data_train = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    loader_train = torch.utils.data.DataLoader(data_train, batch_size=batch_size, shuffle=True, num_workers=num_workers)

    return loader_train

# build mobilenet v2 for image classification on CIFAR-10
def build_net():
    mobilenet = models.mobilenet_v2(pretrained=True)
    model_fm = mobilenet.classifier[1].in_features
    mobilenet.classifier = nn.Sequential(
        nn.Linear(model_fm, 10),
        nn.Softmax(dim=1)
    )

    if is_gpu:
        mobilenet = mobilenet.cuda(gpu_id)

    return mobilenet

# training
def train():
    loader_train = load_dataset()
    mobilenet = build_net()
    loss_func = nn.CrossEntropyLoss()
    optm = optim.Adam(mobilenet.parameters(), lr=init_lr)

    steps_per_epoch = len(loader_train)
    # calc. steps per cycle
    step_size = math.ceil(steps_per_epoch * args.epochs_per_cycle)
    # total step size
    progress_bar = tqdm(total=(steps_per_epoch * num_epochs))

    if lr_scheduler == 'Cyclical':
        lr_schd = CyclicalLR(optm, step_size=math.ceil(step_size / 2))
    elif lr_scheduler == 'SGDR':
        lr_schd = SGDRLR(optm, step_size=step_size)
    elif lr_scheduler == 'LRFinder':
        lr_schd = FinderLR(optm, steps_per_epoch=steps_per_epoch, epochs=num_epochs)

    mobilenet.train(True)

    for epoch in range(num_epochs):

        for step, data_batch in enumerate(loader_train, 0):
            inputs, labels = data_batch

            optm.zero_grad()

            if is_gpu:
                inputs = inputs.cuda(gpu_id)
                labels = labels.cuda(gpu_id)
                logits = mobilenet(inputs.cuda())
                loss = loss_func(logits, labels.cuda())
            else:
                logits = mobilenet(inputs)
                loss = loss_func(logits, labels)



            loss.backward()
            optm.step()
            lr_schd.step()
            lr_schd.append_loss_history(loss.item())

            if step % 10 == 0:
                progress_bar.update(10)


    return lr_schd

# for visualization
def show_lr(lr_schd):
    plt.figure(0)
    plt.plot(lr_schd.history['iterations'], lr_schd.history['lr'])
    plt.xlabel('Iteration')
    plt.ylabel('Learning rate')
    plt.draw()

def show_loss(lr_schd):
    plt.figure(1)
    # lr_scheduler.step() is called once more after training
    plt.plot(lr_schd.history['iterations'][:-1], lr_schd.history['loss'])
    plt.xlabel('Iteration')
    plt.ylabel('Loss')
    plt.draw()
# x in Cyclical LR
def show_x(lr_schd):
    plt.figure(2)
    plt.plot(lr_schd.history['iterations'], lr_schd.history['x'])
    plt.xlabel('Iteration')
    plt.ylabel('x')
    plt.draw()

def main():

    scheduler = train()
    show_lr(scheduler)
    show_loss(scheduler)
    if lr_scheduler == 'Cyclical':
        show_x(scheduler)
    plt.show()


if __name__ == '__main__':
    main()


