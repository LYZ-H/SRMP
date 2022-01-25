'''Train CIFAR10 with PyTorch.'''
from __future__ import print_function

import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn

import torchvision.transforms as transforms
import os
import argparse

from torch.autograd import Variable

import sys
import time

current_path = os.path.dirname(os.path.realpath(__file__))
ext_path1 = os.path.abspath(current_path + "/../../")
ext_path2 = os.path.abspath(current_path + "/../../../")
sys.path.extend([ext_path1, ext_path2])

# from models import *
from models.tools import get_model
from tools.cifar10_read import CIFAR10_read
import tools.cifar10_modification as cm

from ps.adapters.pytorch import psPyTorchAdapter

parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
parser.add_argument('--lr', default=0.01, type=float, help='learning rate')
parser.add_argument('--tau',
                    default=10,
                    type=float,
                    help='every tau batches to pull parameters')
parser.add_argument('--resume',
                    '-r',
                    action='store_true',
                    help='resume from checkpoint')
parser.add_argument('--batch-size',
                    type=int,
                    default=64,
                    help='Set the batch size (default: 128)')
parser.add_argument('--config-file',
                    type=str,
                    default='dpwa.yaml',
                    help='Dpwa configuration file')
parser.add_argument('--test-batches',
                    type=int,
                    default=100,
                    help='Batch number per test; defaut: 100')
parser.add_argument('--model-name',
                    type=str,
                    default='MobileNet',
                    help='The trained model name; defaut: MobileNet')
parser.add_argument('--name',
                    type=str,
                    required=True,
                    help="This worker's name within config file")
parser.add_argument('--remove-labels',
                    type=str,
                    default='None',
                    help='Set remove-labels like 1,2,3;default: None')
parser.add_argument(
    '--slice',
    type=str,
    default='None',
    help='Set slice like i/M, means i-th part of M-parts;default: None')
parser.add_argument('--issyn',
                    type=int,
                    default=1,
                    help="0: asynchronous; 1: synchronous")
parser.add_argument('--dataset-location',
                    type=str,
                    default='../../../../data',
                    help='the path of the training data')
parser.add_argument('--enable-cuda',
                    action='store_true',
                    default=False,
                    help='xxx')
parser.add_argument('--alr', default=0, type=float)
parser.add_argument('--fb', default=1, type=int)
parser.add_argument('--state', default='', type=str)

args = parser.parse_args()

batch_size = args.batch_size
print("Using batch_size =", batch_size)

if args.enable_cuda:
    use_cuda = torch.cuda.is_available()
else:
    use_cuda = False

best_acc = 0  # best test accuracy
start_epoch = 0  # start from epoch 0 or last checkpoint epoch

# Data
rm_labels = []
if args.remove_labels != 'None':
    remove_str = args.remove_labels.split(',')
    for item in remove_str:
        rm_labels.append(int(item.strip()))
cm.remove_labels(rm_labels)

isSlice = 0
if args.slice != 'None':
    isSlice = 1
    slice_str = args.slice.strip()
    slice_str = slice_str.strip(' ')
    start_str, step_str = slice_str.split('/')
    start_index = int(start_str) - 1
    step_num = int(step_str)
if isSlice == 1:
    cm.remove_index(start_index, step_num)

print('==> Preparing data..')
transform_train = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

if args.name != 'ps':
    id = int(args.name[1:])
    total_worker = 3
    trainset = CIFAR10_read(root=args.dataset_location,
                            train=True,
                            download=True,
                            transform=transform_train,
                            workers=total_worker,
                            wid=id)

    trainloader = torch.utils.data.DataLoader(trainset,
                                              batch_size=batch_size,
                                              shuffle=True,
                                              num_workers=3)
    # trainloader = torch.utils.data.DataLoader(trainset[(id-1)*int(len(trainset)/total_worker):id*int(len(trainset)/total_worker)], batch_size=batch_size, shuffle=True, num_workers=3)
else:
    trainset = CIFAR10_read(root=args.dataset_location,
                            train=True,
                            download=True,
                            transform=transform_train)
    trainloader = torch.utils.data.DataLoader(trainset,
                                              batch_size=batch_size,
                                              shuffle=True,
                                              num_workers=3)

testset = CIFAR10_read(root=args.dataset_location,
                       train=False,
                       download=True,
                       transform=transform_test)
testloader = torch.utils.data.DataLoader(testset,
                                         batch_size=batch_size,
                                         shuffle=False,
                                         num_workers=3)

classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse',
           'ship', 'truck')

# Model
if args.resume:
    # Load checkpoint.
    print('==> Resuming from checkpoint..')
    assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'
    checkpoint = torch.load('./checkpoint/ckpt.t7')
    net = checkpoint['net']
    best_acc = checkpoint['acc']
    start_epoch = checkpoint['epoch']
else:
    model = get_model(args.model_name)
    net = model()

if use_cuda:
    net.cuda()
    net = torch.nn.DataParallel(net,
                                device_ids=range(torch.cuda.device_count()))
    cudnn.benchmark = True

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(net.parameters(), lr=args.lr, weight_decay=5e-4)
conn = psPyTorchAdapter(net, args.name, args.config_file, args.alr, args.fb)

MOVING_AVG_SIZE = 10
# test_batches = args.test_batches
test_batches = 15

starttime = time.time()
cal_time = 0


def print_parameter():
    for k, param in net.named_parameters():
        print(k, param)
        exit(0)


run_state = True


# Training
def train(epoch):
    global run_state
    global cal_time
    net.train()

    accuracies = []
    losses = []
    loss_mean = 9999
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        if use_cuda:
            inputs, targets = inputs.cuda(), targets.cuda()
        time1 = time.time()
        optimizer.zero_grad()
        inputs, targets = Variable(inputs), Variable(targets)
        outputs = net(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        time2 = time.time()

        conn.grad_send(batch_idx)

        time3 = time.time()
        # Calculate the loss
        losses += [loss.item()]
        loss_mean = np.array(losses[-MOVING_AVG_SIZE:]).mean()

        # Calculate the accuracy
        _, predicted = torch.max(outputs.data, 1)
        total = targets.size(0)
        correct = predicted.eq(targets.data).cpu().sum()
        accuracies += [float(correct) / float(total)]
        accuracy = np.array(accuracies[-MOVING_AVG_SIZE:]).mean() * 100.0
        # Show progress
        progress = "[%s] E%d | B%d | Loss: %.3f | Acc: %.3f%%" % (
            args.name, epoch, batch_idx, loss_mean, accuracy)
        print(progress)
        test_loss, test_acc = None, None
        if batch_idx % 20 == 0:
            test_loss, test_acc = fast_test(epoch)

        time4 = time.time()
        cal_time += (time4 - time3) + (time2 - time1)

        ctime = time.time() - starttime - cal_time
        if batch_idx % 20 == 0:
            f = open("./result/{}.txt".format(args.state), "a")
            f.write("%s,%d,%d,%d,%s,%.3f,%.3f%%\n" %
                    (args.state, epoch, batch_idx, ctime, args.name, test_loss,
                     test_acc))
            f.close()

        # wait updated model parameters from ps
        conn.update_wait()

        net.train()


def fast_test(epoch):
    global best_acc
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(testloader):
        if use_cuda:
            inputs, targets = inputs.cuda(), targets.cuda()
        outputs = net(inputs)
        loss = criterion(outputs, targets)

        test_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += targets.size(0)
        correct += predicted.eq(targets.data).cpu().sum()
        if batch_idx > test_batches:
            break
        progress = "[%s] E%d | B%d | Loss: %.3f | Acc: %.3f%% (%d/%d)" % \
                   (args.name, epoch, batch_idx, test_loss / (batch_idx + 1), 100. * correct / total, correct, total)
        print(progress)
    return test_loss / (batch_idx + 1), 100. * correct / total


def test(epoch):
    global best_acc
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(testloader):
        if use_cuda:
            inputs, targets = inputs.cuda(), targets.cuda()
        outputs = net(inputs)
        loss = criterion(outputs, targets)

        test_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += targets.size(0)
        correct += predicted.eq(targets.data).cpu().sum()

        progress = "[%s] E%d | B%d | Loss: %.3f | Acc: %.3f%% (%d/%d)" % \
                   (args.name, epoch, batch_idx, test_loss / (batch_idx + 1), 100. * correct / total, correct, total)
        print(progress)

    # Save checkpoint.
    acc = 100. * correct / total
    if acc > best_acc:
        print('Saving..')
        state = {
            'net': net.module if use_cuda else net,
            'acc': acc,
            'epoch': epoch,
        }
        if not os.path.isdir('checkpoint'):
            os.mkdir('checkpoint')
        torch.save(state, './checkpoint/ckpt.t7')
        best_acc = acc


def ps_init():
    # init the gradient variables for the model
    net.train()
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        if use_cuda:
            inputs, targets = inputs.cuda(), targets.cuda()
        optimizer.zero_grad()
        inputs, targets = Variable(inputs), Variable(targets)
        outputs = net(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        break


def ps_server(issyn):
    # global run_state
    if issyn > 0:
        print('===>synchronous training')
    else:
        print('===>asynchronous training')
    try:
        ps_init()
        # net.train()
        while True:
            if issyn > 0:
                conn.ps_synUpdate(optimizer)
            else:
                conn.ps_asynUpdate(optimizer)
    except Exception as e:
        print(str(e))

def done():
    f = open("state.txt", "a")
    f.write("1\n")
    f.close()

if __name__ == '__main__':
    if args.name == 'ps':
        ps_server(args.issyn)
    else:
        conn.send_zero_grad(-1)
        conn.update_wait()
        for epoch in range(start_epoch, start_epoch + 6):  # 60000):
            train(epoch)
            #test(epoch)
        done()
