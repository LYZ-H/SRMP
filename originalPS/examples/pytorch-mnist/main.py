'''Train MNIST with PyTorch.'''
from __future__ import print_function

import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

import torchvision
import torchvision.transforms as transforms

import os
import argparse

from torch.autograd import Variable


import sys

current_path = os.path.dirname(os.path.realpath(__file__))
ext_path1 = os.path.abspath(current_path + "/../../")
ext_path2 = os.path.abspath(current_path + "/../../../")
sys.path.extend([ext_path1, ext_path2])

from ps.adapters.pytorch import psPyTorchAdapter

#from models import *
from models.tools import get_model
from tools.mnist_read import MNIST


import logging
def init_logging(filename):
   # Create the logs directory
   if not os.path.exists("./logs"):
      os.mkdir("./logs")

   # Init logging to file
   logging.basicConfig(format='[%(asctime)s] [%(levelname)s] [%(name)s]  %(message)s',
                     filename="./logs/%s" % filename,
                     filemode='w',
                     level=logging.DEBUG)

   # logging.getLogger("dpwa.conn").setLevel(logging.INFO)


LOGGER = logging.getLogger(__name__)


parser = argparse.ArgumentParser(description='PyTorch MNIST Training')
parser.add_argument('--lr', default=0.01, type=float, help='learning rate')
parser.add_argument('--resume', '-r', action='store_true', help='resume from checkpoint')
parser.add_argument('--batch-size', type=int, default=32, help='Set the batch size (default: 128)')
parser.add_argument('--config-file', type=str, default='dpwa.yaml', help='Dpwa configuration file')
parser.add_argument('--remove-labels', type=str,default='None', help='Set remove-labels like 1,2,3;default: None')
parser.add_argument('--slice', type=str,default='None', help='Set slice like i/M, means read i-th part of M-parts;default: None')
parser.add_argument('--test-batches', type=int,default=100, help='Batch number per test; defaut: 100')
parser.add_argument('--model-name', type=str,default='MobileNet', help='The trained model name; defaut: MobileNet')
parser.add_argument('--name', type=str, required=True, help="This worker's name within config file")
parser.add_argument('--issyn', type=int, default=1, help="0: asynchronous; 1: synchronous")
parser.add_argument('--dataset-location', type=str,default='../../../../data', help='the path of the training data')
parser.add_argument('--enable-cuda', action='store_true', default=False, help='xxx')

args = parser.parse_args()


remove_labels = []
if args.remove_labels != 'None':
   remove_str = args.remove_labels.split(',')
   for item in remove_str:
       remove_labels.append(int(item.strip()))
isSlice = 0
if args.slice != 'None':
   isSlice = 1
   slice_str = args.slice.strip()
   slice_str = slice_str.strip(' ')
   start_str, step_str = slice_str.split('/')
   start_index = int(start_str)-1
   step_num = int(step_str)

init_logging(args.name + ".log")

batch_size = args.batch_size
print("Using batch_size =", batch_size)

if args.enable_cuda:
   use_cuda = torch.cuda.is_available()
else:
   use_cuda = False

best_acc = 0  # best test accuracy
start_epoch = 0  # start from epoch 0 or last checkpoint epoch

# Data
print('==> Preparing data..')
transform_train = transforms.Compose([
    transforms.Pad(padding=2,fill=0),
    transforms.ToTensor(),
    transforms.Normalize((0.4914,), (0.2023,)),
])

transform_test = transforms.Compose([
    transforms.Pad(padding=2,fill=0),
    transforms.ToTensor(),
    transforms.Normalize((0.4914,), (0.2023,)),
])

prepare_data = MNIST(root=args.dataset_location, train=True, download=True, transform=transform_train)
prepare_data.remove_labels(remove_labels)
if isSlice != 0:
   prepare_data.read_by_index(start_index,step_num)
trainset = MNIST(root=args.dataset_location, train=True, download=False, transform=transform_train)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=2)

testset = MNIST(root=args.dataset_location, train=False, download=False, transform=transform_test)
testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=2)

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
   net = model(channel_num=1)

print('==> Built model: %s', args.model_name)
if use_cuda:
    net.cuda()
    net = torch.nn.DataParallel(net, device_ids=range(torch.cuda.device_count()))
    cudnn.benchmark = True

conn = psPyTorchAdapter(net, args.name, args.config_file)
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)


MOVING_AVG_SIZE = 10
test_batches = args.test_batches


# Training
def train(epoch):
   print('\nEpoch: %d' % epoch)
   net.train()

   accuracies = []
   losses = []
   loss_mean = 9999
   for batch_idx, (inputs, targets) in enumerate(trainloader):
      if use_cuda:
         inputs, targets = inputs.cuda(), targets.cuda()
      optimizer.zero_grad()
      inputs, targets = Variable(inputs), Variable(targets)
      outputs = net(inputs)
      loss = criterion(outputs, targets)
      loss.backward()
      #optimizer.step()
      conn.grad_send(batch_idx)
      # Calculate the loss
      #losses += [loss.data[0]]
      losses += [loss.item()]
      loss_mean = np.array(losses[-MOVING_AVG_SIZE:]).mean()

      # Calculate the accuracy
      _, predicted = torch.max(outputs.data, 1)
      total = targets.size(0) 
      correct = predicted.eq(targets.data).cpu().sum()
      accuracies += [float(correct)/float(total)]
      accuracy = np.array(accuracies[-MOVING_AVG_SIZE:]).mean() * 100.0

      # Show progress
      progress = "[%s] E%d | B%d | Loss: %.3f | Acc: %.3f%%" % \
                  (args.name, epoch, batch_idx, loss_mean, accuracy)
      print(progress)
      LOGGER.info(progress)

      # wait updated parameters from ps
      conn.update_wait()
      # test for every given number of batches
      if batch_idx!=0 and batch_idx % test_batches == 0:
         fast_test(epoch)
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
      inputs, targets = Variable(inputs, volatile=True), Variable(targets)
      outputs = net(inputs)
      loss = criterion(outputs, targets)

      test_loss += loss.item()
      _, predicted = torch.max(outputs.data, 1)
      total += targets.size(0)
      correct += predicted.eq(targets.data).cpu().sum()
      if batch_idx > test_batches:
         break
      progress = "[%s] E%d | B%d | Loss: %.3f | Acc: %.3f%% (%d/%d)" % \
                  (args.name, epoch, batch_idx, test_loss/(batch_idx+1), 100.*correct/total, correct, total)
      print(progress)
      LOGGER.info(progress)
      # progress_bar(batch_idx, len(testloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
      #    % (test_loss/(batch_idx+1), 100.*correct/total, correct, total))


def test(epoch):
   global best_acc
   net.eval()
   test_loss = 0
   correct = 0
   total = 0
   for batch_idx, (inputs, targets) in enumerate(testloader):
      if use_cuda:
         inputs, targets = inputs.cuda(), targets.cuda()
      inputs, targets = Variable(inputs, volatile=True), Variable(targets)
      outputs = net(inputs)
      loss = criterion(outputs, targets)

      test_loss += loss.item()
      _, predicted = torch.max(outputs.data, 1)
      total += targets.size(0)
      correct += predicted.eq(targets.data).cpu().sum()

      progress = "[%s] E%d | B%d | Loss: %.3f | Acc: %.3f%% (%d/%d)" % \
                  (args.name, epoch, batch_idx, test_loss/(batch_idx+1), 100.*correct/total, correct, total)
      print(progress)
      LOGGER.info(progress)
      # progress_bar(batch_idx, len(testloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
      #    % (test_loss/(batch_idx+1), 100.*correct/total, correct, total))

   # Save checkpoint.
   acc = 100.*correct/total
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
   if issyn > 0:
      print('===>synchronous training')
   else:
      print('===>asynchronous training')
   try:
      ps_init()
      while True:
         if issyn > 0:
            conn.ps_synUpdate(optimizer)
         else:
            conn.ps_asynUpdate(optimizer)
   except Exception as e:
         LOGGER.exception("Error: ", str(e))
         print(str(e))


if __name__ == '__main__':
   if args.name == 'ps':
      ps_server(args.issyn)
   else:   
      for epoch in range(start_epoch, start_epoch + 10): # 60000):
         train(epoch)
         test(epoch)

