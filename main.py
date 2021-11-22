import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

import torchvision
import torchvision.transforms as transforms
import os
import argparse
import matplotlib.pyplot as plt


def training_the_model(no_of_epoch, net, criterion, optimizer, device, trainloader, testloader, best_acc, scheduler):
    train_loss = []
    train_acc = []
    test_loss = []
    test_acc = []
    
    for epoch in range(no_of_epoch):
        train_loss, train_acc = train(epoch+1, net, criterion, optimizer, device, trainloader, train_loss, train_acc, False)
        best_acc, test_loss, test_acc = test(epoch+1, net, criterion, device, testloader, best_acc, test_loss, test_acc, False)
        train_loss, train_acc = train(epoch+1, net, criterion, optimizer, device, trainloader, train_loss, train_acc, True)
        best_acc, test_loss, test_acc = test(epoch+1, net, criterion, device, testloader, best_acc, test_loss, test_acc, True)
        
    print("Best Acc is : ", best_acc)
    return train_loss, train_acc, test_loss, 
    
def plot_graph(train_l, train_a, test_l, test_a):
  fig, axs = plt.subplots(2,2,figsize=(15,10))
  axs[0, 0].plot(train_l)
  axs[0, 0].set_title("Training Loss")
  axs[1, 0].plot(train_a)
  axs[1, 0].set_title("Training Accuracy")
  axs[0, 1].plot(test_l)
  axs[0, 1].set_title("Test Loss")
  axs[1, 1].plot(test_a)
  axs[1, 1].set_title("Test Accuracy")