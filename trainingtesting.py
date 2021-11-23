  import torch
from tqdm import tqdm

import torch.optim as optim
import torch.nn as nn

train_losses = []
train_acc = []

from tqdm import tqdm

train_losses = []
test_losses = []

def trainNetwork(net,epoch, optimizer,criterion,trainloader):
  net.train()
  pbar = tqdm(trainloader)
  correct = 0
  processed = 0
  for batch_idx, (data, target) in enumerate(pbar):    
    # get samples
    data, target = data.to(device), target.to(device)
    # Init
    optimizer.zero_grad()
    # Predict
    y_pred = net(data)
    # Calculate loss
    loss = criterion(y_pred, target)
    train_losses.append(loss)
    
    # Backpropagation
    loss.backward()
    optimizer.step()
    
    # Update pbar-tqdm    
    pred = y_pred.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
    correct += pred.eq(target.view_as(pred)).sum().item()
    processed += len(data)

    pbar.set_description(desc= f'Loss={loss.item()} Batch_id={batch_idx} Train Accuracy={100*correct/processed:0.2f}')
        
#Testing
def testNetwork(net, epoch,testloader):
    global best_acc
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
    
    test_loss /= len(testloader.dataset)
    test_losses.append(test_loss)
    
    print('Accuracy of the network on the 10000 test images: %d %%' % (
        100 * correct / total)) 



def getTrainLoss():
    train_losses_cpu = []
    for loss in train_losses:
        train_losses_cpu.append(loss.cpu().data.numpy())  
    
    return train_losses_cpu

def getTestLoss():
    return test_losses
