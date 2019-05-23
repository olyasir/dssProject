import os
import torch
import torch.nn as nn
from torch.autograd import Variable
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torch.nn.functional as F
import torch.optim as optim
from dataLoader import  crypticLettersDataset

from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
import matplotlib.pyplot as plt
import pandas as pd
from sklearn import metrics
## load mnist dataset
use_cuda = torch.cuda.is_available()

root = './data'
if not os.path.exists(root):
    os.mkdir(root)

trans = transforms.Compose( [ AddBackground(),  transforms.ToTensor(), transforms.Normalize((0.5,), (1.0,))])
# if not exist, download mnist dataset
train_set = dset.MNIST(root=root, train=True, transform=trans, download=True)
test_set = dset.MNIST(root=root, train=False, transform=trans, download=True)

batch_size = 100

train_loader = torch.utils.data.DataLoader(
    dataset=train_set,
    batch_size=batch_size,
    shuffle=True)
test_loader = torch.utils.data.DataLoader(
    dataset=test_set,
    batch_size=batch_size,
    shuffle=False)

print ('==>>> total trainning batch number: {}'.format(len(train_loader)))
print('==>>> total testing batch number: {}'.format(len(test_loader)))

class LeNet(nn.Module):
    def __init__(self):
        super(LeNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 20, 5, 1)
        self.conv2 = nn.Conv2d(20, 50, 5, 1)
        self.fc1 = nn.Linear(4 * 4 * 50, 500)
        self.fc2 = nn.Linear(500, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2, 2)
        x = x.view(-1, 4 * 4 * 50)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

    def name(self):
        return "LeNet"


def pretrainMnist():
    ## training
    model = LeNet()

    if use_cuda:
        model = model.cuda()

    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

    criterion = nn.CrossEntropyLoss()

    for epoch in range(10):
        # trainning
        print(epoch)
        ave_loss = 0
        for batch_idx, (x, target) in enumerate(train_loader):
            optimizer.zero_grad()
            if use_cuda:
                x, target = x.cuda(), target.cuda()
            x, target = Variable(x), Variable(target)
            out = model(x)
            loss = criterion(out, target)
            ave_loss = ave_loss * 0.9 + loss.data[0] * 0.1
            loss.backward()
            optimizer.step()
            if (batch_idx + 1) % 100 == 0 or (batch_idx + 1) == len(train_loader):
                print ('==>>> epoch: {}, batch index: {}, train loss: {:.6f}'.format(
                    epoch, batch_idx + 1, ave_loss))
        # testing
        correct_cnt, ave_loss = 0, 0
        total_cnt = 0
        for batch_idx, (x, target) in enumerate(test_loader):
            if use_cuda:
                x, target = x.cuda(), target.cuda()
            x, target = Variable(x, volatile=True), Variable(target, volatile=True)
            out = model(x)
            loss = criterion(out, target)
            _, pred_label = torch.max(out.data, 1)
            #total_cnt += x.data.size()[0]
            correct_cnt += (pred_label == target.data).sum()
            # smooth average
            ave_loss = ave_loss * 0.9 + loss.data[0] * 0.1

            if (batch_idx + 1) % 100 == 0 or (batch_idx + 1) == len(test_loader):
                print ( '==>>> epoch: {}, batch index: {}, test loss: {:.7f}, acc: {:.7f}'.format(
                    epoch, batch_idx + 1, ave_loss, float(correct_cnt) / float(total_cnt)))

    torch.save(model.state_dict(), model.name())


def train():
    model = LeNet()
    state = torch.load('LeNet')
    model.load_state_dict(state)
    for param in model.parameters():
        param.requires_grad = False
        # Replace the last fully-connected layer
        # Parameters of newly constructed modules have requires_grad=True by default
    model.fc2 = nn.Linear(500, 23)  # assuming that the fc7 layer has 512 neurons, otherwise change it
    model.cuda()
    dataset = crypticLettersDataset(root_dir = '/home/olya/Documents/thesis/data', transform = trans)
    totalLen = len(dataset)

    train_loader = torch.utils.data.DataLoader(dataset, batch_size=4, #shuffle=True,
                                               sampler=SubsetRandomSampler(range(0,int(totalLen*0.8))))
    test_loader = torch.utils.data.DataLoader(dataset, batch_size=4, #shuffle=True,
                                              sampler=SubsetRandomSampler( range(int(totalLen*0.8), totalLen)) )


    dataloader = DataLoader(dataset, batch_size=4,
                            shuffle=True, num_workers=4)

    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

    criterion = nn.CrossEntropyLoss()
    test_loss = []
    train_loss = []
    test_accurancy =[]
    train_accurancy = []
    for epoch in range(100):
        # trainning
        total_train_loss = 0
        total_train_loss_cnt=0
        train_acc =0

        for batch_idx, (target, x) in enumerate(train_loader):
            optimizer.zero_grad()
            if use_cuda:
                x, target = x.cuda(), target.cuda()
            x, target = Variable(x), Variable(target)
            out = model(x)
            loss = criterion(out, target)
            loss.backward()
            optimizer.step()

            _, pred_label = torch.max(out.data, 1)
            correct_cnt = (pred_label == target.data).sum()
            train_acc+= int(correct_cnt)/float(x.data.size()[0])
            total_train_loss += loss[0]
            total_train_loss_cnt+=1
        train_loss.append( total_train_loss / float(total_train_loss_cnt))
        train_accurancy.append( train_acc/ float(total_train_loss_cnt))

        # testing

        correct_cnt = 0
        total_test_loss = 0
        total_test_loss_cnt = 0
        test_acc = 0
        for batch_idx, (target, x) in enumerate(test_loader):
            if use_cuda:
                x, target = x.cuda(), target.cuda()
            x, target = Variable(x, volatile=True), Variable(target, volatile=True)
            out = model(x)
            loss = criterion(out, target)
            _, pred_label = torch.max(out.data, 1)
            correct_cnt = (pred_label == target.data).sum()
            total_test_loss += loss[0]
            total_test_loss_cnt +=1
            test_acc  += int(correct_cnt)/float(x.data.size()[0])
        test_loss.append( total_test_loss / float(total_test_loss_cnt) )
        test_accurancy.append( test_acc/ float(total_test_loss_cnt) )

        if epoch ==0:
            expected =[]
            predicted =[]
            for batch_idx, (target, x) in enumerate(test_loader):
                if use_cuda:
                    x, target = x.cuda(), target.cuda()
                x, target = Variable(x, volatile=True), Variable(target, volatile=True)
                out = model(x)
                loss = criterion(out, target)
                _, pred_label = torch.max(out.data, 1)
                expected +=  [ i for i in target.data.cpu().numpy()]
                predicted += [ i for i in pred_label.data.cpu().numpy()]

            y_actu = pd.Series(expected + [ i for i in range(22)], name='Actual')
            y_pred = pd.Series(predicted+[ i for i in range(22)], name='Predicted')
            df_confusion = pd.crosstab(y_actu, y_pred)



    df = pd.DataFrame({'x': range(len(train_loss)), 'train': train_loss, 'test': test_loss, 'test_accurancy': test_accurancy, 'train_accurancy': train_accurancy} )

    # multiple line plot
    plt.plot('x', 'train', data=df, color='skyblue', linewidth=2, label='train')
    plt.plot('x', 'test', data=df, color='olive', linewidth=2, label = 'test')
    plt.legend()
    plt.show()

    plt.plot('x', 'train_accurancy', data=df, color='green', linewidth=2, label='train_accurancy')
    plt.plot('x', 'test_accurancy', data=df, color='red', linewidth=2, label='test_accurancy')
    plt.legend()
    plt.show()

# first pretrain mnist and then train on letters
#pretrainMnist()
train()


class AddBackground( object ):

    def __call__(self, sample ):
        import numpy as np
        sample = 255 - sample.numpy()
        bg = _getBackground((28, 28))
        for d in [0, 1, 2]:
            bg[:, :, d][np.where(sample > 0)] = sample[np.where(sample > 0)]
        return bg

def _getRandomBg():
    import os
    import random
    bgFolder = '/home/olya/Documents/thesis/se/backgrounds'
    allBg = os.listdir(bgFolder)
    bgPath = random.choice(allBg)
    bg = Image.open(bgPath)
    return bg


def _getBackground(shape):
    import random
    inw, inh = shape
    image = _getRandomBg()
    w, h = image.shape
    x = random.randint(0, w - inw)
    y = random.randint(0, h - inh)
    crop = image.crop((x, y, x + inw, y + inh))
    return crop


