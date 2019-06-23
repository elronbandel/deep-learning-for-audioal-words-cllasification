import torch
from time import time
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader
from torch.autograd import Variable

def flat(x):
    return x.view(x.size()[0], -1)
def print_train_start(name):
    print("__________________________________________________________________________")
    print("                     Trainning <" + name + ">")
    print("__________________________________________________________________________")

def print_epoch(epoch,epochs, duration, loss, fitting, accuracy ):
    str = "EPOCH {}/{} | Duaration: {:4.1f}s | Loss: {:4.3f} | Fitting: {:4.3f}".format(epoch,epochs, duration, loss, fitting)
    if accuracy is not None:
        str += "| Accuracy: {:4.3f}".format(accuracy)
    print(str)

def print_train_end(name,  time):
        print("________________________________________________________________________")
        print("          Training FInished in {:4.1f}m".format(time/60))
        print("________________________________________________________________________")

class ANN:
    def __init__(self, name,  network_structure, cuda):
        self.name = name
        self.cuda = cuda
        if cuda:
            self.model = network_structure




    def predict(self, x):
        self.model.eval()
        with torch.no_grad():
            maxes, arg_maxes = self.model(x).max(1)
            return arg_maxes


    def train(self, train_set, epochs, batch_size, criterion, optimizer, valid_set=None):
        loader = DataLoader(train_set, batch_size=batch_size, shuffle=True,
                                   num_workers=20, pin_memory=True, sampler=None)
        validation = None
        if valid_set is not None:
            validation = DataLoader(valid_set, batch_size=100, shuffle=True, num_workers=20, pin_memory=True, sampler=None)
        t_start = time()
        print_train_start(self.name)
        for e in range(epochs):
            e_start = time()
            losses = []

            for k, (input, label) in enumerate(loader):
                #input = torch.squeeze(input)
                if self.cuda:
                    input , label = Variable(input.cuda()), Variable(label.cuda())
                optimizer.zero_grad()
                self.model.train()
                output = self.model(input)
                loss = criterion(output, label)
                losses.append(float(loss.data.mean()))
                loss.backward()
                optimizer.step()

            accuracy = None
            if validation is not None:
                accuracy = self.validate(validation)
            fitting = self.validate(loader)

            print_epoch(1 + e, epochs,  time() - e_start, np.mean(losses), fitting, accuracy)
        print_train_end(self.name, time() - t_start)

    def validate(self, loader):
        accuracies = []
        for input, label in loader:
            if self.cuda:
                input, label = input.cuda(), label.cuda()
            output = self.predict(input)
            equals = torch.eq(output, label).double()
            mea = torch.mean(equals)
            fin = float(mea)
            accuracies.append(fin)
        return np.mean(accuracies)






