from gcommand_loader import GCommandLoader
import torch
from models import Models
from torch import nn, optim
import numpy as np
from os import listdir
from ANN import ANN
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
def heatmap2d(arr: np.ndarray):
    plt.imshow(arr, cmap='viridis')
    plt.colorbar()
    plt.show()
class Flatten(nn.Module):
    def __init__(self):
        super(Flatten, self).__init__()
    def forward(self, x):
        return x.view(x.size()[0], -1)
def load():
    batch_size = 20
    train_set = GCommandLoader('./data/train')
    valid_set = GCommandLoader('./data/valid')
    return train_set, valid_set
def sizes(dataset):
    loader = DataLoader(dataset, num_workers=20, pin_memory=True, sampler=None)
    input, label = next(iter(loader))
    #heatmap2d(input.squeeze().transpose(1,0).numpy())
    input_size = torch.squeeze(input).shape
    output_size = len(loader.dataset.classes)
    return input_size, output_size
def get_data():
    train_set, valid_set = load()
    input_size, output_size = sizes(valid_set)
    return input_size, output_size, train_set, valid_set
def test(model):
    test_set = GCommandLoader('./data/test')
    files = listdir('./data/test/test')
    tests = DataLoader(test_set, num_workers=20, pin_memory=True, sampler=None)
    with open('test_y') as file:
        for (input, label), name in zip(tests, files):
            file.write("{}, {}".format(name, int(model.predict(label))))


def main():
    models = Models()
    data = get_data()

    #train all the models
    models.CNN_LSTM256_FC512(data)








if __name__ == "__main__":
    main()



