from gcommand_loader import GCommandLoader
import torch
from models import Models
from torch import nn, optim
from ANN import ANN
from torch.utils.data import DataLoader

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
    input_size = torch.squeeze(input).shape
    output_size = len(loader.dataset.classes)
    return input_size, output_size
def get_data():
    train_set, valid_set = load()
    input_size, output_size = sizes(valid_set)
    return input_size, output_size, train_set, valid_set

def main():
    models = Models()
    data = get_data()

    #train all the models
    #models.CNN_3XConv2D_Relu_Dropout_2XFC(data)

    lstm = models.LSTM_H256(data)
    convnet = models.CNN_2XConv2D_Relu_Dropout_FC(data)



    mlp_dropout = models.MLP_256_Relu_Dropout_256_Relu(data)
    mlp = models.MLP_256_Relu_256_Relu(data)






if __name__ == "__main__":
    main()



