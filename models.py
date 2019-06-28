import torch
from torch import nn, optim
from ANN import ANN
from RNN import RNN
from torch.utils.data import DataLoader
class Flatten(nn.Module):
    def __init__(self):
        super(Flatten, self).__init__()
    def forward(self, x):
        return x.view(x.size()[0], -1)

class CNN_TO_LSTM(nn.Module):
    def __init__(self):
        super(CNN_TO_LSTM, self).__init__()
    def forward(self, x):
        x = x.permute(0,2,1,3).contiguous()
        x = x.view(x.shape[0], x.shape[1], -1)
        return x
class RNN_Out(nn.Module):
    def __init__(self):
        super(RNN_Out, self).__init__()
    def forward(self, input):
        outputs , output = input
        return torch.squeeze(output)

class LSTM_Out(nn.Module):
    def __init__(self):
        super(LSTM_Out, self).__init__()
    def forward(self, input):
        output , (hn, cn) = input
        return output[:,-1,:]
class SwappSampleAxes(nn.Module):
    def __init__(self):
        super(SwappSampleAxes, self).__init__()
    def forward(self, input):
        return input.permute(0,2,1)
class Squeeze(nn.Module):
    def __init__(self):
        super(Squeeze, self).__init__()
    def forward(self, input):
        return torch.squeeze(input)


class Models:
    def DNN(self, data, test_set = None):
        input_sizes, output_size, train_set, valid_set = data
        input_size=1
        for dim in input_sizes:
            input_size *= dim
        hidden_sizes = [256, 256, 256]
        model = nn.Sequential(Flatten(),
                              nn.Linear(input_size, hidden_sizes[0]),
                              nn.ReLU(),
                              nn.BatchNorm1d(),
                              nn.Linear(hidden_sizes[0], hidden_sizes[1]),
                              nn.ReLU(),
                              nn.BatchNorm1d(),
                              nn.Linear(hidden_sizes[1], hidden_sizes[2]),
                              nn.ReLU(),
                              nn.BatchNorm1d(),
                              nn.Linear(hidden_sizes[2], output_size),
                            nn.LogSoftmax(dim=1)).cuda()
        network = ANN("DNN", model, cuda=True)
        network.train(train_set, epochs=20, batch_size=50, criterion=nn.NLLLoss(),
                      optimizer=optim.Adam(model.parameters()), valid_set=valid_set)
        return network
    def MLP_256_Relu_Dropout_256_Relu(self, data, test_set = None):
        input_sizes, output_size, train_set, valid_set = data
        input_size=1
        for dim in input_sizes:
            input_size *= dim
        hidden_sizes = [256, 256]
        model = nn.Sequential(Flatten(),
                              nn.Linear(input_size, hidden_sizes[0]),
                              nn.ReLU(),
                              nn.Dropout(p = 0.5),
                              nn.Linear(hidden_sizes[0], hidden_sizes[1]),
                              nn.ReLU(),
                              nn.Linear(hidden_sizes[1], output_size),
                            nn.LogSoftmax(dim=1)).cuda()
        network = ANN("MLP_256_Relu_Dropout_256_Relu", model, cuda=True)
        network.train(train_set, epochs=40, batch_size=50, criterion=nn.NLLLoss(),
                      optimizer=optim.Adam(model.parameters()), valid_set=valid_set)
        return network
    def MLP_256_Relu_256_Relu(self, data, test_set = None):
        input_sizes, output_size, train_set, valid_set = data
        input_size = 1
        for dim in input_sizes:
            input_size *= dim
        hidden_sizes = [256, 256]
        model = nn.Sequential(Flatten(),
                              nn.Linear(input_size, hidden_sizes[0]),
                              nn.ReLU(),
                              nn.Dropout(p = 0.5),
                              nn.Linear(hidden_sizes[0], hidden_sizes[1]),
                              nn.ReLU(),
                              nn.Linear(hidden_sizes[1], output_size),
                            nn.LogSoftmax(dim=1)).cuda()
        network = ANN("MLP_256_Relu_256_Relu", model, cuda=True)
        network.train(train_set, epochs=40, batch_size=50, criterion=nn.NLLLoss(),
                      optimizer=optim.Adam(model.parameters()), valid_set=valid_set)
        return network
    def CNN_2XConv2D_Relu_Dropout_FC(self, data, test_set = None):
        input_sizes, output_size, train_set, valid_set = data
        model = nn.Sequential(nn.Conv2d(in_channels=1, out_channels=16, kernel_size=5, stride=1, padding=2),
                              #nn.BatchNorm2d(16),
                              nn.ReLU(),
                              nn.Dropout2d(p=0.5),
                              nn.MaxPool2d(kernel_size=2),
                              nn.Conv2d(in_channels=16, out_channels=32, kernel_size=5, stride=1, padding=2),
                              #nn.BatchNorm2d(32),
                              nn.ReLU(),
                              nn.Dropout(p=0.5),
                              nn.MaxPool2d(kernel_size=2),
                              Flatten(),
                              nn.Linear(32 * 25 * 40, output_size),
                              nn.LogSoftmax(dim=1)).cuda()

        network = ANN("CNN:2XConv2D_Relu-Dropout-FC", model, cuda=True)
        network.train(train_set, epochs=40, batch_size=200, criterion=nn.NLLLoss(),
                      optimizer=optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-1), valid_set=valid_set)
        return network
    def CNN_4XConv2D_2XFC(self, data, test_set = None):
        input_sizes, output_size, train_set, valid_set = data
        model = nn.Sequential(
                              nn.Conv2d(in_channels=1, out_channels=32, kernel_size=5, stride=1, padding=2),
                              #nn.BatchNorm2d(16),
                              nn.ReLU(),
                              nn.Conv2d(in_channels=32, out_channels=32, kernel_size=5, stride=1, padding=2),
                              #nn.BatchNorm2d(16),
                              nn.ReLU(),
                              nn.MaxPool2d(kernel_size=2),
                              nn.Dropout2d(p=0.25),
                              nn.Conv2d(in_channels=32, out_channels=32, kernel_size=5, stride=1, padding=2),
                              #nn.BatchNorm2d(32),
                              nn.ReLU(),
                              nn.Conv2d(in_channels=32, out_channels=64, kernel_size=5, stride=1, padding=2),
                              # nn.BatchNorm2d(32),
                              nn.ReLU(),
                              nn.Dropout(p=0.5),
                              nn.MaxPool2d(kernel_size=2),
                              Flatten(),
                              nn.BatchNorm1d(64 * 25 * 40),
                              nn.Linear(64 * 25 * 40, 512),
                              nn.ReLU(),
                              nn.BatchNorm1d(512),
                              nn.Dropout(),
                              nn.Linear(512, output_size),
                              #nn.BatchNorm1d(output_size),
                              nn.LogSoftmax(dim=1)).cuda()

        network = ANN("CNN:2XConv2D_Relu-Dropout-FC", model, cuda=True)
        network.train(train_set, epochs=40, batch_size=150, criterion=nn.NLLLoss(),
                      optimizer=optim.Adam(model.parameters(),lr=0.01), valid_set=valid_set)
        return network

    def MyRNN_H256(self, data, test_set = None):
        input_sizes, output_size, train_set, valid_set = data
        model = nn.Sequential(Squeeze,
                              RNN(input_sizes[0], output_size, hidden_size=256, cuda=True))
        network = ANN("MyRNN_H256", model , cuda=True)
        network.train(train_set, epochs=60, batch_size=20, criterion=nn.NLLLoss(),
                      optimizer=optim.Adam(model.parameters(),lr=0.01), valid_set=valid_set)
        return network
    def RNN_H256(self, data, test_set = None):
        input_sizes, output_size, train_set, valid_set = data
        hidden_layer = 256
        batch_size =50
        model = nn.Sequential(Squeeze(),
                              SwappSampleAxes(),
                              nn.RNN(input_sizes[0],hidden_layer, batch_first=True),
                              RNN_Out(),
                              nn.Linear(hidden_layer, output_size),
                              nn.LogSoftmax(dim=1)).cuda()
        network = ANN("RNN_H256", model , cuda=True)
        network.train(train_set, epochs=60, batch_size=batch_size, criterion=nn.NLLLoss(),
                      optimizer=optim.Adam(model.parameters()), valid_set=valid_set)
        return network
    def LSTM_H256(self, data,  test_set = None):
        input_sizes, output_size, train_set, valid_set = data
        hidden_layer = 512
        model = nn.Sequential(Squeeze(),
                              SwappSampleAxes(),
                              nn.BatchNorm1d(input_sizes[1]),
                              nn.LSTM(input_sizes[0],hidden_layer, batch_first=True),
                              LSTM_Out(),
                              nn.BatchNorm1d(hidden_layer),
                              nn.Dropout(),
                              nn.Linear(hidden_layer, 256),
                              nn.ReLU(),
                              nn.Dropout(),
                              nn.Linear(256, output_size),
                              #nn.BatchNorm1d(output_size),
                              nn.LogSoftmax(dim=1)).cuda()
        network = ANN("LSTM_H512_FC256", model , cuda=True)
        network.train(train_set, epochs=60, batch_size=10, criterion=nn.NLLLoss(),
                      optimizer=optim.Adam(model.parameters(), weight_decay=1e-6), valid_set=valid_set)
        return network

    def CNN_LSTM256_FC512(self, data,  test_set = None):
        input_sizes, output_size, train_set, valid_set = data
        hidden_layer = 512
        model = nn.Sequential(nn.Conv2d(in_channels=1, out_channels=32, kernel_size=5, stride=1, padding=2),
                              #nn.BatchNorm2d(16),
                              nn.ReLU(),
                              nn.Conv2d(in_channels=32, out_channels=32, kernel_size=5, stride=1, padding=2),
                              #nn.BatchNorm2d(16),
                              nn.ReLU(),
                              CNN_TO_LSTM(),
                              nn.BatchNorm1d(input_sizes[0]),
                              nn.LSTM(input_sizes[1] * 32 ,hidden_layer, batch_first=True),
                              LSTM_Out(),
                              nn.BatchNorm1d(hidden_layer),
                              nn.Dropout(),
                              nn.Linear(hidden_layer, 256),
                              nn.ReLU(),
                              nn.Dropout(),
                              nn.Linear(256, output_size),
                              #nn.BatchNorm1d(output_size),
                              nn.LogSoftmax(dim=1)).cuda()
        network = ANN("LSTM_H512_FC256", model , cuda=True)
        network.train(train_set, epochs=60, batch_size=10, criterion=nn.NLLLoss(),
                      optimizer=optim.Adam(model.parameters(), weight_decay=1e-6), valid_set=valid_set)
        return network