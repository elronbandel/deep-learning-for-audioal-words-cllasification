import torch.nn as nn
import torch
from torch.autograd import Variable
import torch.nn.functional as F


class RNN(nn.Module):
    def __init__(self, input_size, output_size, hidden_size, cuda=False):
        super(RNN, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.is_cuda = cuda
        if cuda:
            self.w_x = nn.Linear(input_size, hidden_size).cuda()
            self.w_h = nn.Linear(hidden_size, hidden_size).cuda()
            self.w_o = nn.Linear(hidden_size, output_size).cuda()
            self.tahn = nn.Tanh().cuda()
            self.softmax = nn.LogSoftmax(dim=1).cuda()
        else:
            self.w_x = nn.Linear(input_size, hidden_size)
            self.w_h = nn.Linear(hidden_size, hidden_size)
            self.w_o = nn.Linear(hidden_size, output_size)
            self.tahn = nn.Tanh()
            self.softmax = nn.LogSoftmax(dim=1)







    def init_hidden(self):
        if not self.is_cuda:
            hidden = Variable(torch.zeros(1, self.hidden_size))
        else:
            hidden = Variable(torch.cuda.FloatTensor(1, self.hidden_size).fill_(0))
        return hidden

        return
    def forward(self, input):
        hidden = self.init_hidden()
        duration = input.size()[2]
        output = None
        for i in range(duration):
            msec_batch = input[:, :, i]
            output, hidden = self.network(msec_batch, hidden)
        return output



    def network(self, input, hidden):
        hidden = self.w_h(hidden) + self.w_x(input)
        hidden = self.tahn(hidden)
        output = self.w_o(hidden)
        output = self.softmax(output)
        return output, hidden
