import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable

torch.manual_seed(42)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

from matplotlib import pyplot as plt

class simpleLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super(simpleLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        # x shape (batch, time_step, input_size)
        # out shape (batch, time_step, output_size)
        # h_n shape (n_layers, batch, hidden_size)
        # h_c shape (n_layers, batch, hidden_size)
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)

        # forward propagate lstm
        out, (h_n, h_c) = self.lstm(x, (h0, c0))

        out = self.fc(out[:, :, :])
        return out

import numpy as np

with open("predata_v2.txt", "r") as my_train_file:
    data_1 = my_train_file.read().split(',')

with open("../data/EncodedStrands.txt", "r") as my_train_file:
    data_2 = my_train_file.read().splitlines()

import random

bmap = {"A":0, "C":1, "G":2, "T":3, "-": 4, "1": 4} 
def one_hot(b):
    t = [[0,0,0,0,0]]
    i = bmap[b]
    t[0][i] = 1
    return t

print("one-hot encoding for DNA bases")
print("A:", one_hot("A"))
print("C:", one_hot("C"))
print("G:", one_hot("G"))
print("T:", one_hot("T"))

seq = [random.choice(["A","C","G","T"]) for _ in range(220)]

seqs = data_1
seqs_2 = data_2

from random import shuffle

c = list(zip(seqs, seqs_2))

shuffle(c)

seqs, seqs_2 = zip(*c)

# convert the `seq` to a PyTorch tensor

seq_t = torch.Tensor([[one_hot(c) for c in ee] for ee in seqs_2])

seqs_t = torch.Tensor([[one_hot(c) for c in e] for e in seqs])

# Hyper Parameters
epochs = 1           
batch_size = 64
time_step = 120      
input_size = 5     
hidden_size = 64
num_layers = 1
num_classes = 5
lr = 0.002           # learning rate

dcnet = simpleLSTM(input_size, hidden_size, num_layers, num_classes)

dcnet.cuda()

loss_function = nn.CrossEntropyLoss()

optimizer = torch.optim.Adam(dcnet.parameters(), lr)


range_ = (1, 100)

max_correct = 0

a_a = []

a_c = []

a_g = []

a_t = []

acc = []

a_strand = []

mini_batch_size = batch_size
for epoch in range(600): 
    print("epoch: ", epoch)
    correct = 0
    num = 0
    num_strand = 0
    num_strand_v2 = 0

    num_correct_a = [0, 0]
    num_correct_c = [0, 0]
    num_correct_g = [0, 0]
    num_correct_t = [0, 0]

    result = []

    collect_labels = [] 

    for i in range(int(len(seqs_t)/mini_batch_size)):

        images = seqs_t[i * mini_batch_size : (i+1) * mini_batch_size]
        images = torch.mean(images.view(-1, 10, 5), dim=1)

        images = images.view(-1, time_step, input_size).to(device)

        labels = seq_t[i * mini_batch_size : (i+1) * mini_batch_size].view(-1, 120, 5)

        labels = labels.to(device)

        # forward pass
        outputs = dcnet(images)

        outputs_v2 = torch.argmax(outputs, dim=-1).view(-1, 120)

        result.append(outputs_v2.view(-1).cpu().numpy())

        labels_v2 = seq_t[i * mini_batch_size : (i+1) * mini_batch_size].view(-1, 120, 5)
        labels_v2 = torch.argmax(labels, dim=-1).view(-1,120)

        collect_labels.append(labels_v2.view(-1).cpu().numpy()) 

        loss = loss_function(outputs.view(-1,5), torch.argmax(labels.view(-1,5), dim=-1)) 
        
        # backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        correct = correct + torch.sum(F.one_hot(torch.argmax(outputs, dim=-1), 5) * labels)

        strand = torch.sum(torch.sum(F.one_hot(torch.argmax(outputs, dim=-1), 5) * labels, dim=-1).view(-1, 120), dim=-1)  


        a_1 = torch.Tensor([1, 0, 0, 0, 0]).to(device) * labels.view(-1,5)
        correct_a_1 = torch.sum(a_1)
        correct_a_2 = torch.sum(a_1 * F.one_hot(torch.argmax(outputs, dim=-1), 5).view(-1,5))

        num_correct_a = [num_correct_a[0] + correct_a_1, num_correct_a[1] + correct_a_2]

        c_1 = torch.Tensor([0, 1, 0, 0, 0]).to(device) * labels.view(-1,5)
        correct_c_1 = torch.sum(c_1)
        correct_c_2 = torch.sum(c_1 * F.one_hot(torch.argmax(outputs, dim=-1), 5).view(-1,5))

        num_correct_c = [num_correct_c[0] + correct_c_1, num_correct_c[1] + correct_c_2]

        g_1 = torch.Tensor([0, 0, 1, 0, 0]).to(device) * labels.view(-1,5)
        correct_g_1 = torch.sum(g_1)
        correct_g_2 = torch.sum(g_1 * F.one_hot(torch.argmax(outputs, dim=-1), 5).view(-1,5))

        num_correct_g = [num_correct_g[0] + correct_g_1, num_correct_g[1] + correct_g_2]

        t_1 = torch.Tensor([0, 0, 0, 1, 0]).to(device) * labels.view(-1,5)
        correct_t_1 = torch.sum(t_1)
        correct_t_2 = torch.sum(t_1 * F.one_hot(torch.argmax(outputs, dim=-1), 5).view(-1,5))

        num_correct_t = [num_correct_t[0] + correct_t_1, num_correct_t[1] + correct_t_2]

        for k in range(64):
            if(strand[k] == 120):
                num_strand += 1
        num_strand_v2 += 64

        num = num + 120 * 64

    print("a: ", correct/num)
    print("a_strand: ", num_strand/num_strand_v2)  
    print("a_a: ", num_correct_a[1]/num_correct_a[0], num_correct_a[0], num_correct_a[1])
    print("a_c: ", num_correct_c[1]/num_correct_c[0], num_correct_c[0], num_correct_c[1])
    print("a_g: ", num_correct_g[1]/num_correct_g[0], num_correct_g[0], num_correct_g[1])
    print("a_t: ", num_correct_t[1]/num_correct_t[0], num_correct_t[0], num_correct_t[1])
    acc.append(correct/num)
    a_strand.append(num_strand/num_strand_v2)
    a_a.append(num_correct_a[1]/num_correct_a[0])
    a_c.append(num_correct_c[1]/num_correct_c[0])
    a_g.append(num_correct_g[1]/num_correct_g[0])
    a_t.append(num_correct_t[1]/num_correct_t[0])

    np.savetxt("acc.txt", acc)
    np.savetxt("a_strand.txt", a_strand)
    np.savetxt("a_a.txt", a_a)
    np.savetxt("a_c.txt", a_c)
    np.savetxt("a_g.txt", a_g)
    np.savetxt("a_t.txt", a_t)
    if(correct/num > max_correct):
        max_correct = correct/num
        np.savetxt("result.txt", result)
        np.savetxt("label.txt", collect_labels)
        torch.save(dcnet, "model.pkl") 
        print("saved")

