#!/usr/bin/env python3
# -*- coding:utf8 -*-

"""
dev with python3.8 and torch 1.10.2+cu113
https://pytorch.org/tutorials/intermediate/char_rnn_classification_tutorial.html

use rnn to classify names
"""
import string
from io import open
import glob
import os
import random
import time
import math
import unicodedata
import torch
import torch.nn as nn

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

torch.device('cuda')
torch.cuda.device('cuda')

all_letters = string.ascii_letters + " .,;'"
n_letters = len(all_letters)


def find_files(path: str) -> list:
    return glob.glob(path)


def unicode2ascii(s: str):
    return ''.join(c for c in unicodedata.normalize('NFD', s)
                   if unicodedata.category(c) != 'Mn' and c in all_letters)


def read_lines(filename):
    with open(filename) as of:
        lines = of.read().strip().split('\n')
        return [unicode2ascii(line) for line in lines]


def letter2index(letter):
    return all_letters.find(letter)


def letter2tensor(letter):
    """
    one hot encoding for one letter
    :param letter:
    :return:
    """
    tensor = torch.zeros(1, n_letters)
    tensor[0][letter2index(letter)] = 1
    return tensor


def line2tensor(line):
    tensor = torch.zeros(len(line), 1, n_letters)
    for li, letter in enumerate(line):
        tensor[li][0][letter2index(letter)] = 1
    return tensor


def load_data():
    all_categories = []
    category_lines = {}
    path_pattern = '/home/page/workspace/deeplab/data/names/names/*.txt'
    files = find_files(path_pattern)
    for filename in files:
        category = os.path.splitext(os.path.basename(filename))[0]
        all_categories.append(category)
        lines = read_lines(filename)
        category_lines[category] = lines

    return all_categories, category_lines


class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(RNN, self).__init__()
        self.hidden_size = hidden_size
        self.i2h = nn.Linear(input_size + hidden_size, hidden_size)
        self.i2o = nn.Linear(input_size + hidden_size, output_size)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, inp, hidden):
        combined = torch.cat((inp, hidden), 1)
        hidden = self.i2h(combined)
        output = self.i2o(combined)
        output = self.softmax(output)
        return output, hidden

    def init_hidden(self):
        return torch.zeros(1, self.hidden_size)


def category_from_output(output: torch.Tensor, all_categories):
    top_n, top_i = output.topk(1)
    category_i = top_i[0].item()
    return all_categories[category_i], category_i


def random_choice(n: list):
    return n[random.randint(0, len(n)-1)]


def random_training_example(all_categories: list, category_line: dict):
    category = random_choice(all_categories)
    line = random_choice(category_line[category])
    category_tensor = torch.tensor([all_categories.index(category)], dtype=torch.long)
    line_tensor = line2tensor(line)
    return category, line, category_tensor, line_tensor


def train(rnn: RNN, category_tensor, line_tensor, criterion, lr: float):
    hidden = torch.zeros(1, rnn.hidden_size)
    rnn.zero_grad()
    for i in range(line_tensor.size()[0]):
        output, hidden = rnn(line_tensor[i], hidden.cuda())
    loss = criterion(output, category_tensor.cuda())
    loss.backward()

    for p in rnn.parameters():
        p.data.add_(p.grad.data, alpha=-lr)
    return output, loss.item()


def time_since(since):
    now = time.time()
    s = now - since
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)


def train_epoch(all_categories, category_line):
    start = time.time()
    n_iters = 100000
    print_every = 5000
    plot_every = 1000
    n_hidden = 128
    criterion = nn.NLLLoss()

    # Keep track of losses for plotting
    current_loss = 0
    all_losses = []
    rnn = RNN(n_letters, n_hidden, len(all_categories))
    rnn = rnn.cuda()
    for i in range(1, n_iters + 1):
        category, line, category_tensor, line_tensor = random_training_example(all_categories, category_line)
        output, loss = train(rnn, category_tensor, line_tensor.cuda(), criterion, lr=0.005)

        if i % print_every == 0:
            guess, guess_i = category_from_output(output, all_categories)
            correct = '✓' if guess == category else '✗ (%s)' % category
            print('%d %d%% (%s) %.4f %s / %s %s'% (i, i/n_iters * 100, time_since(start), loss, line, guess, correct))

        current_loss += loss
        if i % plot_every == 0:
            all_losses.append(current_loss / plot_every)
            current_loss = 0
    return rnn, all_losses


def plotting_loss(loss_list):
    plt.figure()
    plt.plot(loss_list)
    plt.show()


def evaluate(rnn: RNN, line_tensor):
    hidden = torch.zeros(1, rnn.hidden_size)
    for i in range(line_tensor.size()[0]):
        output, hidden = rnn(line_tensor[i], hidden)
    return output


def print_confusion_matrix(model: RNN, all_categories: list, category_line:dict):
    n_categories = len(all_categories)
    confusion = torch.zeros(n_categories, n_categories)
    n_confusion = 10000
    for i in range(n_confusion):
        category, line, category_tensor, line_tensor = random_training_example(all_categories, category_line)
        output = evaluate(model, line_tensor)
        guess, guess_i = category_from_output(output, all_categories)
        category_i = all_categories.index(category)
        confusion[category_i][guess_i] += 1
    for i in range(n_categories):
        confusion[i] = confusion[i]/confusion[i].sum()

    fig = plt.figure(dpi=300)
    ax = fig.add_subplot(111)
    cax = ax.matshow(confusion.numpy())
    fig.colorbar(cax)

    ax.set_xticklabels([''] + all_categories, rotation=90)
    ax.set_yticklabels([''] + all_categories)

    ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
    ax.yaxis.set_major_locator(ticker.MultipleLocator(1))
    plt.show()


def predict(model: RNN, input_line, all_categories, n_predictions=3):
    print('\n %s' % input_line)
    with torch.no_grad():
        output = evaluate(model, line2tensor(input_line))
        top_v, top_i = output.topk(n_predictions, 1, True)
        predictions = []
        for i in range(n_predictions):
            value = top_v[0][i].item()
            category_index = top_i[0][i].item()
            print('(%.2f ) %s' % (value, all_categories[category_index]))
            predictions.append([value, all_categories[category_index]])

        return predictions


def practice():
    files = find_files('/home/page/workspace/deeplab/data/names/names/*.txt')
    print(files)
    print(unicode2ascii('Ślusàrski'))
    a, b = load_data()
    print(len(a))
    print(b['Italian'][:5])
    print(letter2tensor('X'))
    n_hidden = 128
    rnn = RNN(n_letters, n_hidden, len(a))
    inp = letter2tensor('A')
    hidden = torch.zeros(1, n_hidden)
    output, next_hidden = rnn(inp, hidden)
    print(output)

    inp = line2tensor('Albert')
    hidden = torch.zeros(1, n_hidden)
    output, next_hidden = rnn(inp[1], hidden)
    print(output)
    print(next_hidden.shape)
    print(type(output))
    print(category_from_output(output, a))

    for i in range(10):
        cate, line, cate_tensor, line_tensor = random_training_example(a, b)
        print('category = ', cate, '/ line = ', line)


def main():
    all_categories, category_line = load_data()
    model, loss_list = train_epoch(all_categories, category_line)
    plotting_loss(loss_list)
    print_confusion_matrix(model, all_categories, category_line)
    # #
    # torch.save(model, 'rnn_classify_model.pth')

    #reload_model = torch.load('rnn_classify_model.pth')
    predict(model, 'wang', all_categories)


if __name__ == '__main__':
    main()
