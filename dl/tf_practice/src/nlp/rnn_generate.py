#!/usr/bin/env python3
# -*- coding:utf8 -*-

"""
dev with python3.8 and torch 1.10.2+cu113
https://pytorch.org/tutorials/intermediate/char_rnn_generation_tutorial.html

use rnn to generate names
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

from constant import base_dir


all_letters = string.ascii_letters + " .,;'"
n_letters = len(all_letters) + 1  # plus eos


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


def load_data(path_pattern: str):
    all_categories = []
    category_lines = {}
    files = find_files(path_pattern)
    for filename in files:
        category = os.path.splitext(os.path.basename(filename))[0]
        all_categories.append(category)
        lines = read_lines(filename)
        category_lines[category] = lines
    return all_categories, category_lines


class RNN(nn.Module):
    """
    model define
    """
    def __init__(self, category_size, input_size, hidden_size, output_size):
        super(RNN, self).__init__()
        self.hidden_size = hidden_size
        self.i2h = nn.Linear(category_size + input_size + hidden_size, hidden_size)
        self.i2o = nn.Linear(category_size + input_size + hidden_size, output_size)
        self.o2o = nn.Linear(hidden_size + output_size, output_size)
        self.dropout = nn.Dropout(0.1)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, category, inp, hidden):
        combined = torch.cat((category, inp, hidden), 1)
        hidden = self.i2h(combined)
        output1 = self.i2o(combined)
        output_combined = torch.cat((hidden, output1), 1)
        output = self.o2o(output_combined)
        output = self.dropout(output)
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


def category2tensor(all_categories, category):
    ln = all_categories.index(category)
    tensor = torch.zeros(1, len(all_categories))
    tensor[0][ln] = 1
    return tensor


def input_line2tensor(line):
    tensor = torch.zeros(len(line), 1, n_letters)
    for li, letter in enumerate(line):
        tensor[li][0][letter2index(letter)] = 1
    # for li in range(len(line)):
    #     letter = line[li]
    #     tensor[li][0][all_letters.find(letter)] = 1
    return tensor


def output_line2tensor(line):
    letter_indexes = [all_letters.find(line[n]) for n in range(1, len(line))]
    letter_indexes.append(n_letters - 1)
    return torch.LongTensor(letter_indexes)


# def category2tensor(category, all_categories: list):
#     idx = all_categories.index(category)
#     tensor = torch.zeros(1, len(all_categories))
#     tensor[0][idx] = 1
#     return tensor


def random_training_example(all_categories: list, category_line: dict):
    category = random_choice(all_categories)
    line = random_choice(category_line[category])
    # one hot
    # category_tensor = torch.tensor([all_categories.index(category)], dtype=torch.long)
    category_tensor = category2tensor(all_categories, category)
    input_line_tensor = input_line2tensor(line)
    target_line_tensor = output_line2tensor(line)
    return category, line, category_tensor, input_line_tensor, target_line_tensor


def train(rnn: RNN, category_tensor, input_line_tensor, target_line_tensor, criterion, lr: float):
    hidden = rnn.init_hidden()
    target_line_tensor.unsqueeze_(-1)

    rnn.zero_grad()
    loss = 0
    for i in range(input_line_tensor.size(0)):
        output, hidden = rnn(category_tensor, input_line_tensor[i], hidden)
        step_loss = criterion(output, target_line_tensor[i])
        loss += step_loss
    loss.backward()

    for p in rnn.parameters():
        p.data.add_(p.grad.data, alpha=-lr)
    return output, loss.item() / input_line_tensor.size(0)


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
    n_categories = len(all_categories)
    # Keep track of losses for plotting
    current_loss = 0
    all_losses = []
    rnn = RNN(n_categories, n_letters, n_hidden, n_letters)
    for i in range(1, n_iters + 1):
        category, line, category_tensor, input_line_tensor, target_line_tensor = random_training_example(all_categories, category_line)
        output, loss = train(rnn, category_tensor, input_line_tensor, target_line_tensor, criterion, lr=0.0005)

        if i % print_every == 0:
            # guess, guess_i = category_from_output(output, all_categories)
            # correct = '✓' if guess == category else '✗ (%s)' % category
            # print('%d %d%% (%s) %.4f %s / %s %s'% (i, i/n_iters * 100, time_since(start), loss, line, guess, correct))
            print('%d %d%% (%s) %.4f %s' % (i, i / n_iters * 100, time_since(start), loss, line))

        current_loss += loss
        if i % plot_every == 0:
            all_losses.append(current_loss / plot_every)
            current_loss = 0
    return rnn, all_losses


def plotting_loss(loss_list):
    plt.figure()
    plt.plot(loss_list)
    plt.show()


def predict(model: RNN, category, all_categories, start_letter="A", max_length=20):
    print('\n %s' % category)
    with torch.no_grad():
        category_tensor = category2tensor(all_categories, category)
        input_tensor = input_line2tensor(start_letter)
        hidden = model.init_hidden()

        output_name = start_letter
        for i in range(max_length):
            output, hidden = model(category_tensor, input_tensor[0], hidden)
            topv, topi = output.topk(1)
            topi = topi[0][0]
            if topi == n_letters - 1:  # eos
                break
            else:
                letter = all_letters[topi]
                output_name += letter
            input_tensor = input_line2tensor(letter)
        return output_name


def practice():
    path_ptn = os.path.join(base_dir, "data/names/names/*.txt")
    files = find_files(path_ptn)
    print(files)
    print(unicode2ascii('Ślusàrski'))
    a, b = load_data(path_ptn)
    print(len(a))
    print(b['Italian'][:5])
    print(letter2tensor('X'))
    print(all_letters, len(all_letters))
    print(output_line2tensor("XYZ"))
    print(input_line2tensor("ABC"))
    n_hidden = 128
    # rnn = RNN(n_letters, n_hidden, len(a))
    # inp = letter2tensor('A')
    # hidden = torch.zeros(1, n_hidden)
    # output, next_hidden = rnn(inp, hidden)
    # print(output)
    #
    # inp = input_line2tensor('Albert')
    # hidden = torch.zeros(1, n_hidden)
    # output, next_hidden = rnn(inp[1], hidden)
    # print(output)
    # print(next_hidden.shape)
    # print(type(output))
    # print(category_from_output(output, a))
    #
    # for i in range(10):
    #     cate, line, cate_tensor, line_tensor = random_training_example(a, b)
    #     print('category = ', cate, '/ line = ', line)


def main():
    path_pattern = os.path.join(base_dir, "data/names/names/*.txt")
    print(path_pattern)
    all_categories, category_line = load_data(path_pattern)
    print(all_categories, len(all_categories))

    # model, loss_list = train_epoch(all_categories, category_line)
    # plotting_loss(loss_list)
    # sampled = predict(model, 'Russian', all_categories)
    # print(sampled)
    # torch.save(model, 'rnn_generate_model.pth')

    reload_model = torch.load('rnn_generate_model.pth')
    sampled = predict(reload_model, 'Chinese', all_categories, start_letter="W")
    print(sampled)


if __name__ == '__main__':
    main()
    # practice()
