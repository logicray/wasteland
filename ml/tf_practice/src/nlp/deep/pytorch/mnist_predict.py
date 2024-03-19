# !/usr/bin/python
# coding: utf-8


import torch
import torch.nn as nn
from torchvision import transforms
from torchvision.models import squeezenet1_1
from PIL import Image


def load_model(path):
    CUDA = torch.cuda.is_available()
    # module = squeezenet1_1(pretrained=True)#torchvision官方内置模型直接调用
    module = Net()#自己的模型调用
    module.load_state_dict(torch.load(r'./param/mnist_cnn.pkl'))
    module.eval()
    if CUDA:
        module.cuda()


def predict_img(img_path):
    img = Image.open(img_path)
    img2data = transforms.Compose([
        transforms.Resize(28),  # 因为这个模型是专门针对32*32的CIFAR10
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    data = img2data(img)
    '因为PyTorch将所有的图像当做批次,默认只处理四维数据'
    data = data.unsqueeze_(0)
    if CUDA:
        data = data.cuda()
    out = module(data)
    # print(out)
    _, index = torch.max(out, 1)
    return index.item()


if __name__ == '__main__':
    index = predict_img(r'C:\Users\87419\Desktop/00.jpg')
