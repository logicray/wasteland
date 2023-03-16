#!/usr/bin/env python3
# -*- coding:utf8 -*-
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
import numpy as np
import time


class EncoderDecoder(nn.Module):
    """encoder-decoder 架构"""

    def __init__(self, encoder, decoder, src_embedding, target_embedding, generator):
        super(EncoderDecoder, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.src_embedding = src_embedding
        self.target_embedding = target_embedding
        self.generator = generator

    def encode(self, src, src_mask):
        return self.encoder(self.src_embedding(src), src_mask)

    def decode(self, memory, src_mask, target, target_mask):
        return self.decoder(self.target_embedding(target), memory, src_mask, target_mask)

    def forward(self, src, target, src_mask, target_mask):
        return self.decode(self.encode(src, src_mask), src_mask, target, target_mask)


class Generator(nn.Module):
    def __init__(self, d_model, target_vocab):
        super(Generator, self).__init__()
        self.proj = nn.Linear(d_model, target_vocab).cuda()

    def forward(self, x):
        return F.log_softmax(self.proj(torch.tensor(x, dtype=torch.float32)), dim=-1)


def clones(module, N):
    """clone n layers"""
    return nn.ModuleList(copy.deepcopy(module) for _ in range(N))


class LayerNorm(nn.Module):
    def __init__(self, features, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.a_2 = nn.Parameter(torch.ones(features).cuda())
        self.b_2 = nn.Parameter(torch.zeros(features).cuda())
        self.eps = eps

    def forward(self, x):
        x = torch.tensor(x, dtype=torch.float32)
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2


class SublayerConnection(nn.Module):
    def __init__(self, size, dropout):
        super(SublayerConnection, self).__init__()
        self.norm = LayerNorm(size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer):
        return x + self.dropout(sublayer(self.norm(x)))


class Encoder(nn.Module):
    def __init__(self, layer, N):
        super(Encoder, self).__init__()
        self.layers = clones(layer, N)
        self.norm = LayerNorm(layer.size)

    def forward(self, x, mask):
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)


class EncoderLayer(nn.Module):
    def __init__(self, size, self_attn, ffn, dropout):
        super(EncoderLayer, self).__init__()
        self.size = size
        self.self_attn = self_attn
        self.ffn = ffn
        self.sublayer = clones(SublayerConnection(size, dropout), 2)

    def forward(self, x, mask):
        x = self.sublayer[0](x, lambda a: self.self_attn(a, a, a, mask))
        return self.sublayer[1](x, self.ffn)


class Decoder(nn.Module):
    def __init__(self, layer, N):
        super(Decoder, self).__init__()
        self.layers = clones(layer, N)
        self.norm = LayerNorm(layer.size)

    def forward(self, x, memory, src_mask, target_mask):
        for layer in self.layers:
            x = layer(x, memory, src_mask, target_mask)
        return self.norm(x)


class DecodeLayer(nn.Module):
    """a single decode layer made of self attention, source attention and feed forward"""

    def __init__(self, size, self_attn, src_attn, ffn, dropout):
        super(DecodeLayer, self).__init__()
        self.size = size
        self.self_attn = self_attn
        self.src_attn = src_attn
        self.ffn = ffn
        self.sublayer = clones(SublayerConnection(size, dropout), 3)

    def forward(self, x, memory, src_mask, target_mask):
        x = self.sublayer[0](x, lambda a: self.self_attn(a, a, a, target_mask))
        x = self.sublayer[1](x, lambda b: self.src_attn(b, memory, memory, src_mask))
        return self.sublayer[2](x, self.ffn)


def attention(query, key, value, mask, dropout):
    d_k = query.size(-1)
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)
    p_attn = F.softmax(scores, dim=-1)
    if dropout is not None:
        p_attn = dropout(p_attn)
    return torch.matmul(p_attn, value), p_attn


class MultiHeadedAttn(nn.Module):
    def __init__(self, h, d_model, dropout=0.1):
        super(MultiHeadedAttn, self).__init__()
        assert d_model % h == 0
        self.d_k = d_model // h
        self.attn = None
        self.h = h
        self.linears = clones(nn.Linear(d_model, d_model).cuda(), 4)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, query, key, value, mask=None):
        if mask is not None:
            mask = mask.unsqueeze(1)
        n_batches = query.size(0)
        query, key, value = [
            l(torch.tensor(x.cpu().detach().numpy().astype(np.float32)).cuda()).view(n_batches, self.h, -1, self.d_k) for l, x in
            zip(self.linears, (query, key, value))]
        x, self.attn = attention(query, key, value, mask, self.dropout)
        x = x.transpose(1, 2).contiguous().view(n_batches, -1, self.h * self.d_k)
        return self.linears[-1](x)


class PositionWiseFFN(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):
        super(PositionWiseFFN, self).__init__()
        self.w1 = nn.Linear(d_model, d_ff).cuda()
        self.w2 = nn.Linear(d_ff, d_model).cuda()
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x):
        return self.w2(self.dropout(F.relu(self.w1(torch.tensor(x, dtype=torch.float32)))))


class Embedding(nn.Module):
    def __init__(self, d_model, vocab):
        super(Embedding, self).__init__()
        self.d_model = d_model  # dim of embedding
        self.lut = nn.Embedding(vocab, d_model).cuda()

    def forward(self, x):
        return self.lut(x) * math.sqrt(self.d_model)


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0., max_len).unsqueeze(1)
        tmp = torch.arange(0., d_model, 2) * (-(math.log(10000.0) / d_model))
        div_term = torch.exp(tmp)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + torch.autograd.Variable(self.pe[:, :x.size(1)].cuda(), requires_grad=False)
        return self.dropout(x)


def make_model(src_vocab, target_vocab, N=6, d_model=512, d_ff=2048, h=8, dropout=0.1):
    attn = MultiHeadedAttn(h, d_model)
    ff = PositionWiseFFN(d_model, d_ff, dropout)
    position = PositionalEncoding(d_model, dropout)

    cp = copy.deepcopy
    model = EncoderDecoder(
        Encoder(EncoderLayer(d_model, cp(attn), cp(ff), dropout), N),
        Decoder(DecodeLayer(d_model, cp(attn), cp(attn), cp(ff), dropout), N),
        nn.Sequential(Embedding(d_model, src_vocab), cp(position)),
        nn.Sequential(Embedding(d_model, target_vocab), cp(position)),
        Generator(d_model, target_vocab)
    )

    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform(p)
    return model


def test_multi_headed_attn():
    batch_size = 16
    L = 50
    d_model = 512
    h = 8
    x = torch.randn(batch_size, L, d_model)
    print(x.size())


class SimpleLossCompute(object):
    def __init__(self, generator, criterion, opt=None):
        self.generator = generator
        self.criterion = criterion
        self.opt = opt

    def __call__(self, x, y, norm):
        x = self.generator(x)
        loss = self.criterion(x.contiguous().view(-1, x.size(-1)),
                              y.contiguous().view(-1)) / norm
        loss.backward()
        if self.opt is not None:
            self.opt.step()
            self.opt.optimizer.zero_grad()

        return loss * norm.float()


class LabelSmoothing(nn.Module):
    def __init__(self, size, padding_idx, smoothing):
        super(LabelSmoothing, self).__init__()
        self.criterion = nn.KLDivLoss(reduction='sum')
        self.padding_idx = padding_idx
        self.smoothing = smoothing
        self.confidence = 1.0 - smoothing
        self.size = size
        self.true_dist = None

    def forward(self, x, target):
        assert x.size(1) == self.size
        true_dist = x.data.clone()
        true_dist.fill_(self.smoothing / (self.size - 2))
        tmp = torch.tensor(target.data.unsqueeze(1), dtype=torch.int64)
        true_dist.scatter_(1, tmp, self.confidence)
        true_dist[:, self.padding_idx] = 0
        mask = torch.nonzero(target.data == self.padding_idx)
        if mask.dim() > 0:
            true_dist.index_fill_(0, mask.squeeze(), 0.0)
        self.true_dist = true_dist
        return self.criterion(x, torch.autograd.Variable(true_dist, requires_grad=False))


class NoamOpt(object):
    "Optim wrapper that implements rate."

    def __init__(self, model_size, factor, warmup, optimizer):
        self.optimizer = optimizer
        self._step = 0
        self.warmup = warmup
        self.factor = factor
        self.model_size = model_size
        self._rate = 0

    def step(self):
        "Update parameters and rate"
        self._step += 1
        rate = self.rate()
        for p in self.optimizer.param_groups:
            p['lr'] = rate
        self._rate = rate
        self.optimizer.step()

    def rate(self, step=None):
        "Implement `lrate` above"
        if step is None:
            step = self._step
        return self.factor * \
               (self.model_size ** (-0.5) *
                min(step ** (-0.5), step * self.warmup ** (-1.5)))


def get_std_opt(model):
    return NoamOpt(model.src_embed[0].d_model, 2, 4000,
                   torch.optim.Adam(model.parameters(), lr=0, betas=(0.9, 0.98), eps=1e-9))


def data_gen(V, batch, nbatches):
    """Generate random data for a src-tgt copy task.## 数据生成"""
    for i in range(nbatches):
        data = torch.from_numpy(np.random.randint(1, V, size=(batch, 10), dtype=np.int32))
        data[:, 0] = 1
        src = torch.autograd.Variable(data, requires_grad=False)
        tgt = torch.autograd.Variable(data, requires_grad=False)
        yield Batch(src, tgt, 0)


class Batch(object):
    """定义一个训练时需要的批次数据对象，封装了用于训练的src和tgt句子，以及mask"""

    def __init__(self, src, trg=None, pad=0):
        self.src = src  # B 个序列[1,5,3, 0]
        self.src_mask = (src != pad).unsqueeze(-2)  # [[1,1,1,0]]
        if trg is not None:
            self.trg = trg[:, :-1]  #
            self.trg_y = trg[:, 1:]  # 后挪一个位置开始
            self.trg_mask = \
                self.make_std_mask(self.trg, pad)
            self.ntokens = (self.trg_y != pad).data.sum()

    @staticmethod
    def make_std_mask(tgt, pad):
        "Create a mask to hide padding and future words."
        tgt_mask = (tgt != pad).unsqueeze(-2)
        tgt_mask = tgt_mask & torch.autograd.Variable(
            subsequent_mask(tgt.size(-1)).type_as(tgt_mask.data))
        return tgt_mask


def subsequent_mask(size):
    """
    mask后续的位置，返回[size, size]尺寸下三角Tensor
    对角线及其左下角全是1，右上角全是0
    """
    attn_shape = (1, size, size)
    subsequent_mask = np.triu(np.ones(attn_shape), k=1).astype('uint8')
    return torch.from_numpy(subsequent_mask) == 0


def run_epoch(epoch_cnt, data_iter, model, loss_compute, device):
    """提供训练和日志功能"""
    start = time.time()
    total_tokens = 0
    total_loss = 0
    tokens = 0
    for i, batch in enumerate(data_iter):
        src = batch.src.to(device)
        trg = batch.trg.to(device)
        src_mask = batch.src_mask.to(device)
        trg_mask = batch.trg_mask.to(device)
        trg_y = batch.trg_y.to(device)
        ntokens = batch.ntokens.to(device)

        out = model.forward(src, trg, src_mask, trg_mask)
        loss = loss_compute(out, trg_y, ntokens)
        # 必须加上.cpu().numpy() 否则报错floating point exception (core dumped)
        total_loss += loss.detach().cpu().numpy()
        total_tokens += ntokens.cpu().numpy()
        tokens += ntokens.cpu().numpy()
        if i % 20 == 0:
            elapsed = time.time() - start
            print("Epoch:%d, Step: %d Loss: %f, Tokens per Sec: %f" %
                  (epoch_cnt, i, loss.detach().cpu().numpy() / ntokens.cpu().numpy(), tokens / elapsed))
            start = time.time()
            tokens = 0
    return total_loss / total_tokens


def main():
    V = 11
    model = make_model(V, V, N=2)
    # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    device = torch.device("cuda:0")
    criterion = LabelSmoothing(size=V, padding_idx=0, smoothing=0.1)
    model_opt = NoamOpt(model.src_embedding[0].d_model, 1, 400,
                        torch.optim.Adam(model.parameters(), lr=0, betas=(0.9, 0.98), eps=1e-9))

    for epoch in range(10):
        model.train()
        loss_func = SimpleLossCompute(model.generator, criterion, model_opt)
        run_epoch(epoch, data_gen(V, 30, 100), model, loss_func, device)
        model.eval()
        print(run_epoch(epoch, data_gen(V, 30, 5), model,
                        SimpleLossCompute(model.generator, criterion, None), device))


if __name__ == '__main__':
    print(torch.cuda.is_available())
    test_multi_headed_attn()
    main()
