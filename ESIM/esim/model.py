import os
import argparse
import pickle
import torch
import json
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.nn as nn
from esim.layers import RNNDropout, Seq2SeqEncoder, SoftmaxAttention
from esim.utils import get_mask, replace_masked
from tqdm import tqdm
from esim.data import NLIDataset

class ESIM(nn.Module):
    """
    Implementation of the ESIM model presented in the paper "Enhanced LSTM for
    Natural Language Inference" by Chen et al.
    """

    def __init__(self,
                 vocab_size,
                 embedding_dim,
                 hidden_size,
                 embeddings=None,
                 padding_idx=0,
                 dropout=0.5,
                 num_classes=3,
                 device="cpu"):
        super(ESIM, self).__init__()

        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.hidden_size = hidden_size
        self.num_classes = num_classes
        self.dropout = dropout
        self.device = device

        self._word_embedding = nn.Embedding(self.vocab_size,
                                            self.embedding_dim,
                                            padding_idx=padding_idx,
                                            _weight=embeddings)

        if self.dropout:
            self._rnn_dropout = RNNDropout(p=self.dropout)
            # self._rnn_dropout = nn.Dropout(p=self.dropout)

        self._encoding = Seq2SeqEncoder(nn.LSTM,
                                        self.embedding_dim,
                                        self.hidden_size,
                                        bidirectional=True)

        self._attention = SoftmaxAttention()

        self._projection = nn.Sequential(nn.Linear(4*2*self.hidden_size,
                                                   self.hidden_size),
                                         nn.ReLU())

        self._composition = Seq2SeqEncoder(nn.LSTM,
                                           self.hidden_size,
                                           self.hidden_size,
                                           bidirectional=True)

        self._classification = nn.Sequential(nn.Dropout(p=self.dropout),
                                             nn.Linear(2*4*self.hidden_size,
                                                       self.hidden_size),
                                             nn.Tanh(),
                                             nn.Dropout(p=self.dropout),
                                             nn.Linear(self.hidden_size,
                                                       self.num_classes))

    def forward(self,
                premises,
                premises_lengths,
                hypotheses,
                hypotheses_lengths):
        #print(premises.shape)
        #获取句子掩码，找出最长的句子对齐，有单词的部分为1，否则为0
        premises_mask = get_mask(premises, premises_lengths).to(self.device)
        hypotheses_mask = get_mask(hypotheses, hypotheses_lengths).to(self.device)
        #GLOVE做embedding
        embedded_premises = self._word_embedding(premises)
        embedded_hypotheses = self._word_embedding(hypotheses)
        #print(embedded_premises.shape)

        if self.dropout:
            embedded_premises = self._rnn_dropout(embedded_premises)
            embedded_hypotheses = self._rnn_dropout(embedded_hypotheses)
        #BiLSTM对输入做encode，要做pack_padded_sequence
        encoded_premises = self._encoding(embedded_premises,premises_lengths)
        encoded_hypotheses = self._encoding(embedded_hypotheses,hypotheses_lengths)
        #print(encoded_premises.shape)

        attended_premises,attended_hypotheses=self._attention(encoded_premises, premises_mask, encoded_hypotheses, hypotheses_mask)

        print(attended_premises.shape)

        #将encode结果和attention结果连接，公式14，15，更好反映句间关系,最后一位变成4倍
        enhanced_premises = torch.cat([encoded_premises,
                                       attended_premises,
                                       encoded_premises - attended_premises,
                                       encoded_premises * attended_premises],
                                      dim=-1)
        enhanced_hypotheses = torch.cat([encoded_hypotheses,
                                         attended_hypotheses,
                                         encoded_hypotheses -
                                         attended_hypotheses,
                                         encoded_hypotheses *
                                         attended_hypotheses],
                                        dim=-1)
        print(enhanced_premises.shape)

        #使用全连接层
        projected_premises = self._projection(enhanced_premises)
        projected_hypotheses = self._projection(enhanced_hypotheses)
        print(projected_premises.shape)
        return attended_premises
