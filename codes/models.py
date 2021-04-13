#!/usr/bin/python3
# -*- coding: utf-8 -*-
# Author:   Xukun Luo
# Date:     2021.04.08

import sys
sys.path.append("../")

import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

class BiLstmModel(nn.Module):
    def __init__(self, args):
        super(BiLstmModel, self).__init__()

        self.label_number = args.label_number
        self.lstm_layers = args.lstm_layers
        self.lstm_hidden = args.lstm_hidden
        self.lstm_dropout = args.lstm_dropout
        self.use_cuda = args.use_cuda
        self.embed_size = args.embed_size
        self.num_embeddings = args.vocab_len
        
        self.embedding = nn.Embedding(self.num_embeddings, self.embed_size)
        self.lstm_encoder = nn.LSTM(input_size=self.embed_size,
                            hidden_size=self.lstm_hidden,
                            num_layers=self.lstm_layers,
                            bidirectional=True,
                            dropout=self.lstm_dropout,
                            batch_first=True)
        self.lstm_decoder = nn.LSTM(input_size=self.lstm_hidden*2,
                            hidden_size=self.lstm_hidden,
                            num_layers=self.lstm_layers,
                            bidirectional=True,
                            dropout=self.lstm_dropout,
                            batch_first=True)
        self.linear = nn.Linear(self.lstm_hidden*2, self.label_number)
        self.droplayer = nn.Dropout(p=self.lstm_dropout)

    def forward(self, src, src_len):
        '''
            Forward Algorithm.

            Args:

                src (batch_size, seq_length) : word-level representation of sentence
                src_len (batch_size)         : the sentence length

            Returns:

                feats (batch_size, seq_length, num_labels) : predect feats.
        '''
        batch_size, seq_len = src.size(0), src.size(1)
        # Embedding.
        emb = self.embedding(src)
        emb = pack_padded_sequence(emb, src_len, True)
        # Encoder. (batch_size, seq_length, lstm_hidden*2)
        context_vector, _ = self.lstm_encoder(emb)
        #context_vector = self.droplayer(context_vector)
        # Decoder. (batch_size, seq_length, lstm_hidden*2)
        lstm_out, hidden = self.lstm_decoder(context_vector)
        lstm_out, _ = pad_packed_sequence(lstm_out, True)
        lstm_out = lstm_out.contiguous().view(-1, self.lstm_hidden*2)
        lstm_out = self.droplayer(lstm_out)
        # Linear layer. (batch_size, seq_length, label_number)
        lstm_feats = self.linear(lstm_out).view(batch_size, seq_len, -1)
        return lstm_feats

ModelDict = {
    "bilstm": BiLstmModel
}