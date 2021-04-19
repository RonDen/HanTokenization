#!/usr/bin/python3
# -*- coding: utf-8 -*-
# Author:   Xukun Luo
# Date:     2021.04.08

import sys
sys.path.append("../")

import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from codes.layers import CRF

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

class BiLstmCRFModel(nn.Module):
    def __init__(self, args):
        super(BiLstmCRFModel, self).__init__()

        self.label_number = args.label_number
        self.lstm_layers = args.lstm_layers
        self.lstm_hidden = args.lstm_hidden
        self.lstm_dropout = args.lstm_dropout
        self.use_cuda = args.use_cuda
        self.embed_size = args.embed_size
        self.num_embeddings = args.vocab_len
        self.device = args.device
        
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
        self.crf = CRF(target_size=self.label_number,
                        average_batch=True,
                        use_cuda=self.use_cuda,
                        bad_pairs=[],
                        good_pairs=[])
        self.linear = nn.Linear(self.lstm_hidden*2, self.label_number+2)
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

    def get_mask(self, src_len, batch_size, seq_len):
        """ Generate the mask matrix. """
        src_range = torch.arange(0, seq_len).long()     # [0, 1, 2, 3, 4]
        src_len = torch.LongTensor(src_len)             # [3, 4, 5, 1]
        # [
        #   [0, 1, 2, 3, 4],
        #   [0, 1, 2, 3, 4],
        #   [0, 1, 2, 3, 4],
        #   [0, 1, 2, 3, 4]
        # ]
        src_range_expand = src_range.unsqueeze(0).expand(batch_size, seq_len)
        # [
        #   [3, 3, 3, 3, 3],
        #   [4, 4, 4, 4, 4],
        #   [5, 5, 5, 5, 5],
        #   [1, 1, 1, 1, 1]
        # ]
        src_len_expand = src_len.unsqueeze(1).expand_as(src_range_expand)
        # [
        #   [1, 1, 1, 0, 0],
        #   [1, 1, 1, 1, 0],
        #   [1, 1, 1, 1, 1],
        #   [1, 0, 0, 0, 0]
        # ]
        mask = src_range_expand < src_len_expand
        return mask
    
    def criterion(self, feats, src_len, labels):
        """
            CRF LOSS.

            Args:

                feats:      size=(batch_size, seq_len, tag_size)
                src_len:    size=(batch_size)
                tags:       size=(batch_size, seq_len)

            Returns:

                loss_value
        """ 
        batch_size, seq_len = feats.size(0), feats.size(1)

        # Generate the mask matrix.
        mask = self.get_mask(src_len, batch_size, seq_len)

        # Get loss.
        loss_value = self.crf.neg_log_likelihood_loss(feats, mask.long().to(self.device), labels)
        batch_size = feats.size(0)
        loss_value /= float(batch_size)
        return loss_value
    
    def get_best_path(self, feats, src_len):
        """ 
            Best path.

            Args:

                feats:      size=(batch_size, seq_len, tag_size)
                src_len:    size=(batch_size)
            
            Returns:

                best_path:  size=(batch_size, seq_len)
        """
        batch_size, seq_len = feats.size(0), feats.size(1)

        # Generate the mask matrix.
        mask = self.get_mask(src_len, batch_size, seq_len)

        # Get best path.
        path_score, best_path = self.crf(feats, mask.bool().to(self.device))

        return best_path

ModelDict = {
    "bilstm": BiLstmModel,
    "bilstmcrf": BiLstmCRFModel
}