import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
import random
from torch.autograd import Variable as Var
from utils import *
from data import *
from lf_evaluator import *
import numpy as np
from typing import List
import sklearn
from sklearn.mixture import GaussianMixture
from sklearn import decomposition

class MGMAE(nn.Module):
    def __init__(self, autoencoder, filters, mixture, out_max_length=99):
        super(MGMAE, self).__init__()
        self.autoencoder = autoencoder
        self.gm = mixture
        self.filters = filters
        self.out_max_length = out_max_length

    def decode(self, test_data: List[Example]) -> List[List[Derivation]]:
        derivations = []
        max_length = np.max(np.array([len(ex.x_indexed) for ex in test_data]))
        input_data = make_padded_input_tensor(test_data, self.autoencoder.input_indexer, max_length)
        for i in range(len(test_data)):
            ex = test_data[i]
            x_tensor = torch.from_numpy(input_data[i]).unsqueeze(0)
            inp_lens_tensor = torch.from_numpy(np.array(len(test_data[i].x_indexed))).unsqueeze(0)
            # encode the data
            enc_outputs, _, state = self.autoencoder.encode_input(x_tensor, inp_lens_tensor)
            # enc_outputs: 1 * batch * hid_size, state: 1 * hid_size
            enc_outputs = torch.swapaxes(enc_outputs, 0, 1)

            # determine filter
            filter_idx = self.gm.predict(state[0].squeeze(0).detach().numpy())[0] # int
            decoder = self.filters[filter_idx]

            word = torch.ones(len(state[0]), 1, dtype=torch.int) * decoder.indexer.index_of(SOS_SYMBOL)
            word_idx = decoder.indexer.index_of(SOS_SYMBOL)
            length = 0
            y_toks = []
            probability = 1
            while length < self.out_max_length:
                output, state = decoder(word, state, enc_outputs)
                # output: 1 * 1 * vocab_size
                word_idx = torch.argmax(output[0][0])
                # stoo when hit the EOS token
                if word_idx == decoder.indexer.index_of(EOS_SYMBOL):
                    break
                probability *= output[0][0][word_idx].item()
                y_toks.append(decoder.indexer.get_object(word_idx.item()))
                word = torch.tensor([word_idx]).unsqueeze(1)
                state = (state[0].unsqueeze(0), state[1].unsqueeze(0))
                length += 1
            derivations.append([Derivation(ex, probability, y_toks)])
        return derivations

class EmbeddingLayer(nn.Module):
    """
    Embedding layer that has a lookup table of symbols that is [full_dict_size x input_dim]. Includes dropout.
    Works for both non-batched and batched inputs
    """
    def __init__(self, input_dim: int, full_dict_size: int, embedding_dropout_rate: float):
        """
        :param input_dim: dimensionality of the word vectors
        :param full_dict_size: number of words in the vocabulary
        :param embedding_dropout_rate: dropout rate to apply
        """
        super(EmbeddingLayer, self).__init__()
        self.dropout = nn.Dropout(embedding_dropout_rate)
        self.word_embedding = nn.Embedding(full_dict_size, input_dim)

    def forward(self, inputs):
        """
        :param input: either a non-batched input [sent len x voc size] or a batched input
        [batch size x sent len x voc size]
        :return: embedded form of the input words (last coordinate replaced by input_dim)
        """
        embedded_words = self.word_embedding(inputs)
        final_embeddings = self.dropout(embedded_words)
        return final_embeddings


class Autoencoder(nn.Module):
    def __init__(self, input_indexer, output_indexer, emb_dim, hidden_size, attention, embedding_dropout=0.2, bidirect=True):
        super(Autoencoder, self).__init__()
        self.input_indexer = input_indexer
        self.bidirect = bidirect

        self.encoder = Encoder(len(input_indexer), emb_dim, embedding_dropout, hidden_size, bidirect)
        self.decoder = Decoder(output_indexer, emb_dim, embedding_dropout, hidden_size, bidirect, attention)
        self.loss = nn.NLLLoss()

    def sent_lens_to_mask(self, lens, max_length):
        return torch.from_numpy(np.asarray([[1 if j < lens.data[i].item() else 0 for j in range(0, max_length)] for i in range(0, lens.shape[0])]))

    def mask(self, vector, context_mask, seq_idx):
        for i in range(len(context_mask)):
            vector[i] *= context_mask[i, seq_idx]
        return vector

    def encode_input(self, x_tensor, inp_lens_tensor):
        (enc_output_each_word, enc_context_mask, enc_final_states) = self.encoder.forward(x_tensor, inp_lens_tensor)
        enc_final_states_reshaped = (enc_final_states[0].unsqueeze(0), enc_final_states[1].unsqueeze(0))
        return (enc_output_each_word, enc_context_mask, enc_final_states_reshaped)

    def forward(self, x_tensor, inp_lens_tensor, y_tensor, out_lens_tensor):
        """
        :param x_tensor/y_tensor: either a non-batched input/output [sent len x voc size] or a batched input/output
        [batch size x sent len x voc size]
        :param inp_lens_tensor/out_lens_tensor: either a vecor of input/output length [batch size] or a single integer.
        lengths aren't needed if you don't batchify the training.
        :return: loss of the batch
        """
        max_length = torch.max(out_lens_tensor).item()
        context_mask = self.sent_lens_to_mask(out_lens_tensor, max_length)
        # context_mask: batch * max_seq_len
        loss = 0
        enc_output_each_word, enc_context_mask, state = self.encode_input(x_tensor, inp_lens_tensor)
        # max_seq_len * batch * hid_size, batch * max_seq_len, batch * hid_size
        enc_output_each_word = torch.swapaxes(enc_output_each_word, 0, 1)

        word_idx = self.decoder.indexer.index_of(SOS_SYMBOL)
        word = torch.ones(len(x_tensor), 1, dtype=torch.int) * word_idx
        for i in range(max_length):
            output, state = self.decoder(word, state, enc_output_each_word)
            # output: batch * 1 * vocab_size
            output = self.mask(output.clone(), context_mask, i)
            labels = y_tensor[:, i]
            loss += self.loss(output.squeeze(1), labels)
            word = labels.unsqueeze(1)
            state = (state[0].unsqueeze(0), state[1].unsqueeze(0))
        return loss

class Encoder(nn.Module):
    def __init__(self, vocab_size, emb_dim, embedding_dropout, hidden_size: int, bidirect: bool):
        """
        :param emb_dim: size of word embeddings output by embedding layer
        :param hidden_size: hidden size for the LSTM
        :param bidirect: True if bidirectional, false otherwise
        """
        super(Encoder, self).__init__()
        self.embedding = EmbeddingLayer(emb_dim, vocab_size, embedding_dropout)
        self.bidirect = bidirect
        self.hidden_size = hidden_size
        self.reduce_h_W = nn.Linear(hidden_size * 2, hidden_size, bias=True)
        self.reduce_c_W = nn.Linear(hidden_size * 2, hidden_size, bias=True)
        self.rnn = nn.LSTM(emb_dim, hidden_size, num_layers=1, batch_first=True,
                               dropout=0., bidirectional=self.bidirect)
        self.init_weight()

    def init_weight(self):
        """
        Initializes weight matrices using Xavier initialization
        :return:
        """
        nn.init.xavier_uniform_(self.rnn.weight_hh_l0, gain=1)
        nn.init.xavier_uniform_(self.rnn.weight_ih_l0, gain=1)
        if self.bidirect:
            nn.init.xavier_uniform_(self.rnn.weight_hh_l0_reverse, gain=1)
            nn.init.xavier_uniform_(self.rnn.weight_ih_l0_reverse, gain=1)
        nn.init.constant_(self.rnn.bias_hh_l0, 0)
        nn.init.constant_(self.rnn.bias_ih_l0, 0)
        if self.bidirect:
            nn.init.constant_(self.rnn.bias_hh_l0_reverse, 0)
            nn.init.constant_(self.rnn.bias_ih_l0_reverse, 0)

    def sent_lens_to_mask(self, lens, max_length):
        return torch.from_numpy(np.asarray([[1 if j < lens.data[i].item() else 0 for j in range(0, max_length)] for i in range(0, lens.shape[0])]))

    def forward(self, x_tensor, input_lens):
        embedded_words = self.embedding.forward(x_tensor)
        packed_embedding = nn.utils.rnn.pack_padded_sequence(embedded_words, input_lens, batch_first=True, enforce_sorted=False)
        output, hn = self.rnn(packed_embedding)
        output, sent_lens = nn.utils.rnn.pad_packed_sequence(output)
        max_length = torch.max(input_lens).item()
        context_mask = self.sent_lens_to_mask(sent_lens, max_length)

        if self.bidirect:
            h, c = hn[0], hn[1]
            h_, c_ = torch.cat((h[0], h[1]), dim=1), torch.cat((c[0], c[1]), dim=1)
            new_h = self.reduce_h_W(h_)
            new_c = self.reduce_c_W(c_)
            h_t = (new_h, new_c)
        else:
            h, c = hn[0][0], hn[1][0]
            h_t = (h, c)
        return (output, context_mask, h_t)

class Decoder(nn.Module):
    def __init__(self, indexer, emb_dim, embedding_dropout, hidden_size: int, bidirect: bool, attention: bool):
        """
        :param emb_dim: size of word embeddings output by embedding layer
        :param hidden_size: hidden size for the LSTM
        :param bidirect: True if bidirectional, false otherwise
        """
        super(Decoder, self).__init__()
        self.indexer = indexer
        self.embedding = EmbeddingLayer(emb_dim, len(indexer), embedding_dropout)
        self.hidden_size = hidden_size
        self.bidirect = bidirect
        self.attention = attention
        self.rnn = nn.LSTM(emb_dim, hidden_size, num_layers=1, batch_first=True,
                           dropout=0., bidirectional=False)
        self.reduce_h_W = nn.Linear(2 * hidden_size, hidden_size)
        coef = 2 if attention else 1
        self.h2o = nn.Linear(coef * hidden_size, len(indexer))
        self.softmax = nn.LogSoftmax(dim=1)
        self.loss = nn.NLLLoss()
        self.init_weight()

    def init_weight(self):
        """
        Initializes weight matrices using Xavier initialization
        :return:
        """
        nn.init.xavier_uniform_(self.rnn.weight_hh_l0, gain=1)
        nn.init.xavier_uniform_(self.rnn.weight_ih_l0, gain=1)
        nn.init.constant_(self.rnn.bias_hh_l0, 0)
        nn.init.constant_(self.rnn.bias_ih_l0, 0)

    def forward(self, x_tensor, hidden, encoder_outputs):
        '''
        hidden: (1 * batch * hid_size, 1 * batch * hid_size)
        encoder_outputs: batch * 1 * vocab_size
        '''
        embedded_words = self.embedding.forward(x_tensor)
        if self.bidirect:
            encoder_outputs = self.reduce_h_W(encoder_outputs)
        output, hn = self.rnn(embedded_words, hidden)
        h, c = hn[0][0], hn[1][0] # h, c: batch * hid_size
        if self.attention:
            hid = h.unsqueeze(1) # batch * 1 * hid_size
            enc_out = torch.swapaxes(encoder_outputs, 1, 2) # batch * hid_size * seq_len
            attn_dot_product = torch.matmul(hid, enc_out).squeeze(1)
            # attn_dot_product: batch * seq_len
            attn_weight = F.softmax(attn_dot_product, dim=1)
            # attn_weight: batch * seq_len
            context = torch.matmul(attn_weight.unsqueeze(1), encoder_outputs).squeeze(1)
            # context: batch * hid_size
            out = torch.cat([h, context], dim=1)
            # out: batch * (2 * hid_size)
            output = self.softmax(self.h2o(out)).unsqueeze(1)
        else:
            output = self.softmax(self.h2o(output))

        h_t = (h, c)
        return (output, h_t)

    def sent_lens_to_mask(self, lens, max_length):
        return torch.from_numpy(np.asarray([[1 if j < lens.data[i].item() else 0 for j in range(0, max_length)] for i in range(0, lens.shape[0])]))

    def mask(self, vector, context_mask, seq_idx):
        for i in range(len(context_mask)):
            vector[i] *= context_mask[i, seq_idx]
        return vector

    def train(self, enc_outputs, state, y_tensor, out_lens_tensor):
        max_length = torch.max(out_lens_tensor).item()
        context_mask = self.sent_lens_to_mask(out_lens_tensor, max_length)
        # context_mask: batch * max_seq_len
        loss = 0
        enc_outputs = torch.swapaxes(enc_outputs, 0, 1)
        # batch * max_seq_len * hid_size

        sos_idx = self.indexer.index_of(SOS_SYMBOL)
        word = torch.ones(len(y_tensor), 1, dtype=torch.int) * sos_idx
        # word: batch * 1
        for i in range(max_length):
            output, state = self.forward(word, state, enc_outputs)
            # output: batch * 1 * vocab_size
            output = self.mask(output.clone(), context_mask, i)
            labels = y_tensor[:, i]
            loss += self.loss(output.squeeze(1), labels)
            word = labels.unsqueeze(1)
            state = (state[0].unsqueeze(0), state[1].unsqueeze(0))
        return loss


