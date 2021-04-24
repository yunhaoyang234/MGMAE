# utils.py
import torch
from typing import List
import numpy as np
import glob
import matplotlib.pyplot as plt
import sklearn
import pickle
from sklearn import decomposition
from sklearn import metrics

class Example(object):
    """
    Wrapper class for a single (natural language, logical form) input/output (x/y) pair
    Attributes:
        x: the natural language as one string
        x_tok: tokenized natural language as a list of strings
        x_indexed: indexed tokens, a list of ints
        y: the raw logical form as a string
        y_tok: tokenized logical form, a list of strings
        y_indexed: indexed logical form, a list of ints
    """
    def __init__(self, x: str, x_tok: List[str], x_indexed: List[int], y, y_tok, y_indexed):
        self.x = x
        self.x_tok = x_tok
        self.x_indexed = x_indexed
        self.y = y
        self.y_tok = y_tok
        self.y_indexed = y_indexed

    def __repr__(self):
        return " ".join(self.x_tok) + " => " + " ".join(self.y_tok) + "\n   indexed as: " + repr(self.x_indexed) + " => " + repr(self.y_indexed)

    def __str__(self):
        return self.__repr__()


#
class Derivation(object):
    """
    Wrapper for a possible solution returned by the model associated with an Example. Note that y_toks here is a
    predicted y_toks, and the Example itself contains the gold y_toks.
    Attributes:
          example: The underlying Example we're predicting on
          p: the probability associated with this prediction
          y_toks: the tokenized output prediction
    """
    def __init__(self, example: Example, p, y_toks):
        self.example = example
        self.p = p
        self.y_toks = y_toks

    def __str__(self):
        return "%s (%s)" % (self.y_toks, self.p)

    def __repr__(self):
        return self.__str__()


PAD_SYMBOL = "<PAD>"
UNK_SYMBOL = "<UNK>"
SOS_SYMBOL = "<SOS>"
EOS_SYMBOL = "<EOS>"

class Indexer(object):
    """
    Bijection between objects and integers starting at 0. Useful for mapping
    labels, features, etc. into coordinates of a vector space.

    Attributes:
        objs_to_ints
        ints_to_objs
    """
    def __init__(self):
        self.objs_to_ints = {}
        self.ints_to_objs = {}

    def __repr__(self):
        return str([str(self.get_object(i)) for i in range(0, len(self))])

    def __str__(self):
        return self.__repr__()

    def __len__(self):
        return len(self.objs_to_ints)

    def get_object(self, index):
        """
        :param index: integer index to look up
        :return: Returns the object corresponding to the particular index or None if not found
        """
        if (index not in self.ints_to_objs):
            return None
        else:
            return self.ints_to_objs[index]

    def contains(self, object):
        """
        :param object: object to look up
        :return: Returns True if it is in the Indexer, False otherwise
        """
        return self.index_of(object) != -1

    def index_of(self, object):
        """
        :param object: object to look up
        :return: Returns -1 if the object isn't present, index otherwise
        """
        if (object not in self.objs_to_ints):
            return -1
        else:
            return self.objs_to_ints[object]

    def add_and_get_index(self, object, add=True):
        """
        Adds the object to the index if it isn't present, always returns a nonnegative index
        :param object: object to look up or add
        :param add: True by default, False if we shouldn't add the object. If False, equivalent to index_of.
        :return: The index of the object
        """
        if not add:
            return self.index_of(object)
        if (object not in self.objs_to_ints):
            new_idx = len(self.objs_to_ints)
            self.objs_to_ints[object] = new_idx
            self.ints_to_objs[new_idx] = object
        return self.objs_to_ints[object]

def make_padded_input_tensor(exs: List[Example], input_indexer: Indexer, max_len: int, reverse_input=False) -> np.ndarray:
    """
    Takes the given Examples and their input indexer and turns them into a numpy array by padding them out to max_len.
    Optionally reverses them.
    :param exs: examples to tensor-ify
    :param input_indexer: Indexer over input symbols; needed to get the index of the pad symbol
    :param max_len: max input len to use (pad/truncate to this length)
    :param reverse_input: True if we should reverse the inputs (useful if doing a unidirectional LSTM encoder)
    :return: A [num example, max_len]-size array of indices of the input tokens
    """
    if reverse_input:
        return np.array(
            [[ex.x_indexed[len(ex.x_indexed) - 1 - i] if i < len(ex.x_indexed) else input_indexer.index_of(PAD_SYMBOL)
              for i in range(0, max_len)]
             for ex in exs])
    else:
        return np.array([[ex.x_indexed[i] if i < len(ex.x_indexed) else input_indexer.index_of(PAD_SYMBOL)
                          for i in range(0, max_len)]
                         for ex in exs])

def make_padded_output_tensor(exs, output_indexer, max_len):
    """
    Similar to make_padded_input_tensor, but does it on the outputs without the option to reverse input
    :param exs:
    :param output_indexer:
    :param max_len:
    :return: A [num example, max_len]-size array of indices of the output tokens
    """
    return np.array([[ex.y_indexed[i] if i < len(ex.y_indexed) else output_indexer.index_of(PAD_SYMBOL) for i in range(0, max_len)] for ex in exs])

def plot_latent(data, autoencoder, gm, num_filters):
    input_lens = torch.tensor([len(ex.x_indexed) for ex in data])
    input_max_len = torch.max(input_lens).item()
    x_tensor = make_padded_input_tensor(data, autoencoder.input_indexer, input_max_len, reverse_input=False)
    (o, c, hn) = autoencoder.encoder(torch.tensor(x_tensor), input_lens)
    X = hn[0].detach().numpy()
    labels = gm.predict(X)
    sil_score = metrics.silhouette_score(X, labels)
    print('Silhouette Score: ', sil_score)
    pca =  decomposition.PCA(n_components=2)
    X = pca.fit_transform(X)
    colors=['red', 'blue', 'green', 'purple']
    for i in range(num_filters):
        plt.scatter(X[labels==i, 0], X[labels==i, 1], s=5, c=colors[i])
    plt.show()

def save_model(model, path):
    torch.save(model, path)

def load_model(path):
    return torch.load(path)

def store_skmodel(model, path):
    filename = path + '.sav'
    pickle.dump(model, open(filename, 'wb'))

def load_skmodel(path):
    return pickle.load(open(path, 'rb'))

def save_models(autoencoder, filters, gm, dir_path='models/'):
    save_model(autoencoder, dir_path+'autoencoder')
    store_skmodel(gm, dir_path+'gaussian_mix')
    for i in range(len(filters)):
        save_model(fil[i], dir_path+'filter_'+str(i))

def load_models(dir_path='models/'):
    autoencoder = load_model(dir_path + 'autoencoder')
    gm = load_skmodel(dir_path + 'gaussian_mix.sav')
    files = glob.glob(dir_path  + '*')
    filters = []
    for i in range(len(files) - 1):
        filters.append(load_model(dir_path + 'filter_' + str(i)))
    return autoencoder, filters
