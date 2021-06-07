import argparse
import random
import numpy as np
import random
from lf_evaluator import *
from models import *
from data import *
from utils import *
from train import *
from typing import List
import time
import matplotlib.pyplot as plt
from sklearn import model_selection

def _parse_args():
    parser = argparse.ArgumentParser(description='main.py')
    
    parser.add_argument('--train_path', type=str, default='data/geo_train.tsv', help='path to train data')
    parser.add_argument('--dev_path', type=str, default='data/geo_dev.tsv', help='path to dev data')
    parser.add_argument('--test_output_path', type=str, default='geo_test_output.tsv', help='path to write blind test results')
    parser.add_argument('--domain', type=str, default='geo', help='domain (geo for geoquery)')
    parser.add_argument('--no_java_eval', dest='perform_java_eval', default=True, action='store_false', help='run evaluation of constructed query against java backend')
    parser.add_argument('--blind', type=int, default=1)
    parser.add_argument('--seed', type=int, default=0, help='RNG seed (default = 0)')
    parser.add_argument('--epochs', type=int, default=10, help='num epochs to train for')
    parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
    parser.add_argument('--embedding_dim', type=int, default=150, help='embedding dimension')
    parser.add_argument('--hidden_size', type=int, default=100, help='hidden layer size')
    parser.add_argument('--batch_size', type=int, default=1, help='training batch size; 1 by default and you do not need to batch unless you want to')
    parser.add_argument('--dropout', type=float, default=0.2, help='dropout rate')
    parser.add_argument('--decoder_len_limit', type=int, default=65, help='output length limit of the decoder')
    parser.add_argument('--num_filters', type=int, default=2, help='number of decoders in the network')
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = _parse_args()
    print("domain %s, %i epochs, %i batch size, %i hidden dim, %i filters" %(args.domain, args.epochs, args.batch_size, args.hidden_size, args.num_filters))
    random.seed(args.seed)
    np.random.seed(args.seed)
    # Load the training and test data
    if args.domain == 'translate':
        train, dev = load_datasets1(('data/trans_train.en', 'data/trans_train.fr'), ('data/trans_test.en', 'data/trans_test.fr'))
        train_data_indexed, dev_data_indexed, input_indexer, output_indexer = index_datasets(train[:10000], dev, args.decoder_len_limit)
        print("%i train exs, %i dev exs, %i input types, %i output types" % (len(train_data_indexed), len(dev_data_indexed), len(input_indexer), len(output_indexer)))
    else:
        train, dev = load_datasets(args.train_path, args.dev_path, domain=args.domain)
        train_data_indexed, dev_data_indexed, input_indexer, output_indexer = index_datasets(train, dev, args.decoder_len_limit)
        print("%i train exs, %i dev exs, %i input types, %i output types" % (len(train_data_indexed), len(dev_data_indexed), len(input_indexer), len(output_indexer)))

    t = time.time()
    autoencoder = train_encoder(train_data_indexed, input_indexer, output_indexer, args)
    print('Training Time: ', time.time() - t)
    gm = train_gaussian_mixture(train_data_indexed, autoencoder, args.num_filters)
    #plot_latent(train_data_indexed, autoencoder, gm, args.num_filters)
    
    t1 = time.time()
    mgmae = train_decoders(train_data_indexed, input_indexer, output_indexer, autoencoder, gm, args.num_filters, args)
    print('Training Time: ', time.time() - t1)
    print("=======DEV SET=======")
    translation = args.domain == 'translate'
    evaluate(dev_data_indexed, mgmae, translation, use_java=args.perform_java_eval)

