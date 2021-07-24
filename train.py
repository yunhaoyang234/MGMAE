# train.py
from utils import *
from models import *
import copy
import sklearn
from sklearn.mixture import GaussianMixture
from sklearn.pipeline import Pipeline
from sklearn import decomposition

def train_encoder(train_data: List[Example], input_indexer, output_indexer, args):
    HIDDEN_SIZE = args.hidden_size
    EMBEDDING_DIM = args.embedding_dim
    LR = args.lr
    BATCH_SIZE = args.batch_size
    EPOCH = args.epochs
    DROPOUT = args.dropout
    MAX_LEN = args.decoder_len_limit

    autoencoder = Autoencoder(input_indexer, output_indexer, EMBEDDING_DIM, HIDDEN_SIZE, 
                              attention=True, embedding_dropout=DROPOUT)
    optimizer = optim.Adam(autoencoder.parameters(), lr=LR)
    for i in range(EPOCH):
        random.shuffle(train_data)
        total_loss = 0
        for j in range(0, len(train_data), BATCH_SIZE):
            autoencoder.zero_grad()
            batch_exs = train_data[j: j+BATCH_SIZE]

            x_inp_len, y_inp_len = [],[]
            for ex in batch_exs:
                x_inp_len.append(len(ex.x_tok))
                y_inp_len.append(len(ex.y_tok)+1) # include EOS
            x_tensor = make_padded_input_tensor(batch_exs, input_indexer, max(x_inp_len), reverse_input=False)
            y_tensor = make_padded_output_tensor(batch_exs, output_indexer, max(y_inp_len))
            x_inp_len, y_inp_len = torch.tensor(x_inp_len), torch.tensor(y_inp_len)

            loss = autoencoder.forward(torch.tensor(x_tensor), x_inp_len, torch.tensor(y_tensor), y_inp_len)
            total_loss += loss
            loss.backward()
            optimizer.step()
        print('Epoch ', i+1, total_loss)
    return autoencoder

def train_cluster_classifier(train_data: List[Example], autoencoder, input_indexer, hidden_size, num_filters):
    cluster_classifier = nn.Sequential()
    cluster_classifier.add_module('linear1', nn.Linear(hidden_size, hidden_size//2))
    cluster_classifier.add_module('linear2', nn.Linear(hidden_size//2, num_filters))
    cluster_classifier.add_module('activate', nn.GELU())
    cluster_classifier.add_module('softmax', nn.LogSoftmax(dim=1))

    input_lens = torch.tensor([len(ex.x_indexed) for ex in train_data])
    input_max_len = torch.max(input_lens).item()
    x_tensor = make_padded_input_tensor(train_data, autoencoder.input_indexer, input_max_len, reverse_input=False)
    (o, c, hn) = autoencoder.encoder(torch.tensor(x_tensor), input_lens)

    env = DummyVecEnv([lambda: Latent_Env(cluster_classifier, hn, 0.55, hidden_size, num_filters)])
    rl_model = SAC('MlpPolicy', env, verbose=1, learning_rate=0.001)
    rl_model.learn(total_timesteps=100)

    return cluster_classifier

def determine_filter(autoencoder, cluster_classifier, num_filter, x_tensor, inp_lens_tensor, y_tensor, out_lens_tensor):
    enc_outputs, enc_cm, state = autoencoder.encode_input(x_tensor, inp_lens_tensor)
    # max_seq_len * batch * hid_size, batch * max_seq_len, 1 * batch * hid_size
    labels = torch.argmax(cluster_classifier(state[0].squeeze(0)), dim=1)

    enc_outputs_list, latent_list, y_list, out_len_list = [], [], [], []
    for i in range(num_filter):
        enc_outputs_list.append(enc_outputs[:, labels==i].detach())
        cur_state = (state[0][0, labels==i].unsqueeze(0).detach(), state[1][0, labels==i].unsqueeze(0).detach())
        latent_list.append(cur_state)
        y_list.append(y_tensor[labels==i])
        out_len_list.append(out_lens_tensor[labels==i])
    return enc_outputs_list, latent_list, y_list, out_len_list

def train_decoders(train_data: List[Example], input_indexer, output_indexer, autoencoder, num_filters, args):
    HIDDEN_SIZE = args.hidden_size
    EMBEDDING_DIM = args.embedding_dim
    LR = args.lr
    BATCH_SIZE = args.batch_size
    EPOCH = args.epochs
    DROPOUT = args.dropout
    MAX_LEN = args.decoder_len_limit

    filters = []
    optimizers = []
    for i in range(num_filters):
        filters.append(copy.deepcopy(autoencoder.decoder))
        optimizers.append(optim.Adam(filters[i].parameters(), lr=LR))

    cluster_classifier = train_cluster_classifier(train_data, autoencoder, input_indexer, HIDDEN_SIZE, num_filters)

    for i in range(EPOCH):
        random.shuffle(train_data)
        total_loss = 0
        for j in range(0, len(train_data), BATCH_SIZE):
            batch_exs = train_data[j: j+BATCH_SIZE]
            x_inp_len, y_inp_len = [],[]
            for ex in batch_exs:
                x_inp_len.append(len(ex.x_tok))
                y_inp_len.append(len(ex.y_tok)+1) # include EOS
            x_tensor = make_padded_input_tensor(batch_exs, input_indexer, max(x_inp_len), reverse_input=False)
            y_tensor = make_padded_output_tensor(batch_exs, output_indexer, max(y_inp_len))
            x_inp_len, y_inp_len = torch.tensor(x_inp_len), torch.tensor(y_inp_len)

            enc_outputs_list, latent_list, y_list, out_len_list = determine_filter(autoencoder, cluster_classifier, num_filters,
                                                                                   torch.tensor(x_tensor), x_inp_len, 
                                                                                   torch.tensor(y_tensor), y_inp_len)

            for k in range(num_filters):
                if len(y_list[k]) <= 0:
                    continue
                filters[k].zero_grad()
                loss = filters[k].train(enc_outputs_list[k], latent_list[k], y_list[k], out_len_list[k])
                loss.backward()
                optimizers[k].step()
                total_loss += loss * len(y_list[k]) / BATCH_SIZE
        print('Epoch ', i+1, total_loss)
    
    return LEMS(autoencoder, filters, cluster_classifier, out_max_length=MAX_LEN)


