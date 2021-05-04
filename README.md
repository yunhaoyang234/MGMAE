# MGMAE
Code for reproducing results in **Representation Learning in Sequence to Sequence Tasks: Multi-filter Gaussian Mixture Autoencoder**.

The Multi-filter Gaussian Mixture Autoencoder (abbreviation: MGMAE) utilizes an autoencoder to learn the representations of the inputs. The representations are the outputs from the encoder, lying in the latent space whose dimension is the hidden dimension of the encoder. The representations of training data in the latent space are used to train Gaussian mixtures. The latent space representations are divided into several mixtures of Gaussian distributions. A filter (decoder) is tuned to fit the data in one of the Gaussian distributions specifically. Each Gaussian is corresponding to one filter so that the filter only concentrates on the homogeneous features in this Gaussian. Thus the heterogeneity of the training data can be resolved.

![architecture](https://github.com/yunhaoyang234/MGMAE/blob/main/figures/architecture.png)

## Requirements:
See requirement.txt\
Run
`pip install -r requirement.txt`

## Datasets:
- `Geo-query` - geographical questions and answers.
- `Tab-delimited Bilingual Sentence Pairs (English-French)`- English-French sentence pairs for machine translation. Download from [here](https://www.manythings.org/anki/)

These datasets can also be found in `/data` folder.

## Experiments:
#### Geo-query Question-Answering Experiment
```bash
$ python main.py \
	 --experiment 'geo'\
         --train_path 'data/geo_train.tsv'\
         --dev_path 'data/geo_dev.tsv'\
         --epochs 10\
         --lr 0.001\
         --hidden_size 200\
    	 --num_filters 2\
    	 --dropout 0.2\
    	 --embedding_dim 150
```
The token-level accuracy and denotation accuracy for the development set will be printed out.

#### Bilingual Sentence Pairs Translation Experiment
```bash
$ python main.py \
	 --experiment 'translate'\
         --epochs 10\
         --lr 0.001\
         --hidden_size 250\
    	 --num_filters 2\
    	 --dropout 0.1\
    	 --embedding_dim 200\
    	 --train_size 5000\
    	 --test_size 1000
```
The BLEU score for the test set will be printed out.
