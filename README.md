# LEMS
Code for reproducing results in **Training Heterogeneous Features in Sequence to Sequence Tasks: Latent Enhanced Multi-filter Seq2Seq Model**.

Build upon the encoder-decoder architecture, we design a latent-enhanced multi-filter seq2seq model (LEMS) that analyzes the input-output mapping representations by introducing a latent space transformation and clustering. The representations are extracted from the final hidden state of the encoder and lied on the latent space. A latent space transformation is applied to the representations. It transforms the representations to another space that better represents the input-output mappings. Thus the clustering algorithm can easily separate samples based on their features. Then, multiple filters can be trained by the features from their corresponding clusters, the heterogeneity of the training data can be resolved accordingly.

![architecture](https://github.com/yunhaoyang234/Multi-Filter-Seq2Seq-Model/blob/main/figures/LEMS_struct.png)

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
