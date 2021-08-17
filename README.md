# LEMS
Code for reproducing results in **Enhancing Latent Space Clustering in Multi-filter Seq2Seq Model**.

Build upon the encoder-decoder architecture, we design a latent-enhanced multi-filter seq2seq model (LMS2S) that analyzes the latent space representations using a clustering algorithm. The representations are generated from an encoder and a latent space enhancer. A cluster classifier is applied to group the representations into clusters. A soft actor-critic reinforcement learning algorithm is applied to the cluster classifier to enhance the clustering quality by maximizing the Silhouette score. Then, multiple filters are trained by the features only from their corresponding clusters, the heterogeneity of the training data can be resolved accordingly.

![architecture](https://github.com/yunhaoyang234/Multi-Filter-Seq2Seq-Model/blob/main/figures/architecture.png)

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
The token-level accuracy and denotation accuracy for the development set will be printed out. Set '--plot 1' to display the latent space clustering result. The results may vary each time, you can run multiple times and get an averaged result.

#### Bilingual Sentence Pairs Translation Experiment
```bash
$ python main.py \
	 --experiment 'translate'\
         --epochs 10\
         --lr 0.001\
         --hidden_size 250\
    	 --num_filters 2\
    	 --dropout 0.1\
         --plot 1\
    	 --embedding_dim 200
```
The BLEU score for the test set will be printed out. Set '--plot 1' to display the latent space clustering result.
