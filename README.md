MUSAE ![GitHub stars](https://img.shields.io/github/stars/benedekrozemberczki/MUSAE.svg?style=plastic) ![GitHub forks](https://img.shields.io/github/forks/benedekrozemberczki/MUSAE.svg?color=blue&style=plastic) ![License](https://img.shields.io/github/license/benedekrozemberczki/MUSAE.svg?color=blue&style=plastic)
============================================
The reference implementation of "Multi-scale Attributed Embedding of Networks."
<p align="center">
  <img width="800" src="musae.jpg">
</p>

### Abstract

<p align="justify">
We present algorithms that embed nodes of a network into a Euclidean feature space based on attributes in their neighborhoods. The algorithms operate in the Skip-gram style of observing nearby contexts in random walks and optimizing node representations to align with the observed attributes. We describe a mult-scale embedding algorithm that simultaneously incorporate neighborhoods of multiple sizes into the embedding procedure. Theoretical analysis shows that this approach can be seen as the implicit approximate factorization of a matrix incorporating connectivity and attribute information. The attribute neighborhood perspective at multiple scales allows diverse applications, including latent feature identification across different networks with similar attributes. Experimental results show that the algorithm performs accurately on common social networks, is robust and computationally efficient.</p>

The second-order random walks sampling methods were taken from the reference implementation of [Node2Vec](https://github.com/aditya-grover/node2vec).

This repository provides a Gensim implementation of MUSAE and AE as described in the paper:
> Multi-scale Attributed Embedding of Networks.
> [Benedek Rozemberczki](http://homepages.inf.ed.ac.uk/s1668259/), [Carl Allen](https://scholar.google.com/citations?user=wRcURR8AAAAJ&hl=en&oi=sra), [Rik Sarkar](https://homepages.inf.ed.ac.uk/rsarkar/)
> Arxiv, 2019.
> [[Paper]](https://github.com/benedekrozemberczki/MUSAE/blob/master/musae_paper.pdf)


### Table of Contents

1. [Citing](#citing)  
2. [Requirements](#requirements)
3. [Datasets](#datasets)  
4. [Logging](#logging)  
5. [Options](#options) 
6. [Examples](#examples)

### Citing

If you find MUSAE useful in your research, please consider citing the following paper:

>@misc{1802.03997,    
       author = {Benedek Rozemberczki, Carl Allen and Rik Sarkar},    
       title = {Multi-scale Attributed Embedding of Networks},   
       year = {2019}
       }

### Requirements
The codebase is implemented in Python 3.5.2. package versions used for development are just below.
```
networkx          1.11
tqdm              4.28.1
numpy             1.15.4
pandas            0.23.4
texttable         1.5.0
scipy             1.1.0
argparse          1.1.0
gensim            3.6.0
```
### Datasets

### Logging

The models are defined in a way that parameter settings and runtimes are logged. Specifically we log the followings:

```
1. Hyperparameter settings.     We save each hyperparameter used in the experiment.
2. Optimization runtime.        We measure the time needed for optimization - measured by seconds.
3. Sampling runtime.            We measure the time needed for sampling - measured by seconds.
```

### Options

Learning of the embedding is handled by the `src/main.py` script which provides the following command line arguments.

#### Input and output options

```
  --graph-input      STR   Input edge list csv.     Default is `input/edges/chameleon_edges.csv`.
  --features-input   STR   Input features json.     Default is `input/features/chameleon_features.json`.
  --output           STR   Embedding output path.   Default is `output/chameleon_embedding.csv`.
  --log              STR   Log output path.         Default is `logs/chameleon.json`.
```
#### Random walk options

```
  --sampling      STR       Random walker order (first/second).              Default is `first`.
  --P             FLOAT     Return hyperparameter for second-order walk.     Default is 1.0
  --Q             FLOAT     In-out hyperparameter for second-order walk.     Default is 1.0.
  --walk-number   INT       Walks per source node.                           Default is 5.
  --walk-length   INT       Truncated random walk length.                    Default is 80.
```

#### Model options

```
  --model                 STR        Pooled or multi-scla model (AE/MUSAE).       Default is `musae`.
  --base-model            STR        Use of Doc2Vec base model.                   Default is `null`.
  --approximation-order   INT        Matrix powers approximated.                  Default is 3.
  --dimensions            INT        Number of dimensions.                        Default is 32.
  --down-sampling         FLOAT      Length of random walk per source.            Default is 0.001.
  --exponent              FLOAT      Downsampling exponent of frequency.          Default is 0.75.
  --alpha                 FLOAT      Initial learning rate.                       Default is 0.05.
  --min-alpha             FLOAT      Final learning rate.                         Default is 0.025.
  --min-count             INT        Minimal occurence of features.               Default is 1.
  --negative-samples      INT        Number of negative samples per node.         Default is 5.
  --workers               INT        Number of cores used for optimization.       Default is 4.
  --epochs                INT        Gradient descent epochs.                     Default is 5.
```

### Examples
<p align="center">
  <img width="500" src="musae.gif">
</p>


Training a MUSAE model for a 10 epochs.
```
python src/main.py --epochs 10
```
Changing the dimension size.
```
python src/main.py --dimensions 32
```
