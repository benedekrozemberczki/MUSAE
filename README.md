MUSAE
============================================
The reference implementation of "Jump Around! Multi-scale Attributed Node Embedding."
<p align="center">
  <img width="800" src="musae.jpg">
</p>
<p align="justify">
Lorem ipsum dolor sit amet, consectetur adipiscing elit, sed do eiusmod tempor incididunt ut labore et dolore magna aliqua. Morbi tristique senectus et netus et malesuada fames ac turpis. A lacus vestibulum sed arcu non odio euismod lacinia at. Consectetur lorem donec massa sapien faucibus et molestie. Risus nullam eget felis eget. Quisque egestas diam in arcu cursus. Facilisis gravida neque convallis a cras semper. Sed augue lacus viverra vitae congue. Convallis tellus id interdum velit. Non nisi est sit amet. Id porta nibh venenatis cras sed felis eget velit aliquet. Odio eu feugiat pretium nibh. Non tellus orci ac auctor. Nibh ipsum consequat nisl vel pretium lectus quam id leo. Volutpat ac tincidunt vitae semper quis lectus nulla at. Dui ut ornare lectus sit amet est placerat in.</p>

The second-order random walks sampling methods were taken from the reference implementation of [Node2Vec](https://github.com/aditya-grover/node2vec).

This repository provides a Gensim implementation of MUSAE and AE as described in the paper:
> Jump Around! Multi-scale Attributed Node Embedding.
> [Benedek Rozemberczki](http://homepages.inf.ed.ac.uk/s1668259/), [Carl Allen](https://scholar.google.com/citations?user=wRcURR8AAAAJ&hl=en&oi=sra), [Rik Sarkar](https://homepages.inf.ed.ac.uk/rsarkar/)
> Arxiv, 2019.
> [[Paper]](https://benito.hu)


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
       author = {Benedek Rozemberczki and Ryan Davies and Rik Sarkar and Charles Sutton},    
       title = {GEMSEC: Graph Embedding with Self Clustering},   
       year = {2018},    
       eprint = {arXiv:1802.03997}
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
  --model                 STR
  --base-model            STR
  --approximation-order   INT
  --dimensions            INT        Number of dimensions.                              Default is 32.
  --down-sampling         FLOAT        Length of random walk per source.                  Default is 80.
  --exponent              FLOAT
  --alpha                 FLOAT
  --min-alpha             FLOAT
  --min-count             INT
  --negative-samples      INT
  --workers               INT
  --epochs                INT
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
