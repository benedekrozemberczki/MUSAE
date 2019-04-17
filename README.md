MUSAE
============================================
The reference implementation of "Jump Around! Multi-scale Attributed Node Embedding."
<p align="center">
  <img width="800" src="musae.jpg">
</p>
<p align="justify">
Lorem ipsum dolor sit amet, consectetur adipiscing elit, sed do eiusmod tempor incididunt ut labore et dolore magna aliqua. Morbi tristique senectus et netus et malesuada fames ac turpis. A lacus vestibulum sed arcu non odio euismod lacinia at. Consectetur lorem donec massa sapien faucibus et molestie. Risus nullam eget felis eget. Quisque egestas diam in arcu cursus. Facilisis gravida neque convallis a cras semper. Sed augue lacus viverra vitae congue. Convallis tellus id interdum velit. Non nisi est sit amet. Id porta nibh venenatis cras sed felis eget velit aliquet. Odio eu feugiat pretium nibh. Non tellus orci ac auctor. Nibh ipsum consequat nisl vel pretium lectus quam id leo. Volutpat ac tincidunt vitae semper quis lectus nulla at. Dui ut ornare lectus sit amet est placerat in.</p>

The second-order random walks sampling methods were taken from the reference implementation of [Node2vec](https://github.com/aditya-grover/node2vec).

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
```
### Datasets

### Logging

The models are defined in a way that parameter settings and cluster quality is logged in every single epoch. Specifically we log the followings:

```
1. Hyperparameter settings.     We save each hyperparameter used in the experiment.
2. Cost per epoch.              Embedding, clustering and regularization cost are stored depending on the model type.
3. Cluster quality.             Measured by modularity. We calculate it both for the classical and neural clusterings per epoch.
4. Runtime.                     We measure the time needed for optimization and data generation per epoch -- measured by seconds.
```

### Options

Learning of the embedding is handled by the `src/embedding_clustering.py` script which provides the following command line arguments.

#### Input and output options

```
  --input                STR      Input graph path.                              Default is `data/politician_edges.csv`.
  --embedding-output     STR      Embeddings path.                               Default is `output/embeddings/politician_embedding.csv`.
  --cluster-mean-output  STR      Cluster centers path.                          Default is `output/cluster_means/politician_means.csv`.
  --log-output           STR      Log path.                                      Default is `output/logs/politician.log`.
  --assignment-output    STR      Node-cluster assignment dictionary path.       Default is `output/assignments/politician.json`.
  --dump-matrices        BOOL     Whether the trained model should be saved.     Default is `True`.
  --model                STR      The model type.                                Default is `GEMSECWithRegularization`.
```


#### Random walk options

```
  --walker   STR         Random walker order (first/second).              Default is `first`.
  --P        FLOAT       Return hyperparameter for second-order walk.     Default is 1.0
  --Q        FLOAT       In-out hyperparameter for second-order walk.     Default is 1.0.
```

#### Skipgram options

```
  --dimensions               INT        Number of dimensions.                              Default is 16.
  --random-walk-length       INT        Length of random walk per source.                  Default is 80.
  --num-of-walks             INT        Number of random walks per source.                 Default is 5.
  --window-size              INT        Window size for proximity statistic extraction.    Default is 5.
  --distortion               FLOAT      Downsampling distortion.                           Default is 0.75.
  --negative-sample-number   INT        Number of negative samples to draw.                Default is 10.
```

#### Model options

```
  --initial-learning-rate   FLOAT    Initial learning rate.                                        Default is 0.001.
  --minimal-learning-rate   FLOAT    Final learning rate.                                          Default is 0.0001.
  --annealing-factor        FLOAT    Annealing factor for learning rate.                           Default is 1.0.
  --initial-gamma           FLOAT    Initial clustering weight coefficient.                        Default is 0.1.
  --lambd                   FLOAT    Smoothness regularization penalty.                            Default is 0.0625.
  --cluster-number          INT      Number of clusters.                                           Default is 20.
  --overlap-weighting       STR      Weight construction technique for regularization.             Default is `normalized_overlap`.
  --regularization-noise    FLOAT    Uniform noise max and min on the feature vector distance.     Default is 10**-8.
```

### Examples
<p align="center">
  <img width="500" src="musae.gif">
</p>


Training a CapsGNNN model for a 100 epochs.
```
python src/main.py --epochs 100
```
Changing the batch size.
```
python src/main.py --batch-size 128
```
