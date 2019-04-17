MUSAE
============================================
The reference implementation of "Jump Around! Multi-scale Attributed Node Embedding."
<p align="center">
  <img width="800" src="musae.jpg">
</p>
<p align="justify">
Lorem ipsum dolor sit amet, consectetur adipiscing elit, sed do eiusmod tempor incididunt ut labore et dolore magna aliqua. Morbi tristique senectus et netus et malesuada fames ac turpis. A lacus vestibulum sed arcu non odio euismod lacinia at. Consectetur lorem donec massa sapien faucibus et molestie. Risus nullam eget felis eget. Quisque egestas diam in arcu cursus. Facilisis gravida neque convallis a cras semper. Sed augue lacus viverra vitae congue. Convallis tellus id interdum velit. Non nisi est sit amet. Id porta nibh venenatis cras sed felis eget velit aliquet. Odio eu feugiat pretium nibh. Non tellus orci ac auctor. Nibh ipsum consequat nisl vel pretium lectus quam id leo. Volutpat ac tincidunt vitae semper quis lectus nulla at. Dui ut ornare lectus sit amet est placerat in.</p>

This repository provides a Gensim implementation of MUSAE and AE as described in the paper:
> Jump Around! Multi-scale Attributed Node Embedding.
> Benedek Rozemberczki, Carl Allen and Rik Sarkar.
> Arxiv, 2019.
> [[Paper]](https://benito.hu)

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
The code takes graphs for training from an input folder where each graph is stored as a JSON. Graphs used for testing are also stored as JSON files. Every node id and node label has to be indexed from 0. Keys of dictionaries are stored strings in order to make JSON serialization possible.

Every JSON file has the following key-value structure:

```javascript
{"edges": [[0, 1],[1, 2],[2, 3],[3, 4]],
 "labels": {"0": "A", "1": "B", "2": "C", "3": "A", "4": "B"},
 "target": 1}
```
The **edges** key has an edge list value which descibes the connectivity structure. The **labels** key has labels for each node which are stored as a dictionary -- within this nested dictionary labels are values, node identifiers are keys. The **target** key has an integer value which is the class membership.

### Outputs

The predictions are saved in the `output/` directory. Each embedding has a header and a column with the graph identifiers. Finally, the predictions are sorted by the identifier column.

### Options
Training a CapsGNN model is handled by the `src/main.py` script which provides the following command line arguments.

#### Input and output options
```
  --training-graphs   STR    Training graphs folder.      Default is `dataset/train/`.
  --testing-graphs    STR    Testing graphs folder.       Default is `dataset/test/`.
  --prediction-path   STR    Output predictions file.     Default is `output/watts_predictions.csv`.
```
#### Model options
```
  --epochs                      INT     Number of epochs.                  Default is 10.
  --batch-size                  INT     Number fo graphs per batch.        Default is 8.
  --gcn-filters                 INT     Number of filters in GCNs.         Default is 2.
  --gcn-layers                  INT     Number of GCNs chained together.   Default is 5.
  --inner-attention-dimension   INT     Number of neurons in attention.    Default is 20.  
  --capsule-dimensions          INT     Number of capsule neurons.         Default is 8.
  --number-of-capsules          INT     Number of capsules in layer.       Default is 8.
  --weight-decay                FLOAT   Weight decay of Adam.              Defatuls is 10^-6.
  --lambd                       FLOAT   Regularization parameter.          Default is 1.0.
  --learning-rate               FLOAT   Adam learning rate.                Default is 0.01.
```
### Examples
The following commands learn a model and save the predictions. Training a model on the default dataset:
```
python src/main.py
```
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
