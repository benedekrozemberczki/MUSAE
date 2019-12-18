"""MUSAE model class."""

import json
import time
import numpy as np
import pandas as pd
from tqdm import tqdm
from gensim.models.doc2vec import Doc2Vec
from walkers import FirstOrderRandomWalker, SecondOrderRandomWalker
from utils import load_graph, load_features, create_documents

class MUSAE:
    """
    Multi-Scale Attributed Embedding class.
    For details see the paper:
    Multi-scale Attributed Node Embedding, Benedek Rozemberczki, Carl Allen, Rik Sarkar
    https://arxiv.org/abs/1909.13021
    """
    def __init__(self, args):
        """
        MUSAE and AE machine constructor.
        :param args: Arguments object with the model hyperparameters.
        """
        self.args = args
        self.log = dict()
        self.graph = load_graph(args.graph_input)
        self.features = load_features(args.features_input)

    def do_sampling(self):
        """
        Running a first or second-order random walk sampler.
        Measuring the sampling runtime.
        """
        self.log["walk_start_time"] = time.time()
        if self.args.sampling == "second":
            self.sampler = SecondOrderRandomWalker(self.graph,
                                                   self.args.P,
                                                   self.args.Q,
                                                   self.args.walk_number,
                                                   self.args.walk_length)
        else:
            self.sampler = FirstOrderRandomWalker(self.graph,
                                                  self.args.walk_number,
                                                  self.args.walk_length)
        self.walks = self.sampler.walks
        del self.sampler
        self.log["walk_end_time"] = time.time()

    def _create_single_embedding(self, features):
        """
        Learning an embedding from a feature hash table.
        :param features: A hash table with node keys and feature list values.
        :return embedding: Numpy array of embedding.
        """
        print("\nLearning the embedding.")
        document_collections = create_documents(features)

        model = Doc2Vec(document_collections,
                        vector_size=self.args.dimensions,
                        window=0,
                        min_count=self.args.min_count,
                        alpha=self.args.alpha,
                        dm=0,
                        negative=self.args.negative_samples,
                        ns_exponent=self.args.exponent,
                        min_alpha=self.args.min_alpha,
                        sample=self.args.down_sampling,
                        workers=self.args.workers,
                        epochs=self.args.epochs)

        emb = np.array([model.docvecs[str(n)] for n in range(self.graph.number_of_nodes())])
        return emb


    def _create_documents(self, features):
        print("Creating documents.")
        features_out = {}
        for node, feature_set in tqdm(features.items(), total=len(features)):
            features_out[str(node)] = [feat for feat_elems in feature_set for feat in feat_elems]
        return features_out

    def _setup_musae_features(self, approximation):
        """
        Creating MUSAE feature set.
        :param approximation: Approximation-order.
        :return features: Feature hash-table.
        """
        features = {str(node): [] for node in self.graph.nodes()}
        print("Processing attributed walks.")
        for walk in tqdm(self.walks):
            for i in range(len(walk)-approximation):
                source = walk[i]
                target = walk[i+approximation]
                features[str(source)].append(self.features[str(target)])
                features[str(target)].append(self.features[str(source)])

        return self._create_documents(features)

    def _setup_ae_features(self):
        """
        Create AE feature set.
        :return features: Feature set hash table.
        """
        features = {str(node):[] for node in self.graph.nodes()}
        print("Processing attributed walks.")
        for walk in tqdm(self.walks):
            for i in range(len(walk)-self.args.approximation_order):
                for j in range(self.args.approximation_order):
                    source = walk[i]
                    target = walk[i+j+1]
                    features[str(source)].append(self.features[str(target)])
                    features[str(target)].append(self.features[str(source)])

        return self._create_documents(features)

    def _print_approximation_order(self, approximation):
        """
        Nice printing ofapproximation order for MUSAE.
        :param approximation: Approximation order.
        """
        print("\nApproximation order: " + str(approximation + 1) + ".\n")

    def _learn_musae_embedding(self):
        """
        Learning MUSAE embeddings up to the approximation order.
        """
        for approximation in range(self.args.approximation_order):
            self._print_approximation_order(approximation)
            features = self._setup_musae_features(approximation+1)
            embedding = self._create_single_embedding(features)
            self.embeddings.append(embedding)

    def _learn_ae_embedding(self):
        """
        Learning an AE embedding.
        """
        features = self._setup_ae_features()
        embedding = self._create_single_embedding(features)
        self.embeddings.append(embedding)

    def learn_embedding(self):
        """
        Learning the embeddings and measuring optimization runtime.
        """
        self.log["optim_start_time"] = time.time()
        self.embeddings = []
        if self.args.base_model == "null":
            embedding = self._create_single_embedding(self.features)
            self.embeddings.append(embedding)
        if self.args.model == "musae":
            self._learn_musae_embedding()
        else:
            self._learn_ae_embedding()
        self.embeddings = np.concatenate(self.embeddings, axis=1)
        self.log["optim_end_time"] = time.time()

    def save_embedding(self):
        """
        Method to save the embedding.
        """
        print("\nSaving embedding.\n")
        columns = ["id"] + ["x_"+str(x) for x in range(self.embeddings.shape[1])]
        ids = np.array(range(self.embeddings.shape[0])).reshape(-1, 1)
        self.embeddings = np.concatenate([ids, self.embeddings], axis=1)
        self.embeddings = pd.DataFrame(self.embeddings, columns=columns)
        self.embeddings.to_csv(self.args.output, index=None)

    def save_logs(self):
        """
        Method to save the logs.
        """
        print("Saving the logs.")
        with open(self.args.log, "w") as f:
            json.dump(self.log, f)
