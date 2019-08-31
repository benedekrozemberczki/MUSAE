import pandas as pd
import networkx as nx
import random
import numpy as np
import warnings
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, roc_auc_score, f1_score
from sklearn.linear_model import LogisticRegression, ElasticNet
from sklearn.exceptions import ConvergenceWarning, UndefinedMetricWarning
warnings.filterwarnings(action='ignore', category=UndefinedMetricWarning)
warnings.filterwarnings(action='ignore', category=ConvergenceWarning)

def fit_model(features, target, reps):
    scores = []
    for i in range(reps):
        X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.9, random_state = i)
        model = LogisticRegression(C=10, solver = "lbfgs")

        model.fit(X_train, y_train)
        y_pred = model.predict_proba(X_test)
        score = roc_auc_score(y_test, y_pred[:,1], average = "weighted")
        scores.append(score)
    print("$\\underset{\pm " +str(round(np.std(scores)/(reps**0.5),4))  +"}{"+str(round(np.mean(scores),4))+"}$")



datasets = ["git","facebook","ES","DE","chameleon","crocodile"]
models = ["tene"]
feature_derive = ["average","hadamard","L1","L2"]

for derive in feature_derive:
    print("################################")
    print("################################")
    print(derive)
    print("################################")
    print("################################")
    for model in models:
         print(model)
         print("------------------------------------")
         print("------------------------------------")
         for dataset in datasets:
             random.seed(42)
             edges = pd.read_csv("./input/edges/"+dataset+"_edges.csv").values.tolist()
             attenuated_edges = pd.read_csv("./input/attenuated_edges/"+dataset+"_edges.csv").values.tolist()
             embedding = pd.read_csv("./attenuated_output/"+dataset+"_"+model+".csv")
             base_graph = nx.from_edgelist(edges)
             base_graph.remove_edges_from(attenuated_edges)
             nodes = list(set([edge[0] for edge in edges ]+[edge[1] for edge in edges ]))
             new_edges = [[random.choice(nodes),random.choice(nodes), 0] for edge in base_graph.edges()]
             base_edges = [[edge[0],edge[1], 1] for edge in base_graph.edges()]
             aller = pd.DataFrame(base_edges + new_edges, columns = ["id","id2","target"]) 
             aller = aller.set_index("id").join(embedding.set_index("id")).reset_index()
             embedding.columns = ["id2"] + ["y"+str(i) for i in range(128)]
             aller = aller.set_index("id2").join(embedding.set_index("id2")).reset_index()
             target = np.array(aller["target"])
             X_1 = np.array(aller)[:,3:131]
             X_2 = np.array(aller)[:,131:]
             if model == "line":
                 X_1 = X_1[:,0:64]
                 X_2 = X_2[:,0:64]
             if derive == "L1":
                 features = np.abs(X_1 -X_2)
             elif derive == "L2":
                 features = np.square(X_1-X_2)
             elif derive == "average":
                 features = X_1-X_2
                 features = features/2
             else:
                 features = X_1 * X_2
             print(dataset)
             fit_model(features, target, 10)
