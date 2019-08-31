import pandas as pd
import numpy as np
import json
import math
from tqdm import tqdm
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, roc_auc_score, f1_score
from sklearn.linear_model import LogisticRegression, ElasticNet

def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn

def mapper(x):
    if x =="politician":
        y = 0
    elif x =="company":
        y = 1
    elif x =="government":
        y = 2
    else:
        y = 3
    return y

def tester(dataset, model, reps):
    features = pd.read_csv("./output/"+dataset+"_"+model+".csv")
    if dataset == "PTBR" or dataset == "de" or dataset == "ru" or dataset == "engb"  or dataset == "es" or dataset == "zhtw":
        target = pd.read_csv("./input/targets/"+dataset+"_target.csv").sort_values(["new_id"])
    else:
        target = pd.read_csv("./input/targets/"+dataset+"_target.csv").sort_values(["id"])

    if dataset == "PTBR" or dataset == "de" or dataset == "ru" or dataset == "engb"or dataset == "es"or dataset == "zhtw":
        target = target["mature"]
        target = target.values.tolist()
        target = [1 if t == True else 0 for t in target]
    elif dataset == "git":
        target = target["ml_target"]

    else:
        target = target["page_type"]
        target = target.values.tolist()
        target = [mapper(t) for t in target]
    y = np.array(target)

    X = np.array(features)[:,1:]
    X = X - X.mean(axis=0)
    w_scores, micro_scores, macro_scores = [], [], []
    if model == "tadw" or model == "line":
        X = X[:,64:]
    for i in range(reps):
        X_train, X_test, y_train, y_test = train_test_split( X, y, test_size=0.2, random_state = i)
        model = LogisticRegression(C=10, solver = "lbfgs")

        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        score = f1_score(y_test, y_pred, average = "weighted")
        w_scores.append(score)
        score = f1_score(y_test, y_pred, average = "micro")
        micro_scores.append(score)
        score = f1_score(y_test, y_pred, average = "macro")
        macro_scores.append(score)
    print("weighted")
    print("$\\underset{\pm " +str(round(np.std(w_scores)/(reps**0.5),3))  +"}{"+str(round(np.mean(w_scores),3))+"}$")
    print("micro")
    print("$\\underset{\pm " +str(round(np.std(micro_scores)/(reps**0.5),3))  +"}{"+str(round(np.mean(micro_scores),3))+"}$")
    print("macro")
    print("$\\underset{\pm " +str(round(np.std(macro_scores)/(reps**0.5),3))  +"}{"+str(round(np.mean(macro_scores),3))+"}$")


reps = 10
datasets = ["git"]
models = ["tene"]
for model in models:
    print("-----------------------------------------------------------")
    print(model.upper())
    for dataset in datasets:
        print(dataset.upper())
        try:
            tester(dataset, model, reps)
        except:
            pass
