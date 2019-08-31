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
    if dataset == "ptbr" or dataset == "de" or dataset == "ru" or dataset == "engb"  or dataset == "es" or dataset == "zhtw":
        target = pd.read_csv("./input/targets/"+dataset+"_target.csv").sort_values(["new_id"])
    else:
        target = pd.read_csv("./input/targets/"+dataset+"_target.csv").sort_values(["id"])

    if dataset == "ptbr" or dataset == "de" or dataset == "ru" or dataset == "engb"or dataset == "es"or dataset == "zhtw":
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
    if model == "tadw":
        X = X[:,64:]
    X = X - X.mean(axis=0)
    for ratio in range(1,10):
        scores = []
        for i in range(reps):
            X_train, X_test, y_train, y_test = train_test_split( X, y, train_size=ratio/10, random_state = i)
            model = LogisticRegression(C=10)

            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            score = f1_score(y_test, y_pred, average = "weighted")
            scores.append(score)
        print("(" + str(float(ratio)/10) + "," + str(round(np.mean(scores),3)) + ")")

reps = 10
datasets = ["facebook"]
models = ["tene","musae"]
for model in models:
    print("-----------------------------------------------------------")
    print(model.upper())
    for dataset in datasets:
        print(dataset.upper())
        try:
            tester(dataset, model, reps)
        except:
            pass
