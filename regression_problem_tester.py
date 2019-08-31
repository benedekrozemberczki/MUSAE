import pandas as pd
import numpy as np
import json
import math
from tqdm import tqdm
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, roc_auc_score
from sklearn.linear_model import LogisticRegression, ElasticNet
def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn

def tester(dataset, model, reps):
    features = pd.read_csv("./output/"+dataset+"_"+model+".csv")
    target = pd.read_csv("./input/target/"+dataset+"_target.csv").sort_values(["id"])
    target = target["target"]
    target = target.values.tolist()
    target = list(map(lambda x: math.log(x),target))
    #print(target)
    y = np.array(target)
    y = y-np.mean(y)
    X = np.array(features)[:,1:]
    means = np.mean(X,axis=0)
    stds = np.std(X,axis=0)
    if model != "bane":
        X = (X - means)/stds

    scores = []
    for i in range(reps):
        X_train, X_test, y_train, y_test = train_test_split( X, y, test_size=0.2)
        model = ElasticNet(alpha=0.01)

        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        score = r2_score(y_test, y_pred)
        scores.append(score)
    print("$\\underset{\pm " +str(round(np.std(scores)/(reps**0.5),3))  +"}{"+str(round(np.mean(scores),3))+"}$")

reps = 100
datasets = ["chameleon","crocodile","squirrel"]
models = ["sine"]
for model in models:
    print("-----------------------------------------------------------")
    print(model.upper())
    for dataset in datasets:
        print(dataset.upper())
        try:
            tester(dataset, model, reps)
        except:
            pass
