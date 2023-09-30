#!/home/user-ml-srv/venv_mlops_ml_srv/bin/python
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import sys
import pickle
import yaml
import os
import json

if __name__ == "__main__":
    if len(sys.argv) != 3:
        sys.stderr.write("Arguments error. Usage:\n")
        sys.stderr.write("\tpython3 get_features.py data-file\n")
        sys.exit(1)
    

    params = yaml.safe_load(open("params.yaml"))["train"]
    p_seed = params["seed"]
    p_max_depth = params["max_depth"]

    df = pd.read_csv(sys.argv[1], header=None)
    print(type(df), df.shape)

    X_train = df.iloc[:,:-1]
    y_train = df.iloc[:,-1]

    with open(os.path.join("models", sys.argv[2]), 'rb') as fd:
        model = pickle.load(fd)

    score = model.score(X_train, y_train)

    prc_file = os.path.join("evaluate", "score.json")
    os.makedirs(os.path.join("evaluate"), exist_ok=True)
    

    with open(prc_file, "w") as fd:
        json.dump({"score": score}, fd)