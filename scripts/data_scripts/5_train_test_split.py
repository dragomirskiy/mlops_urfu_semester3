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

if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.stderr.write("Arguments error. Usage:\n")
        sys.stderr.write("\tpython3 get_features.py data-file\n")
        sys.exit(1)
    
    os.makedirs(os.path.join("data", "stage4"), exist_ok=True)

    stage3_file = sys.argv[1]
    stage3_data = pd.read_csv(stage3_file)

    X = stage3_data[['Feature1', 'Feature2']]
    y = stage3_data['Target']

    params = yaml.safe_load(open("params.yaml"))["split"]
    p_split_ratio = params["split_ratio"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=p_split_ratio, random_state=42)
    

    pd.concat([X_train, y_train], axis=1).to_csv(os.path.join("data", "stage4", "train.csv"), header=None, index=None)
    pd.concat([X_test, y_test], axis=1).to_csv(os.path.join("data", "stage4", "test.csv"), header=None, index=None)
