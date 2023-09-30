#!/home/user-ml-srv/venv_mlops_ml_srv/bin/python
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
import sys
import os

if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.stderr.write("Arguments error. Usage:\n")
        sys.stderr.write("\tpython3 get_features.py data-file\n")
        sys.exit(1)
    
    os.makedirs(os.path.join("data", "stage3"), exist_ok=True)

    stage2_file = sys.argv[1]
    stage2_data = pd.read_csv(stage2_file)

    label_encoder = LabelEncoder()
    stage2_data['Feature2'] = label_encoder.fit_transform(stage2_data['Feature2'])
    stage2_data.to_csv('data/stage3/train.csv', index=False)
    
    
    print("Данные со строковым признаком заменены и сохранены в data/stage3/train.csv")
