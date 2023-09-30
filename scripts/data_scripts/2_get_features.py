#!/home/user-ml-srv/venv_mlops_ml_srv/bin/python
import pandas as pd
import sys
import os

if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.stderr.write("Arguments error. Usage:\n")
        sys.stderr.write("\tpython3 get_features.py data-file\n")
        sys.exit(1)
    
    os.makedirs(os.path.join("data", "stage1"), exist_ok=True)

    dataset_file = sys.argv[1]
    dataset = pd.read_csv(dataset_file)

    selected_features = dataset[['Feature1', 'Feature2']]
    target_variable = dataset['Target']

    stage1_data = pd.concat([selected_features, target_variable], axis=1)
    stage1_data.to_csv('data/stage1/train.csv', index=False)
    

    print("Целевая переменная и признаки сохранены в data/stage1/train.csv")
