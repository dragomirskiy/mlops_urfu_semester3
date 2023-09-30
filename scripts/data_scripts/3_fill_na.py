#!/home/user-ml-srv/venv_mlops_ml_srv/bin/python
import pandas as pd
from sklearn.impute import SimpleImputer
import sys
import os

if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.stderr.write("Arguments error. Usage:\n")
        sys.stderr.write("\tpython3 get_features.py data-file\n")
        sys.exit(1)
    
    os.makedirs(os.path.join("data", "stage2"), exist_ok=True)

    stage1_file = sys.argv[1]
    stage1_data = pd.read_csv(stage1_file)

    imputer = SimpleImputer(strategy='mean')
    stage1_data['Feature1'] = imputer.fit_transform(stage1_data[['Feature1']])
    stage1_data.to_csv('data/stage2/train.csv', index=False)
    

    print("Пропущенные значения заменены на средние и сохранены в data/stage2/train.csv")
