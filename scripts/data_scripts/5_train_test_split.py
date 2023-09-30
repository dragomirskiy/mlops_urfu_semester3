#!/home/user-ml-srv/venv_mlops_ml_srv/bin/python
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import sys
import pickle

if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.stderr.write("Arguments error. Usage:\n")
        sys.stderr.write("\tpython3 get_features.py data-file\n")
        sys.exit(1)

    stage3_file = sys.argv[1]
    stage3_data = pd.read_csv(stage3_file)

    X = stage3_data[['Feature1', 'Feature2']]
    y = stage3_data['Target']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = DecisionTreeClassifier(random_state=42)
    model.fit(X_train, y_train)

    with open('../../models/model.pkl', 'wb') as model_file:
        pickle.dump(model, model_file)

    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Точность модели: {accuracy:.2f}")
