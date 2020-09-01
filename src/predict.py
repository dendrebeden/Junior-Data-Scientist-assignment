import pandas as pd
import pickle as pkl
import sklearn
import os
from build_features import build_features 
from sklearn.metrics import classification_report
from preprocess import preprocess
from build_features import build_features


# Preprocess and prepare data.
preprocess("../data/val.csv", "../data/val_.csv")
build_features("../data/val_.csv", "../data/val_.csv")
# Import validation data.
df = pd.read_csv("../data/val_.csv")
assert(all([item in df.columns for item in ["Survived", "Pclass", "Age", "SibSp", "Parch", "Fare"]]))

# Create 'data/model.pkl' file if it does not exist.
dirname = os.path.dirname("data/model.pkl")
if not os.path.exists(dirname):
    os.makedirs(dirname)
# Read model from 'data/model.pkl'.
model_pickle = open("data/model.pkl", 'rb')
model = pkl.load(model_pickle)
model_pickle.close()

# Predict results.
pred = model.predict(df.drop("Survived", axis = 1))
# Reassign target (if it was present) and predictions.
metric_result = sklearn.metrics.accuracy_score(df["Survived"], pred)

print("train_accuracy for the model is " + str(metric_result))