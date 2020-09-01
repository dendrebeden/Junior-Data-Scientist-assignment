import numpy as np
import pandas as pd
import sklearn
import pickle as pkl
import os
from build_features import build_features
from preprocess import preprocess 

# Preprocess and prepare data.
preprocess("../data/train.csv", "../data/train_.csv")
build_features("../data/train_.csv", "../data/train_.csv")

# Import training data.
df = pd.read_csv("../data/train_.csv")
assert(all([item in df.columns for item in ["Survived", "Pclass",  "Age", "SibSp", "Parch", "Fare"]]))

# Create a classifier and select scoring methods.
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
clf = RandomForestClassifier(n_estimators=10)

# Fit full model and predict on both train and test.
y = df["Survived"]
X = df.drop("Survived", axis = 1)
clf.fit(X, y)
pred = clf.predict(X)
metric_result = accuracy_score(y, pred)

# Create 'data/model.pkl' file if it does not exist.
dirname = os.path.dirname("data/model.pkl")
if not os.path.exists(dirname):
    os.makedirs(dirname)
# Write in model into 'data/model.pkl'.
model_pickle = open("data/model.pkl", 'wb')
pkl.dump(clf, model_pickle)
model_pickle.close()

print("train_accuracy for the model is " + str(metric_result))