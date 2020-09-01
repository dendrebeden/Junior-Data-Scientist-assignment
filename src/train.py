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
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
logistic_model = LogisticRegression(max_iter=1000)

# Fit full model and predict on both train and test.
y = df["Survived"]
X = df.drop("Survived", axis = 1)
logistic_model.fit(X, y)
pred = logistic_model.predict(X)
metric_result = accuracy_score(y, pred)

# Create 'data/model.pkl' file if it does not exist.
dirname = os.path.dirname("data/model.pkl")
if not os.path.exists(dirname):
    os.makedirs(dirname)
# Write in model into 'data/model.pkl'.
model_pickle = open("data/model.pkl", 'wb')
pkl.dump(logistic_model, model_pickle)
model_pickle.close()

print("train_accuracy for the model is " + str(metric_result))