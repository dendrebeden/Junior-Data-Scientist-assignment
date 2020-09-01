# Import required libraries for performance metrics
from sklearn.metrics import make_scorer
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.model_selection import cross_validate

# Define dictionary with performance metrics
scoring = {'accuracy':make_scorer(accuracy_score), 
           'precision':make_scorer(precision_score),
           'recall':make_scorer(recall_score), 
           'f1_score':make_scorer(f1_score)}

# Import required libraries for machine learning classifiers
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB

# Define the models evaluation function
def models_evaluation(X, y, folds):
    
    '''
    X : data set features
    y : data set target
    folds : number of cross-validation folds
    
    '''
    # Instantiate the machine learning classifiers
    log_model = LogisticRegression(max_iter=1000)
    svc_model = LinearSVC(dual=False)
    dtr_model = DecisionTreeClassifier()
    rfc_model = RandomForestClassifier(n_estimators = 10)
    gnb_model = GaussianNB()

    # Perform cross-validation to each machine learning classifier
    log = cross_validate(log_model, X, y, scoring=scoring, cv = folds)
    svc = cross_validate(svc_model, X, y, scoring=scoring, cv = folds)
    dtr = cross_validate(dtr_model, X, y, scoring=scoring, cv = folds)
    rfc = cross_validate(rfc_model, X, y, scoring=scoring, cv = folds)
    gnb = cross_validate(gnb_model, X, y, scoring=scoring, cv = folds)

    # Create a data frame with the models perfoamnce metrics scores
    models_scores_table = pd.DataFrame({'Logistic Regression':[log['test_accuracy'].mean(),
                                                               log['test_precision'].mean(),
                                                               log['test_recall'].mean(),
                                                               log['test_f1_score'].mean()],
                                       
                                      'Support Vector Classifier':[svc['test_accuracy'].mean(),
                                                                   svc['test_precision'].mean(),
                                                                   svc['test_recall'].mean(),
                                                                   svc['test_f1_score'].mean()],
                                       
                                      'Decision Tree':[dtr['test_accuracy'].mean(),
                                                       dtr['test_precision'].mean(),
                                                       dtr['test_recall'].mean(),
                                                       dtr['test_f1_score'].mean()],
                                       
                                      'Random Forest':[rfc['test_accuracy'].mean(),
                                                       rfc['test_precision'].mean(),
                                                       rfc['test_recall'].mean(),
                                                       rfc['test_f1_score'].mean()],
                                       
                                      'Gaussian Naive Bayes':[gnb['test_accuracy'].mean(),
                                                              gnb['test_precision'].mean(),
                                                              gnb['test_recall'].mean(),
                                                              gnb['test_f1_score'].mean()]},
                                      
                                      index=['Accuracy', 'Precision', 'Recall', 'F1 Score'])
    
    # Add 'Best Score' column
    models_scores_table['Best Score'] = models_scores_table.idxmax(axis=1)
    
    # Return models performance metrics scores data frame
    return(models_scores_table)
  
# Run models_evaluation function
from build_features import build_features
from preprocess import preprocess
import pandas as pd

# Preprocess and prepare data.
preprocess("../data/train.csv", "../data/train_.csv")
build_features("../data/train_.csv", "../data/train_.csv")
# Import training data.
df = pd.read_csv("../data/train_.csv")

print(models_evaluation(df.drop("Survived", axis = 1), df["Survived"], 10))