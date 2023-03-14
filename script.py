import gc
import logging
import warnings

import xgboost as xgb
import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score, f1_score, accuracy_score, confusion_matrix, make_scorer
from lightgbm import LGBMClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.dummy import DummyClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline

logging.basicConfig(level=logging.WARN)
logger = logging.getLogger(__name__)


# Preprocess application_train.csv
def application_train(num_rows=None, nan_as_category=False):
    # Read data
    df = pd.read_csv('./application_train.csv', nrows=num_rows)
    print("Train samples: {}".format(len(df)))
    # Remove applications with XNA CODE_GENDER
    df = df[df['CODE_GENDER'] != 'XNA']
    # NaN values for DAYS_EMPLOYED: 365 243 -> nan
    df['DAYS_EMPLOYED'].replace(365243, np.nan, inplace=True)
    # Some simple new features (percentages)
    df['DAYS_EMPLOYED_PERC'] = df['DAYS_EMPLOYED'] / df['DAYS_BIRTH']
    df['INCOME_CREDIT_PERC'] = df['AMT_INCOME_TOTAL'] / df['AMT_CREDIT']
    df['INCOME_PER_PERSON'] = df['AMT_INCOME_TOTAL'] / df['CNT_FAM_MEMBERS']
    df['ANNUITY_INCOME_PERC'] = df['AMT_ANNUITY'] / df['AMT_INCOME_TOTAL']
    df['PAYMENT_RATE'] = df['AMT_ANNUITY'] / df['AMT_CREDIT']
    gc.collect()
    return df


# Buisness score
# Assigning a weight to FN and FP
def cost(actual, pred, TN_val=0, FN_val=10, TP_val=0, FP_val=1):
    matrix = confusion_matrix(actual, pred)
    TN = matrix[0, 0]
    FN = matrix[1, 0]
    FP = matrix[0, 1]
    TP = matrix[1, 1]
    total_gain = TP * TP_val + TN * TN_val + FP * FP_val + FN * FN_val
    return total_gain


# Metrics
def eval_metrics(actual, pred):
    accuracy = accuracy_score(actual, pred)
    AUC = roc_auc_score(actual, pred)
    f1 = f1_score(actual, pred)
    bank_cost = cost(actual, pred)
    return f1, AUC, accuracy, bank_cost


if __name__ == "__main__":
    warnings.filterwarnings("ignore")

    # Split the data into training and test sets. (0.75, 0.25) split.
    df = application_train(10000)
    train, test = train_test_split(df)

    # The predicted column is "TARGET" (0 or 1)
    train_x = train.drop(["TARGET"], axis=1)
    test_x = test.drop(["TARGET"], axis=1)
    train_y = train[["TARGET"]]
    test_y = test[["TARGET"]]

    # Pipeline that aggregates preprocessing steps (encoder + scaler + SMOTE + model)
    steps = [("ohe", OneHotEncoder(handle_unknown="ignore")),
             ("std", StandardScaler(with_mean=False)),
             ("sampling", SMOTE(random_state=42, sampling_strategy=0.2)),
             ("model", LGBMClassifier())]

    pipe = Pipeline(steps)
    pipe.fit(train_x, train_y)

    # GridSearchCV that allows to choose the best model for the problem
    param_grid = {"model": [DummyClassifier(),
                            LogisticRegression(),
                            LGBMClassifier(),
                            KNeighborsClassifier(),
                            xgb.XGBClassifier(),
                            DecisionTreeClassifier(),
                            RandomForestClassifier(),
                            GaussianNB(),
                            SVC()]}
    score = make_scorer(cost)
    grid = GridSearchCV(pipe, param_grid, cv=5, return_train_score=True, scoring=score)
    grid.fit(test_x, test_y)
    print("Best score: ", grid.best_score_, "using ", grid.best_params_)
