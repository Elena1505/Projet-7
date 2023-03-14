import pandas as pd
import numpy as np
import gc
from sklearn.metrics import roc_auc_score, f1_score, accuracy_score, confusion_matrix


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
