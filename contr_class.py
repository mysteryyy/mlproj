import pandas as pd
import numpy as np
from scipy import stats
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from aif360.algorithms.preprocessing.optim_preproc_helpers.data_preproc_functions import (
    load_preproc_data_adult,
)
from aif360.datasets import BankDataset
import numpy as np
from aif360.datasets import BinaryLabelDataset
from aif360.sklearn.datasets import fetch_adult, fetch_german, fetch_bank
from sklego.linear_model import DemographicParityClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, make_scorer
from sklego.metrics import correlation_score
from cvxpy.error import SolverError


def bank_preproc():
    # function for cleaning up
    bank = BankDataset(privileged_classes=[])
    np.random.seed(2021)
    train, test = bank.split([0.7], shuffle=True)
    return train, test


train, test = bank_preproc()
train_df, test_df = train.convert_to_dataframe()[0], test.convert_to_dataframe()[0]
x_train, y_train = train_df.drop(columns=["y"]), train_df["y"]

x_test, y_test = test_df.drop(columns=["y"]), test_df["y"]
feature_cols = x_train.columns

# Prepare feature vestors with dropped sensitive attribute
x_drop_train = x_train.drop(columns=["age"])

x_drop_test = x_test.drop(columns=["age"])

lr_bias = LogisticRegression(solver="liblinear")
lr_bias_notdrop = LogisticRegression(solver="liblinear")

lr_bias.fit(x_drop_train, y_train)
lr_bias_notdrop.fit(x_train, y_train)
pred_train_bias = lr_bias.predict(x_drop_train)
pred_train_bias_notdrop = lr_bias_notdrop.predict(x_train)


pred_test_bias = lr_bias.predict(x_drop_test)
pred_test_bias_notdrop = lr_bias_notdrop.predict(x_test)


age_train = np.array(x_train["age"])

age_test = np.array(x_test["age"])


def val_score(cov_thresh):

    scores = []
    corrs = []
    skf = StratifiedKFold(n_splits=5, random_state=2025, shuffle=True)
    for train_index, val_index in skf.split(x_train, y_train):
        x_train_cv = x_train.iloc[train_index]
        y_train_cv = y_train.iloc[train_index]
        x_test_cv = x_train.iloc[val_index]
        y_test_cv = y_train.iloc[val_index]
        clf = DemographicParityClassifier(
            sensitive_cols="age", covariance_threshold=cov_thresh
        )
        # skip iteration in case of convergence error
        try:
            clf.fit(x_train_cv, y_train_cv)
        except SolverError:
            continue

        age_test_cv = np.array(x_test_cv["age"])
        score = clf.score(x_test_cv, y_test_cv)
        labels = clf.predict(x_test_cv)
        corr = np.corrcoef(age_test_cv, labels)[0][1]

        scores.append(score)
        corrs.append(corr)
    return np.mean(scores), np.mean(corrs)


cv_res = pd.DataFrame(
    columns=["Accuracy", "correlation"], index=np.linspace(0.01, 1, 5)
)
for i in cv_res.index:
    acc, corr = val_score(i)
    cv_res.at[i, "Accuracy"] = acc
    cv_res.at[i, "correlation"] = corr


# Compute score
cv_res["mod_corr"] = 1 - abs(cv_res["correlation"])
cv_res["score"] = stats.hmean(cv_res[["mod_corr", "Accuracy"]], axis=1)
best_thresh = float(cv_res.score.astype("float").idxmax())
print("Best covariance threshold: ", best_thresh)
print(cv_res)

final_clf = DemographicParityClassifier(
    sensitive_cols="age", covariance_threshold=best_thresh
)
final_clf.fit(x_train, y_train)
preds_clf = final_clf.predict(x_test)
res1 = pd.DataFrame()
res1["Not Dropped"] = [
    np.corrcoef(age_test, pred_test_bias_notdrop)[0][1],
    accuracy_score(y_test, pred_test_bias_notdrop),
]
res1["Dropped"] = [
    np.corrcoef(age_test, pred_test_bias)[0][1],
    accuracy_score(y_test, pred_test_bias),
]
res1["Constrained Classifier"] = [
    np.corrcoef(age_test, preds_clf)[0][1],
    accuracy_score(y_test, preds_clf),
]
res1.index = ["Fairness Score", "Accuracy"]
print("Test set results")
print(res1)
