import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from aif360.algorithms.inprocessing.gerryfair.auditor import *
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score
from aif360.datasets import BankDataset, BinaryLabelDataset
from aif360.algorithms.inprocessing import GerryFairClassifier
from aif360.metrics import BinaryLabelDatasetMetric
from aif360.sklearn.metrics import (
    disparate_impact_ratio,
    average_odds_error,
    generalized_fpr,
)
from aif360.algorithms.inprocessing.gerryfair.clean import array_to_tuple
from aif360.algorithms.inprocessing.gerryfair.auditor import Auditor
from sklearn import tree
from scipy import stats


def bank_preproc():
    # function for cleaning up
    bank = BankDataset(privileged_classes=[])
    np.random.seed(2021)
    train, test = bank.split([0.7], shuffle=True)
    return train, test


def pareto_curve(
    dataset,
    figname,
    gamma_list=[0.0005, 0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1, 0.15, 0.2, 0.25],
):

    results_dict = {}
    max_iters = 150
    fair_clf = GerryFairClassifier(C=10, printflag=True, gamma=1, max_iters=max_iters)
    fair_clf.printflag = False
    fair_clf.max_iters = max_iters
    errors, fp_violations, fn_violations = fair_clf.pareto(dataset, gamma_list)
    results_dict = {
        "gamma": gamma_list,
        "errors": errors,
        "fp_violations": fp_violations,
    }
    plt.plot(fp_violations, errors)
    plt.xlabel("Unfairness")
    plt.ylabel("Error")
    plt.legend()
    plt.title(f"Error vs. Unfairness\n({figname} Dataset)")
    plt.savefig(f"gerryfair_pareto_{figname}.png")
    plt.close()
    plt.show()
    return results_dict


def get_groups(data, preds):
    # To return group predictions given class probabilities
    audit = Auditor(data, "FP")

    gpred, gdisp = audit.audit(preds)

    return gpred, gdisp


def group_preds_demo(data, figname, gamma):
    # For demonstrating the groups found by the auditor
    fair_clf = GerryFairClassifier(C=10, printflag=True, gamma=gamma, max_iters=2)
    fair_clf.fit(data)
    dt_yhat = fair_clf.predict(data, threshold=False)  # for soft predictions
    dt_yhat1 = fair_clf.predict(data)  # for getting hard predictions

    # get group labels
    gpred, gdisp = get_groups(data, dt_yhat.labels.ravel())
    tr = train.convert_to_dataframe()[0]
    tr["group_label"] = gpred
    tr["predictions"] = dt_yhat1.labels.ravel()
    sns.scatterplot(
        y="predictions", x="age", data=tr, hue="group_label"
    ).get_figure().savefig(f"initital_group_predictions_{figname}.png")
    return tr


def preds(clf, test):

    dt_yhat = clf.predict(test, threshold=False)
    dt_yhat1 = clf.predict(test)

    # get group labels
    gpred, gdisp = get_groups(test, dt_yhat.labels.ravel())
    acc = accuracy_score(test.labels, dt_yhat1.labels)
    tst = test.convert_to_dataframe()[0]
    tst["group_label"] = gpred
    tst["predictions"] = dt_yhat.labels.ravel()
    return acc, gdisp, tst


def make_score(res):
    # For giving back score which is combination of error and fairness
    res["inv violate"] = 1 - abs(res.fp_violations)
    res["inv error"] = 1 - res.errors
    res["score"] = stats.hmean(res[["inv violate", "inv error"]], axis=1)
    return res


train, test = bank_preproc()
results = pareto_curve(train, "bank")
res = pd.DataFrame(results)
res = make_score(res)
res.to_pickle("bank_pareto.pkl")
print("gamma values results")
print(res)
gamma = res.gamma[res.score.idxmax()]
# result dataframe with group predictions
res = group_preds_demo(train, "bank", gamma)
clf = GerryFairClassifier(C=10, printflag=True, gamma=gamma, max_iters=100)
clf.fit(train, early_termination=True)
gerry_acc, gerry_gamma_disp, tst = preds(clf, train)
result = pd.DataFrame()
# Initializing Logreg classifier for comparision
lr = LogisticRegression(solver="liblinear")
# Getting its values
lr.fit(train.features, train.labels.ravel())
preds = lr.predict_proba(train.features)[:, 1]
lr_acc = lr.score(train.features, train.labels.ravel())
# Getting disparity for LR classifier
lr_disp = get_groups(train, preds)[1]

result["Accuracy"] = [gerry_acc, lr_acc]
result["Fairness Disparity"] = [gerry_gamma_disp, lr_disp]
result.index = ["GerryFair Classifier", "Logistic Regression"]
print(result)
# Trying to fit classifier with more iterations and less gamma value
clf = GerryFairClassifier(C=10, printflag=True, gamma=0.0005, max_iters=1400)
clf.fit(train)
