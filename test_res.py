import os
import pandas as pd
from sklearn.metrics import accuracy_score
from aif360.algorithms.preprocessing.optim_preproc_helpers.data_preproc_functions import (
    load_preproc_data_adult,
)
from aif360.datasets import BankDataset
from train_model import fair_models
from scipy import stats


if (
    os.path.isdir("/home/sahil/ML/Results") == False
):  # Change filepath to change location for storing results
    os.mkdir("Results")

os.chdir("/home/sahil/ML/Results")


def fair_acc_test(train_obj, cross_val):
    cross_val["C"] = cross_val["C"].astype(float)
    cross_val = cross_val.sort_values(by="C")
    cross_val["eq"] = 1 - abs(cross_val["Equal Opportunity Difference"])
    # Calculate fairness+accuracy score
    cross_val["score"] = stats.hmean(cross_val[["Average CV Accuracy", "eq"]], axis=1)
    # cross_val["change_2"] = abs(cross_val.change.diff())
    cross_val = cross_val.drop(["eq"], axis=1)
    best_c_value = float(cross_val.at[cross_val.score.idxmax(), "C"])
    print(best_c_value)
    clf = train_obj.train_model(train_obj.train, best_c_value)
    x_test = train_obj.test.features
    y_test = train_obj.test.labels
    lr_fairacc_wise = {
        "Accuracy": [clf.score(x_test, y_test)],
        "Equal Opportunity Difference": [
            train_obj.test_metric(train_obj.test, clf.predict(x_test))
        ],
    }
    return pd.DataFrame(lr_fairacc_wise), best_c_value, cross_val


best_c_values = {}
bank = BankDataset()
data = load_preproc_data_adult(["sex"])
unprivileged_groups_adult = [{"sex": 0}]
privileged_groups_adult = [{"sex": 1}]
unprivileged_groups_bank = [{"age": 0}]
privileged_groups_bank = [{"age": 1}]

weighted_train_adult = fair_models(
    unprivileged_groups=unprivileged_groups_adult,
    privileged_groups=privileged_groups_adult,
    data=data,
    reweighing=True,
    suffix="adult_notdropped",
)

cross_val1, test1 = weighted_train_adult.get_results()
res, best_c, cv1 = fair_acc_test(weighted_train_adult, cross_val1)
best_c_values["adult_notdropped_weighted"] = best_c
print(res)
res.to_pickle("test_weighted_train.pkl")

unweighted_train_adult = fair_models(
    unprivileged_groups=unprivileged_groups_adult,
    privileged_groups=privileged_groups_adult,
    data=data,
    reweighing=False,
    suffix="adult_notdropped",
)


cross_val1, test1 = unweighted_train_adult.get_results()
res, best_c, cv2 = fair_acc_test(unweighted_train_adult, cross_val1)
best_c_values["adult_notdropped_unweighted"] = best_c
res.to_pickle("test_unweighted_train.pkl")

train_adult_drop = fair_models(
    unprivileged_groups=unprivileged_groups_adult,
    privileged_groups=privileged_groups_adult,
    data=data,
    reweighing=False,
    suffix="adult_dropped",
    drop_prot_feats=True,
)
best_c_values["adult_dropped"] = best_c

cross_val1, test1 = train_adult_drop.get_results()
res, best_c, cv3 = fair_acc_test(train_adult_drop, cross_val1)
res.to_pickle("test_adult_drop.pkl")

weighted_train_bank = fair_models(
    unprivileged_groups=unprivileged_groups_bank,
    privileged_groups=privileged_groups_bank,
    data=bank,
    reweighing=True,
    suffix="bank_notdropped",
)

cross_val1, test1 = weighted_train_bank.get_results()
res, best_c, cv4 = fair_acc_test(weighted_train_bank, cross_val1)
best_c_values["bank_notdropped_weighted"] = best_c
res.to_pickle("test_bank_notdrop_weighted.pkl")
print(res)

unweighted_train_bank = fair_models(
    unprivileged_groups=unprivileged_groups_bank,
    privileged_groups=privileged_groups_bank,
    data=bank,
    reweighing=False,
    suffix="bank_notdropped",
)

cross_val2, test2 = unweighted_train_bank.get_results()
res, best_c, cv5 = fair_acc_test(unweighted_train_bank, cross_val2)
best_c_values["bank_notdropped_unweighted"] = best_c
res.to_pickle("test_bank_notdrop_unweighted.pkl")
print(res)

drop_train_bank = fair_models(
    unprivileged_groups=unprivileged_groups_bank,
    privileged_groups=privileged_groups_bank,
    data=bank,
    reweighing=False,
    suffix="bank_dropped",
    drop_prot_feats=True,
)

cross_val3, test3 = drop_train_bank.get_results()
res, best_c, cv6 = fair_acc_test(drop_train_bank, cross_val3)
best_c_values["bank_dropped"] = best_c
res.to_pickle("test_bank_dropped.pkl")
print(res)


print(cross_val1)
print(res)
