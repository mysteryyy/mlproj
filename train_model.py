import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from aif360.metrics import ClassificationMetric
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold
from aif360.datasets import StructuredDataset
from aif360.algorithms.preprocessing.reweighing import Reweighing
from typing import Dict, Iterable, Any


class fair_models:
    def __init__(
        self,
        suffix,
        unprivileged_groups,
        privileged_groups,
        data,
        reweighing=False,
        drop_prot_feats=False,
    ):
        np.random.seed(2021)  # Set seed for splitting
        self.suffix = suffix
        self.privileged_groups = privileged_groups
        self.unprivileged_groups = unprivileged_groups
        self.train, self.test = data.split([0.7], shuffle=True)
        self.reweighing = reweighing
        self.drop_prot_feats = drop_prot_feats
        if reweighing == True:
            RW = Reweighing(
                unprivileged_groups=self.unprivileged_groups,
                privileged_groups=self.privileged_groups,
            )

            self.train = RW.fit_transform(self.train)

    def remove_protected_feats(self, data, feats_to_drop):
        # function to remove protected feature column from feature set
        feats_to_drop = list(feats_to_drop)
        dt = data.convert_to_dataframe()[0]
        dt = dt.drop(columns=feats_to_drop)
        final_dt = StructuredDataset(
            df=dt, label_names=self.train.label_names, protected_attribute_names=[]
        )
        return final_dt

    def train_model(self, train, reg_param):
        x_train = train.features
        if self.drop_prot_feats == False:
            assert x_train.shape[1] == self.train.features.shape[1]
        y_train = train.labels.ravel()
        clf = make_pipeline(
            StandardScaler(),
            LogisticRegression(
                C=reg_param, max_iter=1000, verbose=1, solver="liblinear"
            ),
        )
        clf.fit(
            x_train,
            y_train,
            logisticregression__sample_weight=train.instance_weights,
        )
        return clf

    def test_metric(self, test, y_pred):
        test_pred = test.copy()
        if len(y_pred.shape) < 2:
            y_pred = y_pred.reshape(len(y_pred), 1)
        test_pred.labels = y_pred
        metric = ClassificationMetric(
            test,
            test_pred,
            unprivileged_groups=self.unprivileged_groups,
            privileged_groups=self.privileged_groups,
        )
        metrics = metric.equal_opportunity_difference()
        print(metric)
        return metrics

    def val_score(self, C_lr):

        scores = []
        metrics = []
        skf = StratifiedKFold(n_splits=5, random_state=2025, shuffle=True)
        for train_index, val_index in skf.split(
            self.train.features, self.train.labels.ravel()
        ):
            train_cv = self.train.subset(train_index)
            test_cv = self.train.subset(val_index)
            test_cv_orig = test_cv.copy()
            if self.drop_prot_feats == True:
                train_cv = self.remove_protected_feats(
                    train_cv, self.privileged_groups[0].keys()
                )
                test_cv = self.remove_protected_feats(
                    test_cv, self.privileged_groups[0].keys()
                )

            clf = self.train_model(train_cv, C_lr)
            score = clf.score(test_cv.features, test_cv.labels.ravel())
            labels = clf.predict(test_cv.features)
            metric = self.test_metric(test_cv_orig, labels)

            scores.append(score)
            metrics.append(metric)

        return np.mean(scores), np.mean(metrics)

    def get_results(self):
        model_scores_lr = dict()
        lr_metrics = dict()
        # function for getting the best hyperparameter with maximum accuracy
        best_hyperparam_acc = lambda model_scores: max(
            model_scores, key=model_scores.get
        )
        # Function for getting the best hyperparameter with minimum absolute value of
        # fairness score
        best_hyperparam_eqop = lambda model_scores: min(
            model_scores, key=lambda x: abs(model_scores[x])
        )

        for i in [1.0, 100.0, 10000.0, 100000.0, 0.001, 0.0001, 0.00001, 0.000001]:

            model_scores_lr[str(i)], lr_metrics[str(i)] = self.val_score(i)

        lr_C_acc = float(best_hyperparam_acc(model_scores_lr))
        # For learning rate with least parity

        lr_C_eqop = float(best_hyperparam_eqop(lr_metrics))
        # Retrain models with best hyperparameter value on the entire training data
        best_lr_acc = self.train_model(self.train, lr_C_acc)
        best_lr_eqop = self.train_model(self.train, lr_C_eqop)
        # Preparing results table
        lr_dict = {
            "C": list(model_scores_lr.keys()),
            "Average CV Accuracy": list(model_scores_lr.values()),
            "Equal Opportunity Difference": list(lr_metrics.values()),
        }
        result_df = pd.DataFrame(lr_dict)
        result_df["C"] = result_df["C"].astype("float")
        result_df = result_df.sort_values(by="C")
        x_test, y_test = self.test.features, self.test.labels.ravel()
        lr_eqop_wise = {
            "Accuracy": [best_lr_eqop.score(x_test, y_test)],
            "Equal Opportunity Difference": [
                self.test_metric(self.test, best_lr_eqop.predict(x_test))
            ],
        }

        lr_acc_wise = {
            "Accuracy": [best_lr_acc.score(x_test, y_test)],
            "Equal Opportunity Difference": [
                self.test_metric(self.test, best_lr_acc.predict(x_test))
            ],
        }

        model_perf_test = {
            "LR Accuracy Wise": pd.DataFrame(lr_acc_wise),
            "LR Fairness Wise": pd.DataFrame(lr_eqop_wise),
        }

        test_result = pd.concat(
            model_perf_test.values(), keys=model_perf_test.keys(), axis=1
        )
        # Saving figures
        if self.reweighing == True:
            result_df.to_pickle(f"Cross_Val_Results_reweighted_{self.suffix}.pkl")
            test_result.to_pickle(f"models_testset_reweighted_{self.suffix}.pkl")
            # line plot
            sns.lineplot(
                y="Average CV Accuracy",
                x=abs(result_df["Equal Opportunity Difference"]),
                data=result_df,
            ).get_figure().savefig(f"acc_fair_line_reweighted_{self.suffix}.png")
            plt.clf()
            # Scatter plot
            sns.scatterplot(
                y="Average CV Accuracy",
                x=abs(result_df["Equal Opportunity Difference"]),
                data=result_df,
            ).get_figure().savefig(f"acc_fair_scat_reweighted_{self.suffix}.png")
            plt.clf()
            # C vs Accuracy
            sns.lineplot(
                y="Average CV Accuracy",
                x="C",
                data=result_df,
            ).get_figure().savefig(f"acc_v_C_reweighted_{self.suffix}.png")
            plt.clf()
            # C vs fairness
            sns.lineplot(
                y=abs(result_df["Equal Opportunity Difference"]),
                x="C",
                data=result_df,
            ).get_figure().savefig(f"fair_v_C_reweighted_{self.suffix}.png")
            plt.clf()

        else:
            result_df.to_pickle(f"Cross_Val_Results_{self.suffix}.pkl")
            test_result.to_pickle(f"models_testset_{self.suffix}.pkl")
            # line plot
            sns.lineplot(
                y="Average CV Accuracy",
                x=abs(result_df["Equal Opportunity Difference"]),
                data=result_df,
            ).get_figure().savefig(f"acc_fair_line_unweighted_{self.suffix}.png")
            plt.clf()
            # Scatter plot
            sns.scatterplot(
                y="Average CV Accuracy",
                x=abs(result_df["Equal Opportunity Difference"]),
                data=result_df,
            ).get_figure().savefig(f"acc_fair_scat_unweighted_{self.suffix}.png")
            plt.clf()
            # C vs Accuracy
            sns.lineplot(
                y="Average CV Accuracy",
                x="C",
                data=result_df,
            ).get_figure().savefig(f"acc_v_C_unweighted_{self.suffix}.png")
            plt.clf()
            # C vs fairness
            sns.lineplot(
                y=abs(result_df["Equal Opportunity Difference"]),
                x="C",
                data=result_df,
            ).get_figure().savefig(f"fair_v_C_unweighted_{self.suffix}.png")
            plt.clf()

        print("Cross Val result: \n")
        print(result_df)
        print("Test Set Result: \n")
        print(test_result)
        return result_df, test_result
