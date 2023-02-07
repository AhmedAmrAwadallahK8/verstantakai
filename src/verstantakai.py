from sklearn.model_selection import KFold
import numpy as np
# import custom_exceptions as ce
import src.user_errors as ue
from itertools import product
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.metrics import roc_curve
import random


class ModelSearch():
    supported_models = [
        "RandomForestClassifier",
        "LinearRegression",
        "LogisticRegression"
    ]

    supported_hyperparams = {
        "RandomForestClassifier": [
            "n_estimators", "criterion", "max_depth"
        ],
        "LinearRegression": [
            "n_jobs"
        ],
        "LogisticRegression": [
            "C", "max_iter"
        ]
    }

    supported_metrics = {
        "RandomForestClassifier": [
            "accuracy_score",
            "f1_score",
            "precision_score",
            "recall_score",
            "roc_auc_score",
            "classification_report"
        ],
        "LinearRegression": [
            "mean_squared_error",
            "r2_score"
        ],
        "LogisticRegression": [
            "accuracy_score",
            "f1_score",
            "precision_score",
            "recall_score",
            "roc_auc_score",
            "classification_report"
        ]
    }

    optimal_at_0_metrics = {
        "mean_squared_error"
    }

    invalid_optimizing_metrics = {
       "classification_report"
    }

    regression_model = {
        "LinearRegression"
    }

    classification_model = {
        "LogisticRegression",
        "RandomForestClassifier"
    }

    def __init__(
        self,
        x,
        y_true,
        clf_packages: list,
        n_folds=1,
        generate_plots=False
    ):
        self.generate_plots = generate_plots
        self.is_reset = False
        self.invalid_input = False
        self.x = x
        self.y_true = y_true
        self.clf_packages = clf_packages
        self.n_folds = n_folds
        self.search_complete = False
        self.model_num = {}
        self.optimal_results = {}
        self.results = {}
        self.data_state = {}
        self.plots = {}

        self.ret = {}
        self.ret_data = {}
        self.ret_plots = {}

        self.check_valid_package()

        if self.invalid_input:
            self.reset_state()

    def reset_state(self):
        self.is_reset = True
        self.invalid_input = None
        self.x = None
        self.y_true = None
        self.clf_packages = None
        self.n_folds = None
        self.search_complete = False
        self.model_num = None
        self.optimal_results = None
        self.results = None
        self.data_state = None
        print("Object state has been reset.\n")

    def model_supported(self):
        if self.invalid_input:
            return False
        else:
            return True

    def check_valid_package(self):
        for clf_package in self.clf_packages:
            clf_object = clf_package[0]
            clf_name = clf_object.__name__
            hyperparams = clf_package[1]
            metrics = clf_package[2]

            # Add Check Valid Types Here

            self.check_supported_model(clf_name)
            if self.model_supported():
                self.check_valid_params(clf_name, hyperparams)
                self.check_valid_metrics(clf_name, metrics)

    def check_supported_model(self, clf_name: str):
        if clf_name in self.supported_models:
            return
        else:
            ue.UnsupportedModelError(clf_name)
            self.invalid_input = True

    def mixed_hyperparam_structure(self, single_model, search_model):
        if (single_model is False) and (search_model is False):
            return True
        else:
            return False

    def check_valid_params(self, clf_name: str, hyperparams):
        valid_hyperparams = self.supported_hyperparams[clf_name]
        single_model_structure = True
        search_model_structure = True
        for hyper_key in hyperparams:
            # Check Keys
            if hyper_key not in valid_hyperparams:
                ue.UnsupportedHyperparamError(hyper_key, clf_name)
                self.invalid_input = True

            # Check Values
            hyper_val = hyperparams[hyper_key]
            if type(hyper_val) is list:
                single_model_structure = False
                if len(hyper_val) == 0:
                    ue.EmptyHyperparamListError(hyper_key, clf_name)
                    self.invalid_input = True
            else:
                search_model_structure = False

        if self.mixed_hyperparam_structure(
            single_model_structure,
            search_model_structure
        ):
            ue.HyperparamListNonListMismatchError(clf_name)
            self.invalid_input = True

    def check_valid_metrics(self, clf_name: str, metrics):
        valid_metrics = self.supported_metrics[clf_name]
        first_metric = True
        for metric in metrics:
            metric_name = metric.__name__
            if first_metric:
                first_metric = False
                if metric_name in self.invalid_optimizing_metrics:
                    ue.InvalidOptimizingMetricError(metric_name, clf_name)
                    self.invalid_input = True
            if metric_name not in valid_metrics:
                ue.UnsupportedMetricError(metric_name, clf_name)
                self.invalid_input = True

    def get_raw_results(self):
        return self.results

    def get_data_state(self):
        return self.data_state

    def get_best_results(self):
        return 0

    def print_model_results(self, results: dict):
        for key in results.keys():
            print("\tFold", key, ": ", results[key])

    def print_results(self):
        if self.search_complete:
            for model_id in self.results.keys():
                print(model_id)
                self.print_model_results(self.results[model_id])
        else:
            print("This object has not been run. Run the object and then try to print results")

    def record_results(
        self,
        clf_name: str,
        ret: dict,
        ret_data: dict,
        ret_plots: dict
    ):
        if clf_name in self.model_num.keys():
            self.model_num[clf_name] += 1
        else:
            self.model_num[clf_name] = 1
        clf_name = clf_name + "#" + str(self.model_num[clf_name])
        self.results[clf_name] = ret
        self.data_state[clf_name] = ret_data
        self.plots[clf_name] = ret_plots

    def run_metrics(self):
        return 0

    def has_hyperparam_permutations(self, clf_hyperparams):
        for k in clf_hyperparams.keys():
            hyperparam_values = clf_hyperparams[k]
            if type(hyperparam_values) is list:
                return True
            else:
                return False

    def current_model_is_better(
        self,
        performance,
        optimal_performance,
        optim_metric
    ):
        if optim_metric in self.optimal_at_0_metrics:
            performance *= -1
            optimal_performance *= -1

        if performance > optimal_performance:
            return True
        else:
            return False

    def perform_optimal_search(
        self,
        clf_object,
        clf_hyperparams,
        optim_metric,
        x_train,
        y_train,
        x_test,
        y_test
    ):
        hyperparam_combinations = [
            dict(
                zip(clf_hyperparams, v)
            ) for v in product(*clf_hyperparams.values())
        ]
        optimal_hyperparam = {}
        optimal_performance = {}
        first_iteration = True
        for hyperparam in hyperparam_combinations:
            clf = clf_object(**hyperparam)
            clf.fit(x_train, y_train)
            y_pred = clf.predict(x_test)
            performance = optim_metric(y_test, y_pred)
            if first_iteration:
                first_iteration = False
                optimal_hyperparam = hyperparam
                optimal_performance = performance
            else:
                if self.current_model_is_better(
                    performance,
                    optimal_performance,
                    optim_metric.__name__
                ):
                    optimal_performance = performance
                    optimal_hyperparam = hyperparam

        return optimal_hyperparam

    def regression_plots(self, id, y_true, y_pred):
        fig = self.calibration_plot(y_true, y_pred)
        self.ret_plots[id]["calibration_plot"] = fig

    def calibration_plot(self, y_true, y_pred):
        trace1 = go.Scatter(
            x=y_true,
            y=y_pred,
            name="Y Pred v Y Pred",
            marker=dict(
                color='rgb(34,163,192)'
                    ),
            mode="markers"
        )
        trace2 = go.Scatter(
            x=y_true,
            y=y_true,
            name='Y True v Y True',
            mode="markers"
        )

        fig = make_subplots()
        fig.add_trace(trace1)
        fig.add_trace(trace2)
        fig['layout'].update(
            height=600, width=800, title="Calibration Plot",
            xaxis=dict(
                tickangle=-90
            ))

        return fig

    def classification_plots(self, id, y_true, y_pred):
        self.ret_plots[id]["roc_curve"] = self.roc_curve(y_true, y_pred)

    def binary_roc_curve(self, y_true, y_pred_prob, pos_label):
        fpr, tpr, tholds = roc_curve(y_true, y_pred_prob, pos_label=pos_label)
        trace1 = go.Scatter(
            x=fpr,
            y=tpr,
            marker=dict(
                color='rgb(34,163,192)'
                    ),
        )

        fig = make_subplots()
        fig.add_trace(trace1)
        fig['layout'].update(
            height=600, width=800, title="ROC Curve",
            xaxis=dict(
                tickangle=-90
            ))

        return fig

    def generate_roc_curve(self, fpr, tpr):
        trace1 = go.Scatter(
            x=fpr,
            y=tpr,
            marker=dict(
                color='rgb(34,163,192)'
                    ),
        )

        fig = make_subplots()
        fig.add_trace(trace1)
        fig['layout'].update(
            height=600, width=800, title="ROC Curve",
            xaxis=dict(
                tickangle=-90
            ))

        return fig

    def multi_roc_curve(self, y_true, y_pred_prob, class_count):
        fig = make_subplots()
        for i in range(class_count):
            preds = y_pred_prob[:, i]
            fpr, tpr, thresholds = roc_curve(y_true, preds, pos_label=i)
            r = random.randint(0, 255)
            g = random.randint(0, 255)
            b = random.randint(0, 255)

            colour = "rgb("
            colour += str(r)
            colour += ","
            colour += str(g)
            colour += ","
            colour += str(b)
            colour += ")"

            label_name = "Label " + str(i) + " as Positive Curve"

            trace = go.Scatter(
                x=fpr,
                y=tpr,
                marker=dict(
                    color=colour
                        ),
                name=label_name,
            )
            fig.add_trace(trace)

        fig['layout'].update(
            height=600, width=800, title="ROC Curve",
            xaxis=dict(
                tickangle=-90
            ))
        return fig

    def binary_classification(self, class_count):
        if (class_count <= 2) and (class_count > 0):
            return True
        else:
            False

    def multi_classification(self, class_count):
        if class_count > 2:
            return True
        else:
            False

    def run_model(self, clf_package, kf):
        self.ret, self.ret_data, self.ret_plots = {}, {}, {}
        clf_object = clf_package[0]
        clf_hyperparams = clf_package[1]
        metrics = clf_package[2]
        optim_metric = metrics[0]
        clf_name = clf_object.__name__
        for id, (train_i, test_i) in enumerate(kf.split(self.x, self.y_true)):
            if self.has_hyperparam_permutations(clf_hyperparams):
                optimal_hyperparam = self.perform_optimal_search(
                    clf_object,
                    clf_hyperparams,
                    optim_metric,
                    self.x[train_i],
                    self.y_true[train_i],
                    self.x[test_i],
                    self.y_true[test_i]
                )
            else:
                optimal_hyperparam = clf_hyperparams

            # Run Final Model
            clf = clf_object(**optimal_hyperparam)
            clf.fit(self.x[train_i], self.y_true[train_i])

            self.ret_data[id] = {"train_indices": train_i,
                                 "test_indices": test_i}
            self.ret[id] = {'clf': clf}
            self.ret_plots[id] = {}

            y_pred = clf.predict(self.x[test_i])
            # Add Metrics
            for metric in metrics:
                metric_name = metric.__name__
                met_result = metric(self.y_true[test_i], y_pred)
                self.ret[id][metric_name] = met_result

            # Add Plots
            if self.generate_plots:
                if clf_name in self.regression_model:
                    calib_plot = self.calibration_plot(self.y_true[test_i],
                                                       y_pred)
                    self.ret_plots[id]["calibration_plot"] = calib_plot
                elif clf_name in self.classification_model:
                    y_true_test = self.y_true[test_i]
                    class_count = np.unique(y_true_test).shape[0]
                    y_pred_prob = clf.predict_proba(self.x[test_i])
                    roc_plot = self.multi_roc_curve(y_true_test,
                                                    y_pred_prob,
                                                    class_count)
                    self.ret_plots[id]["roc_curve"] = roc_plot
                    # if self.binary_classification(class_count):
                    #     # Binary Classification
                    #     print("Binary")
                    # elif self.multi_classification(class_count):
                    #     # Multiclass
                    #     print("Multiclass plotting not supported atm")
                    # else:
                    #     # Unexpected
                    #     print("Unexpected classification plot case")

    def run_models(self):
        kf = KFold(n_splits=self.n_folds)
        for clf_package in self.clf_packages:
            self.run_model(clf_package, kf)
            clf_object = clf_package[0]
            clf_name = clf_object.__name__
            self.record_results(
                clf_name,
                self.ret,
                self.ret_data,
                self.ret_plots
            )
        self.search_complete = True

    def run(self):
        if self.is_reset:
            print("This object has not been setup properly. Aborting attempted run.")
        else:
            self.run_models()
