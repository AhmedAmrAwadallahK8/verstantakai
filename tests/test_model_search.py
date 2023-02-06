import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import classification_report
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
from tests.metrics import bad_metric
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from src.verstantakai import ModelSearch
from tests.bad_classifier import BadClassifier
from sklearn.datasets import load_breast_cancer
import sys


class TestModelSearch():
    def __init__(self):
        return None

    def execute_tests(self):
        self.single_model_test()
        self.repeated_model_test()
        self.multiple_model_test()
        self.invalid_model_test()
        self.invalid_hyperparam_test()
        self.invalid_metrics_test()
        self.single_hyperparam_test()
        self.empty_value_hyperparam_test()
        self.multi_value_hyperparam_test()
        self.list_hyperparam_test()
        self.invalid_optim_metric_test()
        #self.real_data_test()
        print("All Tests Executed Successfully")

    def single_model_test(self):
        x = np.array(
            [[1, 2], [3, 4], [4, 5], [4, 5], [4, 5], [4, 5], [4, 5], [4, 5]]
        )
        y = np.ones(x.shape[0])
        n_folds = 5
        metrics = [accuracy_score]
        clsf_param = [(RandomForestClassifier, {}, metrics)]
        srch = ModelSearch(x, y, clsf_param, n_folds)
        srch.run()
        results = srch.get_raw_results()

        assert len(results) == 1

        first_model = list(results.keys())[0]
        assert first_model == "RandomForestClassifier#1"

        results_first_model = results[first_model]
        assert len(results_first_model) == 5

        results_first_model_first_fold = results_first_model[0]
        assert len(results_first_model_first_fold) == 2

        metric_accuracy = list(results_first_model_first_fold.keys())[1]
        assert metric_accuracy == "accuracy_score"

        clf_obj = results_first_model_first_fold["clf"]
        assert type(clf_obj) is RandomForestClassifier

    def repeated_model_test(self):
        x = np.array(
            [[1, 2], [3, 4], [4, 5], [4, 5], [4, 5], [4, 5], [4, 5], [4, 5]]
        )
        y = np.ones(x.shape[0])
        n_folds = 5
        metrics = [accuracy_score]
        clsf_param = [(RandomForestClassifier, {}, metrics), (RandomForestClassifier, {}, metrics)]
        srch = ModelSearch(x, y, clsf_param, n_folds)
        srch.run()
        results = srch.get_raw_results()

        assert len(results) == 2

        first_model = list(results.keys())[0]
        assert first_model == "RandomForestClassifier#1"

        second_model = list(results.keys())[1]
        assert second_model == "RandomForestClassifier#2"

    def multiple_model_test(self):
        x = np.array(
            [[1, 2], [3, 4], [4, 5], [4, 5], [4, 5], [4, 5], [4, 5], [4, 5]]
        )
        y = np.ones(x.shape[0])
        n_folds = 5
        metrics_forest = [accuracy_score]
        metrics_linear = [mean_squared_error, r2_score]
        clsf_param = [(RandomForestClassifier, {}, metrics_forest),
                      (LinearRegression, {}, metrics_linear)]
        srch = ModelSearch(x, y, clsf_param, n_folds)
        srch.run()
        results = srch.get_raw_results()

        assert len(results) == 2

        first_model = list(results.keys())[0]
        assert first_model == "RandomForestClassifier#1"

        second_model = list(results.keys())[1]
        assert second_model == "LinearRegression#1"

    def invalid_model_test(self):
        x = np.array([[1, 2]])
        y = np.ones(x.shape[0])
        metrics = [accuracy_score]
        hyperparams = {}
        clf_pack = [(BadClassifier, hyperparams, metrics)]
        srch = ModelSearch(x, y, clf_pack)
        assert srch.is_reset

        # try:
        #     srch.check_supported_model(None)
        # except Exception:
        #     print("Unexpected error: {0}".format(sys.exc_info()[0]))
        #     print("Found invalid model")
        #     return

    def invalid_hyperparam_test(self):
        x = np.array([[1, 2]])
        y = np.ones(x.shape[0])
        metrics = [accuracy_score]
        hyperparams = {"bad_param": 0}
        clf_pack = [(RandomForestClassifier, hyperparams, metrics)]
        srch = ModelSearch(x, y, clf_pack)
        assert srch.is_reset

    def invalid_metrics_test(self):
        x = np.array([[1, 2]])
        y = np.ones(x.shape[0])
        metrics = [bad_metric]
        hyperparams = {}
        clf_pack = [(RandomForestClassifier, hyperparams, metrics)]
        srch = ModelSearch(x, y, clf_pack)
        assert srch.is_reset

    def single_hyperparam_test(self):
        x = np.array(
            [[1, 2], [3, 4], [4, 5], [4, 5], [4, 5], [4, 5], [4, 5], [4, 5]]
        )
        y = np.ones(x.shape[0])
        n_folds = 5
        hyperparams = {"n_estimators": 1, "max_depth": 4}
        metrics = [accuracy_score, f1_score, precision_score, recall_score]
        clsf_param = [(RandomForestClassifier, hyperparams, metrics)]
        srch = ModelSearch(x, y, clsf_param, n_folds)
        srch.run()
        assert srch.search_complete

    def empty_value_hyperparam_test(self):
        x = np.array(
            [[1, 2], [3, 4], [4, 5], [4, 5], [4, 5], [4, 5], [4, 5], [4, 5]]
        )
        y = np.ones(x.shape[0])
        n_folds = 5
        hyperparams = {"criterion": []}
        metrics = [accuracy_score, f1_score, precision_score, recall_score]
        clsf_param = [(RandomForestClassifier, hyperparams, metrics)]
        srch = ModelSearch(x, y, clsf_param, n_folds)
        assert srch.is_reset

    def multi_value_hyperparam_test(self):
        x = np.array(
            [[1, 2], [3, 4], [4, 5], [4, 5], [4, 5], [4, 5], [4, 5], [4, 5]]
        )
        y = np.ones(x.shape[0])
        n_folds = 5
        hyperparams = {"criterion": [1, 2], "n_estimators": 3}
        metrics = [accuracy_score, f1_score, precision_score, recall_score]
        clsf_param = [(RandomForestClassifier, hyperparams, metrics)]
        srch = ModelSearch(x, y, clsf_param, n_folds)
        assert srch.is_reset

    def list_hyperparam_test(self):
        x = np.array(
            [[1, 2], [3, 4], [4, 5], [4, 5], [4, 5], [4, 5], [4, 5], [4, 5]]
        )
        y = np.ones(x.shape[0])
        n_folds = 5
        hyperparams = {"n_estimators": [10, 1], "max_depth": [4, 3, 2, 1]}
        metrics = [accuracy_score, f1_score, precision_score, recall_score]
        clsf_param = [(RandomForestClassifier, hyperparams, metrics)]
        srch = ModelSearch(x, y, clsf_param, n_folds)
        srch.run()
        assert srch.search_complete

    def invalid_optim_metric_test(self):
        x = np.array(
            [[1, 2], [3, 4], [4, 5], [4, 5], [4, 5], [4, 5], [4, 5], [4, 5]]
        )
        y = np.ones(x.shape[0])
        n_folds = 5
        hyperparams = {"n_estimators": [10, 1], "max_depth": [4, 3, 2, 1]}
        metrics = [classification_report, f1_score, precision_score, recall_score]
        clsf_param = [(RandomForestClassifier, hyperparams, metrics)]
        srch = ModelSearch(x, y, clsf_param, n_folds)
        assert srch.is_reset

    def real_data_test(self):
        data = load_breast_cancer()
        x = data.data
        y = data.target
        n_folds = 5
        hyperparams_forest = {"n_estimators": [10, 100, 1], "max_depth": [4, 3, 2, 1]}
        hyperparams_log = {"C": list(np.logspace(-3, 3, 5))}
        
        metrics = [accuracy_score, f1_score, precision_score, recall_score]
        clsf_param = [(RandomForestClassifier, hyperparams_forest, metrics),
                      (LogisticRegression, hyperparams_log, metrics)]
        srch = ModelSearch(x, y, clsf_param, n_folds)
        srch.run()
        assert srch.search_complete



    def plot_functionality_test(self):
        assert False
