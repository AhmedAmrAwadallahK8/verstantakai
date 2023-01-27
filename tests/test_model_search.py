import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LinearRegression
from src.verstantakai import ModelSearch
import sys


class TestModelSearch():
    def __init__(self):
        return None

    def execute_tests(self):
        self.single_model_test()
        self.repeated_model_test()
        self.multiple_model_test()
        self.invalid_model_test()
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

        if len(results) != 1:
            raise Exception("Input one model but results returned multiple")

        first_model = list(results.keys())[0]
        if first_model != "RandomForestClassifier#1":
            raise Exception("Input model recieved unexpected name")
        
        results_first_model = results[first_model]

        if len(results_first_model) != 5:
            raise Exception("Input model did not compute exactly 5 different folds")

        results_first_model_first_fold = results_first_model[0]

        if len(results_first_model_first_fold) != 2:
            raise Exception("Fold did not have exactly 2 key value pairs when 2 were expected")

        metric_accuracy = list(results_first_model_first_fold.keys())[1]
        if metric_accuracy != "accuracy_score":
            raise Exception("Second key of dictionary returned an unexpected metric name when accuracy_score was expected")
        
        clf_obj = results_first_model_first_fold["clf"]
        if type(clf_obj) is not RandomForestClassifier:
            raise Exception("Returned model is not type RandomForestClassifier")

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

        if len(results) != 2:
            raise Exception("Expected 2 models but got ", len(results))

        first_model = list(results.keys())[0]
        if first_model != "RandomForestClassifier#1":
            raise Exception("Input model recieved unexpected name")

        first_model = list(results.keys())[1]
        if first_model != "RandomForestClassifier#2":
            raise Exception("Input model recieved unexpected name")

    def multiple_model_test(self):
        x = np.array(
            [[1, 2], [3, 4], [4, 5], [4, 5], [4, 5], [4, 5], [4, 5], [4, 5]]
        )
        y = np.ones(x.shape[0])
        n_folds = 5
        metrics = [accuracy_score]
        clsf_param = [(RandomForestClassifier, {}, metrics), (LinearRegression, {}, metrics)]
        srch = ModelSearch(x, y, clsf_param, n_folds)
        srch.run()
        results = srch.get_raw_results()

        if len(results) != 2:
            raise Exception("Expected 2 models but got ", len(results))

        first_model = list(results.keys())[0]
        if first_model != "RandomForestClassifier#1":
            raise Exception("Input model recieved unexpected name")

        first_model = list(results.keys())[1]
        if first_model != "LinearRegression#1":
            raise Exception("Input model recieved unexpected name")

    def invalid_model_test(self):
        x = np.array([[1, 2]])
        y = np.ones(x.shape[0])
        metrics = [accuracy_score]
        hyperparams = {}
        clf_pack = [(RandomForestClassifier, hyperparams, metrics)]
        srch = ModelSearch(x, y, clf_pack)
        try:
            srch.check_supported_model(None)
        except Exception:
            print("Unexpected error: {0}".format(sys.exc_info()[0]))
            print("Found invalid model")
            return

