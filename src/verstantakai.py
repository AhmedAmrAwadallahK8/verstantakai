from sklearn.model_selection import KFold
# import custom_exceptions as ce
import src.user_errors as ue
from itertools import product


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
            "accuracy_score", "f1_score", "precision_score", "recall_score", "roc_auc_score", "classification_report"
        ],
        "LinearRegression": [
            "mean_squared_error", "r2_score"
        ],
        "LogisticRegression": [
            "accuracy_score", "f1_score", "precision_score", "recall_score", "roc_auc_score", "classification_report"
        ]
    }

    optimal_at_0_metrics = {
        "mean_squared_error"
    }

    invalid_optimizing_metrics = {
       "classification_report"
    }

    def __init__(
        self,
        x,
        y_true,
        clf_packages: list,
        n_folds=1
    ):
        self.is_reset = False
        self.invalid_input = False
        self.x = x
        self.y_true = y_true
        self.clf_packages = clf_packages
        self.n_folds = n_folds
        self.search_complete = False
        self.model_num = {}
        self.optimal_results = {}  # dict
        self.results = {}  # dict of list of dict results that look like
        self.data_state = {}

        self.ret = {}
        self.ret_data = {}

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

            #Check Valid Types

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

    def check_valid_params(self, clf_name: str, hyperparams):
        valid_hyperparams = self.supported_hyperparams[clf_name]
        single_model_structure = True
        model_search_structure = True
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
                model_search_structure = False

        if (single_model_structure is False) and (model_search_structure is False):
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
        ret_data: dict
    ):
        if clf_name in self.model_num.keys():
            self.model_num[clf_name] += 1
        else:
            self.model_num[clf_name] = 1
        clf_name = clf_name + "#" + str(self.model_num[clf_name])
        self.results[clf_name] = ret
        self.data_state[clf_name] = ret_data

    def run_metrics(self):
        return 0

    def has_hyperparam_permutations(self, clf_hyperparams):
        for k in clf_hyperparams.keys():
            hyperparam_values = clf_hyperparams[k]
            if type(hyperparam_values) is list:
                return True
            else:
                return False

    def current_model_is_better(self, performance, optimal_performance, optim_metric):
        if optim_metric in self.optimal_at_0_metrics:
            performance *= -1
            optimal_performance *= -1

        if performance > optimal_performance:
            return True
        else:
            return False

    def perform_optimal_search(self, clf_object, clf_hyperparams, optim_metric, x_train, y_train, x_test, y_test):
        hyperparam_combinations = [dict(zip(clf_hyperparams, v)) for v in product(*clf_hyperparams.values())]
        optimal_hyperparam = {}
        optimal_performance = {}
        first_iteration = True
        for hyperparam in hyperparam_combinations:
            clf = clf_object(**hyperparam)  # unpack parameters into clf if they exist
            clf.fit(x_train, y_train)
            y_pred = clf.predict(x_test)
            performance = optim_metric(y_test, y_pred)
            if first_iteration:
                first_iteration = False
                optimal_hyperparam = hyperparam
                optimal_performance = performance
            else:
                if self.current_model_is_better(performance, optimal_performance, optim_metric.__name__):
                    optimal_performance = performance
                    optimal_hyperparam = hyperparam

        return optimal_hyperparam

    def fit_model(self):
        return 0

    def run_model(self, clf_package, kf):
        self.ret, self.ret_data = {}, {}
        clf_object = clf_package[0]
        clf_hyperparams = clf_package[1]
        metrics = clf_package[2]
        optim_metric = metrics[0]
        for id, (train_indices, test_indices) in enumerate(kf.split(self.x, self.y_true)):
            if self.has_hyperparam_permutations(clf_hyperparams):
                optimal_hyperparam = self.perform_optimal_search(
                    clf_object,
                    clf_hyperparams,
                    optim_metric,
                    self.x[train_indices],
                    self.y_true[train_indices],
                    self.x[test_indices],
                    self.y_true[test_indices])
            else:
                optimal_hyperparam = clf_hyperparams

            clf = clf_object(**optimal_hyperparam)  # unpack parameters into clf if they exist
            clf.fit(self.x[train_indices], self.y_true[train_indices])
            y_pred = clf.predict(self.x[test_indices])

            self.ret_data[id] = {"train_indices": train_indices,
                                 "test_indices": test_indices}
            self.ret[id] = {'clf': clf}

            for metric in metrics:
                metric_name = metric.__name__
                self.ret[id][metric_name] = metric(self.y_true[test_indices], y_pred)

    def run_models(self):
        kf = KFold(n_splits=self.n_folds)
        for clf_package in self.clf_packages:
            self.run_model(clf_package, kf)
            clf_object = clf_package[0]
            clf_name = clf_object.__name__
            self.record_results(clf_name, self.ret, self.ret_data)
        self.search_complete = True

    def run(self):
        if self.is_reset:
            print("This object has not been setup properly. Aborting attempted run.")
        else:
            self.run_models()
    
