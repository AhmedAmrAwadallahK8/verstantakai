from sklearn.model_selection import KFold
# import custom_exceptions as ce
import src.user_errors as ue


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
            "C"
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
        self.search_complete = None
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
        for hyperparam in hyperparams:
            if hyperparam in valid_hyperparams:
                return
            else:
                ue.UnsupportedHyperparamError(hyperparam, clf_name)
                self.invalid_input = True

    def check_valid_metrics(self, clf_name: str, metrics):
        valid_metrics = self.supported_metrics[clf_name]
        for metric in metrics:
            metric_name = metric.__name__
            if metric_name in valid_metrics:
                return
            else:
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
        for model_id in self.results.keys():
            print(model_id)
            self.print_model_results(self.results[model_id])

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

    def has_hyperparam_permutations(self, clf_object, clf_hyperparam, optim_metric):
        return False
    
    def optimal_search(self):
        return 0

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
                optimal_hyperparam = self.perform_optimal_search(clf_object, clf_hyperparams, optim_metric)
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
    
