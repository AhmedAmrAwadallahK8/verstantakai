from sklearn.model_selection import KFold


class ModelSearch():
    supported_models = [
        "RandomForestClassifier",
        "LinearRegression"
    ]

    supported_hyperparams = {
        "RandomForestClassifier": [
            ""
        ],
        "LinearRegression": [
            ""
        ]
    }

    supported_metrics = {
        "RandomForestClassifier": [
            "accuracy_score"
        ],
        "LinearRegression": [
            "accuracy_score"
        ]
    }

    def __init__(
        self,
        x,
        y_true,
        clf_packages: list,
        n_folds=1
    ):
        self.x = x
        self.y_true = y_true
        self.clf_packages = clf_packages
        self.n_folds = n_folds
        self.search_complete = False
        self.model_num = {}
        self.optimal_results = {}  # dict
        self.results = {}  # dict of list of dict results that look like
        self.data_state = {}

    def check_valid_package(self):
        for clf_package in self.clf_packages:
            clf_object = clf_package[0]
            clf_name = clf_object.__name__
            hyperparam = clf_package[1]
            metrics = clf_package[2]
            self.check_supported_model(clf_name)
            self.check_valid_params(clf_name, hyperparam)
            self.check_valid_metrics(clf_name, metrics)

    def check_supported_model(self, clf_name: str):
        if clf_name in self.supported_models:
            return
        else:
            raise Exception(clf_name, ": is not a supported model")
        return 0

    def check_valid_params(self, clf_name: str, hyperparam):
        return 0

    def check_valid_metrics(self, clf_name: str, metrics):
        return 0

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

    def run(self):
        kf = KFold(n_splits=self.n_folds)
        for clf_package in self.clf_packages:
            ret, ret_data = {}, {}
            clf_object = clf_package[0]
            clf_hyperparam = clf_package[1]
            metrics = clf_package[2]
            for id, (train_indices, test_indices) in enumerate(kf.split(self.x, self.y_true)):

                clf = clf_object(**clf_hyperparam)  # unpack parameters into clf if they exist
                clf.fit(self.x[train_indices], self.y_true[train_indices])
                y_pred = clf.predict(self.x[test_indices])

                ret_data[id] = {"train_indices": train_indices,
                                "test_indices": test_indices}
                ret[id] = {'clf': clf}

                for metric in metrics:
                    metric_name = metric.__name__
                    ret[id][metric_name] = metric(self.y_true[test_indices], y_pred)

            clf_name = clf_object.__name__
            self.record_results(clf_name, ret, ret_data)

        self.search_complete = True
