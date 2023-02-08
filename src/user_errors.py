from src.user_message import UserMessage


class UserError(UserMessage):
    def __init__(self, message="No logged message"):
        self.message = "Invalid input. Use the following message to debug.\n"
        self.message += message + "\n"
        super().__init__(self.message)


class UnsupportedModelError(UserError):
    def __init__(self, model_name):
        self.message = model_name + " is not a supported model."
        super().__init__(self.message)


class UnsupportedHyperparamError(UserError):
    def __init__(self, hyperparam, clf_name):
        self.message = hyperparam + " is not a supported hyper-parameter for a"
        self.message += "model of type " + clf_name + "."
        super().__init__(self.message)


class EmptyHyperparamListError(UserError):
    def __init__(self, hyperparam, clf_name):
        self.message = hyperparam + " has give an empty list of values for "
        self.message += clf_name + ". Remove or add values."
        super().__init__(self.message)


class HyperparamListNonListMismatchError(UserError):
    def __init__(self, clf_name):
        self.message = clf_name + "contains a dictionary with a mixture of "
        self.message += "list and non list types. Convert all dictionary "
        self.message += "values to be of only list type or only numeric type"
        super().__init__(self.message)


class UnsupportedMetricError(UserError):
    def __init__(self, metric, clf_name):
        self.message = metric + " is not a supported metric for a model of "
        self.message += "type " + clf_name + "."
        super().__init__(self.message)


class InvalidOptimizingMetricError(UserError):
    def __init__(self, metric, clf_name):
        self.message = "Reminder that the first metric listed within the"
        self.message += " metrics list is used as the optimizing metric.\n"
        self.message += metric + " is not a valid optimizing metric. This "
        self.message += "invalid optimizing metric is present in a "
        self.message += clf_name + " package."
        super().__init__(self.message)
