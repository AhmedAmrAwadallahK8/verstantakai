class UserError():
    def __init__(self, message="No logged message"):
        self.message = "Invalid input. Use the following message to debug.\n"
        self.message += message + "\n"
        print(self.message)


class UnsupportedModelError(UserError):
    def __init__(self, model_name):
        self.message = model_name + " is not a supported model."
        super().__init__(self.message)


class UnsupportedHyperparamError(UserError):
    def __init__(self, hyperparam):
        self.message = hyperparam + " is not a supported hyper-parameter."
        super().__init__(self.message)


class UnsupportedMetricError(UserError):
    def __init__(self, metric):
        self.message = metric + " is not a supported metric."
        super().__init__(self.message)
