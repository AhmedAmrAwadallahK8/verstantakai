class UserError():
    def __init__(self, message="No logged message"):
        self.message = "Invalid input. Use the following message to debug.\n"
        self.message += message + "\n"
        self.message += "Object State is now reset.\n"
        print(self.message)


class UnsupportedModelError(UserError):
    def __init__(self, model_name):
        self.message = model_name + " is not supported."
        super().__init__(self.message)
