class UnsupportedModelException(Exception):
    def __init__(
        self,
        invalid_model: str
    ):
        self.message = invalid_model + " is not a supported model"
        super().__init__(self.message)
