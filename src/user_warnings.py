from src.user_message import UserMessage


class UserWarning(UserMessage):
    def __init__(self, message="No logged warning message"):
        self.message = message + "\n"
        super().__init__(self.message)


class ObjectResetWarning(UserWarning):
    def __init__(self):
        super().__init__("Object state has been reset.")


class ObjectNotRunWarning(UserWarning):
    def __init__(self):
        message = """ This attempted action is being performed on an
         object that has not yet been run but expects it to be run.
         Run the object then try again.
        """
        super().__init__(message)


class UninitializedObjectWarning(UserWarning):
    def __init__(self):
        message = """ An attempted run was being performed on an
         object that has no initialized state. Aborting this run
         attempt.
        """
        super().__init__(message)
