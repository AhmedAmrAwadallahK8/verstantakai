# Opportunity of abstraction here, some greater class that both Warning and
# Error inherits from
class UserWarning():
    def __init__(self, message="No logged message"):
        self.message = """Questionable input. Use the following message to
                          debug.\n"""
        self.message += message + "\n"
        print(self.message)
