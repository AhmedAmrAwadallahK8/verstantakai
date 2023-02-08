from src.user_message import UserMessage


class UserMessageQueue():
    def __init__(self):
        self.message_queue = []

    def add_message(self, message: UserMessage):
        self.message_queue.append(message)

    def print_messages(self):
        for msg in self.message_queue:
            msg.print_message()

    def empty_queue(self):
        self.message_queue.clear()

    def process_queue(self):
        self.print_messages()
        self.empty_queue()
