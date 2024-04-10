# Example list of dictionaries, each representing some kind of record with a chat history
records = [
    {"chat_history": ["Hi", "Hello", "How are you?"], "user_id": 1},
    {"chat_history": ["Hey", "I'm good, thanks!", "And you?"], "user_id": 2}
]

# A dictionary with lambda functions as values for different operations
operations = {
    "chat_history": lambda x: x["chat_history"],
    # Other operations could be defined here as well
}

# Using the lambda function to extract chat histories from each record
chat_histories = [operations["chat_history"](record) for record in records]

print(chat_histories)
