from yamllm import ConversationStore
import pandas as pd
from tabulate import tabulate
"""
This script retrieves and displays conversation messages from a SQLite database.
Modules:
    yamllm: Provides LLM and ConversationStore classes for managing conversation history.
    pandas: Used for creating a DataFrame from the retrieved messages.
    tabulate: Used for printing the DataFrame in a tabular format.
Functions:
    None
Variables:
    history (ConversationStore): An instance of ConversationStore to interact with the conversation history database.
    messages (list): A list of messages retrieved from the conversation history for a specific session.
    df (DataFrame): A pandas DataFrame created from the list of messages.
Usage:
    Run the script to print the conversation messages for 'session1' in a tabular format.
"""

# load the history into a variable
history = ConversationStore(r"yamllm\memory\conversation_history.db")

# use the get_messages() method to load the messages and load into a DataFrame for easy viewing
messages = history.get_messages()
df = pd.DataFrame(messages)


# Print the table using tabulate, which displays in an easy to read format
print(tabulate(df, headers='keys', tablefmt='psql'))