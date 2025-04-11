import sqlite3

# Set up the database
conn = sqlite3.connect("agent_memory.db")
c = conn.cursor()

# Create table to store conversations
c.execute(
    """
    CREATE TABLE IF NOT EXISTS conversations (
        id INTEGER PRIMARY KEY,
        user_input TEXT,
        agent_response TEXT,
        timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
    )
"""
)
conn.commit()


# Function to store conversation
def store_conversation(user_input, agent_response):
    c.execute(
        """
        INSERT INTO conversations (user_input, agent_response)
        VALUES (?, ?)
    """,
        (user_input, agent_response),
    )
    conn.commit()


# Retrieve the last N conversations (example: last 5)
def retrieve_last_conversations(n=5):
    c.execute(
        "SELECT user_input, agent_response FROM conversations ORDER BY timestamp DESC LIMIT ?",
        (n,),
    )
    return c.fetchall()


def generate_contextual_prompt(user_input):
    # Retrieve the last 3 conversations (for example)
    last_conversations = retrieve_last_conversations(3)
    context = "\n".join(
        [f"User: {conv[0]}\nAgent: {conv[1]}" for conv in last_conversations]
    )

    # Build the prompt with context
    prompt = f"""
You are a cybersecurity assistant. Here is the conversation history:

{context}

Now, respond to the following user input:

User: {user_input}
Agent:
"""
    return prompt


# Example usage
# user_input = "What are the open ports now?"
# prompt = generate_contextual_prompt(user_input)
# response = ask_llm(prompt)
# print(response)


# Example usage
# last_conversations = retrieve_last_conversations()
# for conversation in last_conversations:
#    print(f"User: {conversation[0]}")
#    print(f"Agent: {conversation[1]}")


# Example usage
# store_conversation("Check open ports", "Ports 22, 80, 443 are open.")
