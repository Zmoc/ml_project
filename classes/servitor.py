import contextlib
import os
import sqlite3
import sys

import llama_cpp
import yaml
from llama_cpp import Llama

from classes.port_scan import Port_Tools


class Servitor:
    def __init__(self):
        self.role_frame = (
            "You are a cybersecurity agent tasked with defending the local system."
        )
        self.conn = sqlite3.connect("agent_memory.db")
        self.c = self.conn.cursor()

        self.c.execute(
            """
            CREATE TABLE IF NOT EXISTS conversations (
                id INTEGER PRIMARY KEY,
                user_input TEXT,
                agent_response TEXT,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        """
        )
        self.conn.commit()
        self.store_conversation(" ", " ")
        llama_cpp.llama_print_system_info = lambda: b""
        with open("config/config.yaml", "r") as f:
            config = yaml.safe_load(f)
        model_path = config["llm"]["path"]
        n_threads = config["llm"]["n_threads"]
        device = config["llm"]["device"]
        print("Please wait...loading brain...")
        with suppress_stdout_stderr():
            self.llm = Llama(
                model_path=model_path,
                n_ctx=2048,
                n_threads=n_threads,
                n_gpu_layers=0,
                # device=device,
            )
        print("Brain loaded")
        print(
            f"Configurations loaded, Model: {model_path}, Threads: {n_threads}, Device: {device}"
        )
        print(
            "My name is Servitor. I am a cybersecurity agent tasked with defending your local system."
        )
        self.get_input()

    def get_input(self):
        print("Servitor is ready. Type your command:")
        while True:
            try:
                user_input = input("> ")
            except (KeyboardInterrupt, EOFError):
                print("\n Agent exiting.")
                break

            if user_input.lower() in ["exit", "quit"]:
                print("Agent exiting.")
                break

            intent = self.intent(user_input)

            try:
                if intent == "scan_ports":
                    port = Port_Tools()
                    print("[*] Scanning ports...")

                    try:
                        print("[*] Generating summary...")
                        summary = self.summarize_report(port.report)
                        print("\nSummary:\n", summary)
                        self.store_conversation(user_input, summary)
                    except Exception as e:
                        print(f"[!] Failed to summarize report: {e}")

                elif intent == "ignore":
                    print("No action required.")

                elif intent == "request_clarification":
                    print("I need more context to understand the command.")

                else:
                    print("Unrecognized action. Please try again.")
            except Exception as e:
                print(e)

    def query(self, prompt, max_tokens=512):
        print("Prompt Recieved.")
        response = self.llm(
            prompt,
            max_tokens=max_tokens,
            # stop=["</s>"],
            echo=False,
            stream=False,
            temperature=0.7,
            top_p=0.95,
        )

        print("Prompt processed.")

        # Print raw response to inspect
        print(f"Raw response: {response}")

        # Clean up the response
        response_text = response["choices"][0]["text"].strip()
        response_text = "\n".join(
            [line.strip() for line in response_text.splitlines() if line.strip()]
        )  # Remove empty lines

        if len(response_text) == 0:  # If response is still empty, handle gracefully
            response_text = "No meaningful response generated."

        return response_text

    def intent(self, user_input, prefix="", suffix=""):
        """Determines the user's intent based on input."""
        last_conversations = self.retrieve_last_conversations(3)
        """context = "\n".join(
            [f"User: {conv[0]}\nAgent: {conv[1]}" for conv in last_conversations]
        )"""
        framed_input = f"{prefix.strip()} {user_input.strip()} {suffix.strip()}".strip()
        intent_prompt = f"""[INST]

{self.role_frame}

Decide what the following instruction is asking you to do.

Instruction: "{framed_input}"

Respond with only one of the following:
scan_ports
ignore
request_clarification

Only use these keywords exactly.[/INST]
"""
        print("Processing prompt")
        llm_response = self.query(intent_prompt)
        print(f"Response: {llm_response}")
        response_lower = llm_response.lower().replace("\\_", "_")
        return {
            "scan_ports": "scan_ports",
            "ignore": "ignore",
            "request_clarification": "request_clarification",
        }.get(response_lower, "unrecognized")

    def summarize_report(self, report_text):
        """Asks the LLM to summarize the port scan report."""
        summary_prompt = f"""[INST]
    {self.role_frame}
    
    A port scan was run on the local system. Here's the result:

    {report_text}

    Please summarize this report and highlight any ports or services that might require security review.[/INST]
    """
        return self.query(summary_prompt, max_tokens=512)

    # Function to store conversation
    def store_conversation(self, user_input, agent_response):
        self.c.execute(
            """
            INSERT INTO conversations (user_input, agent_response)
            VALUES (?, ?)
        """,
            (user_input, agent_response),
        )
        self.conn.commit()

    # Retrieve the last N conversations (example: last 5)
    def retrieve_last_conversations(self, n):
        self.c.execute(
            "SELECT user_input, agent_response FROM conversations ORDER BY timestamp DESC LIMIT ?",
            (n,),
        )
        return self.c.fetchall()


@contextlib.contextmanager
def suppress_stdout_stderr():
    """Suppresses stdout and stderr during execution."""
    try:
        with open(os.devnull, "w") as devnull:
            old_stdout, old_stderr = sys.stdout, sys.stderr
            sys.stdout, sys.stderr = devnull, devnull
            yield
    finally:
        sys.stdout, sys.stderr = old_stdout, old_stderr
