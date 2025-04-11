import contextlib
import os
import sys

import llama_cpp
from llama_cpp import Llama

from port_scan import (
    generate_report,
    get_processes_by_port,
    parse_nmap_output,
    run_nmap_scan,
)
from sqllite_memory import (
    generate_contextual_prompt,
    retrieve_last_conversations,
    store_conversation,
)


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


# Disable system info print to prevent Windows OSError
llama_cpp.llama_print_system_info = lambda: b""

model_path = "llm/mistral-7b-instruct-v0.1.Q5_K_M.gguf"
num_cores = os.cpu_count()
# n_threads = int(max(1, (num_cores - 1) / 2))
n_threads = 4
# Load model silently
with suppress_stdout_stderr():
    llm = Llama(
        model_path=model_path,
        n_ctx=2048,
        n_threads=n_threads,
        n_gpu_layers=30,
        device="cuda",
    )


def ask_llm(prompt, max_tokens=512):
    response = llm(
        prompt,
        max_tokens=max_tokens,
        stop=["</s>"],
        echo=False,
        stream=False,
        temperature=0.7,
        top_p=0.95,
    )
    return response["choices"][0]["text"].strip()


def detect_intent(user_input):
    """Determines the user's intent based on input."""

    last_conversations = retrieve_last_conversations(3)
    context = "\n".join(
        [f"User: {conv[0]}\nAgent: {conv[1]}" for conv in last_conversations]
    )
    intent_prompt = f"""
You are a cybersecurity system automation assistant. Here is the conversation history:

{context}

Now, decide what the following instruction is asking you to do.

Instruction: "{user_input}"

Respond with only one of the following:
- scan_ports
- ignore
- request_clarification

Only use these keywords exactly.
"""
    llm_response = ask_llm(intent_prompt)
    print(f"[DEBUG] LLM Response: {llm_response}")

    response_lower = llm_response.lower()
    if "scan_ports" in response_lower:
        return "scan_ports"
    elif "ignore" in response_lower:
        return "ignore"
    elif "request_clarification" in response_lower:
        return "request_clarification"
    else:
        return "unrecognized"


def summarize_report(report_text):
    """Asks the LLM to summarize the port scan report."""
    summary_prompt = f"""
You are a cybersecurity assistant. A port scan was run on the local system. Here's the result:

{report_text}

Please summarize this report and highlight any ports or services that might require security review.
"""
    return ask_llm(summary_prompt, max_tokens=512)


def main():
    print("Cybersecurity AI Agent is ready. Type your command:")
    while True:
        try:
            user_input = input("> ")
        except (KeyboardInterrupt, EOFError):
            print("\n👋 Agent exiting.")
            break

        if user_input.lower() in ["exit", "quit"]:
            print("👋 Agent exiting.")
            break

        intent = detect_intent(user_input)
        print(f"[Agent intent]: {intent}")

        if intent == "scan_ports":
            print("[*] Scanning ports...")
            nmap_output = run_nmap_scan()
            open_ports = parse_nmap_output(nmap_output)
            process_map = get_processes_by_port()
            report = generate_report(open_ports, process_map)

            print("\nPort Scan Report:\n", report)
            try:
                print("[*] Generating summary...")
                summary = summarize_report(report)
                print("\nSummary:\n", summary)
                store_conversation(user_input, summary)
            except Exception as e:
                print(f"[!] Failed to summarize report: {e}")

        elif intent == "ignore":
            print("No action required.")

        elif intent == "request_clarification":
            print("I need more context to understand the command.")

        else:
            print("Unrecognized action. Please try again.")


if __name__ == "__main__":
    main()
