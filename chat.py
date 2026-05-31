"""
Terminal Chat Client — Personal Digital Assistant

This is the chat interface for the RAG pipeline. It is provided
complete — you do not need to modify this file.

Usage:
    python chat.py
"""

import subprocess
import os

#Aqui quitamos la funcionn load_config_from_env porq creemos que tener todo centralizado en
#rag.py nos facilita mas las cosas
from rag import Assistant, load_config_from_env

from dotenv import load_dotenv

WELCOME = """
╔══════════════════════════════════════════════════════╗
║           Personal Digital Assistant                 ║
║                                                      ║
║  Ask me about emails, notes, SMS, and calendar.      ║
║  Type '/clear' to reset conversation history.        ║
║  Try: what's the address for Laura's surprise party? ║
║  Type '/exit' to leave.                              ║
╚══════════════════════════════════════════════════════╝
"""


def main():
    print("Initializing assistant...")
    config = load_config_from_env()
    assistant = Assistant.from_config(config)
    subprocess.call('cls' if os.name == 'nt' else 'clear')

    print(WELCOME)

    while True:
        try:
            question = input("You: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nGoodbye!")
            break

        if not question:
            continue

        if question.lower() == "/exit":
            print("Goodbye!")
            break

        if question.lower() == "/clear":
            assistant.clear_history()
            subprocess.call('cls' if os.name == 'nt' else 'clear')
            print("\nConversation history cleared.\n")
            continue

        response = assistant.ask(question)
        print(f"\nAssistant: {response}\n")


if __name__ == "__main__":
    load_dotenv()
    main()
