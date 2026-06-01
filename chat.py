"""
Cliente de chat en terminal para el asistente digital.

Interfaz simple para probar el pipeline RAG desde consola.

Uso:
    python chat.py
"""

import subprocess
import os

# Dejamos la configuracion centralizada en rag.py para no duplicar logica.
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
