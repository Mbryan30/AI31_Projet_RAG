import os

from langchain_mistralai import ChatMistralAI

import time

debut = time.time()

if "MISTRAL_API_KEY" not in os.environ:
    os.environ["MISTRAL_API_KEY"] = 'ApQoCiUccY8izo03NtPIxVaCVkI5lCne'

llm = ChatMistralAI(
    model="mistral-large-latest",
    temperature=0,
    max_retries=2,
)

messages = [
    (
        "system",
        "You are a helpful assistant that translates English to French. Translate the user sentence.",
    ),
    ("human", "I love programming."),
]
ai_msg = llm.invoke(messages)
fin = time.time()
print(ai_msg)
print(f"Temps d'exécution : {fin - debut:.6f} secondes")
