from mistralai.client import Mistral
import os


with Mistral(
    api_key="ApQoCiUccY8izo03NtPIxVaCVkI5lCne",
) as mistral:

    res = mistral.chat.complete(model="mistral-large-latest", messages=[
        {
            "role": "user",
            "content": "Who is the best French painter? Answer in one short sentence.",
        },
    ], stream=False, response_format={
        "type": "text",
    })

    # Handle response
    print(res)

