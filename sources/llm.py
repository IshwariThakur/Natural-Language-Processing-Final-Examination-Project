import re
from ollama import chat

MODEL = "qwen3:8b"          
DEFAULT_NUM_CTX = 3072      


def ask_llm(prompt, num_ctx=DEFAULT_NUM_CTX):
    response = chat(
        model=MODEL,
        messages=[{"role": "user", "content": prompt + "\n\n/no_think"}],
        options={"num_ctx": num_ctx, "num_predict": 2048},
    )
    text = response["message"]["content"]
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
    return text