import requests

response = requests.post("http://localhost:11434/api/generate", json={
    "model": "llama2",
    "prompt": "What AI agent in one line",
    "stream": False,
})
print(response.json()["response"])
