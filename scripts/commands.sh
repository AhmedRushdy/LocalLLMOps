
# request to az
curl -s http://127.0.0.1:8080/generate/ollama \
  -H 'Content-Type: application/json' \
  -d '{"prompt":"What is LLMOps? Answer in two sentences.","max_tokens":60,"temperature":0}'