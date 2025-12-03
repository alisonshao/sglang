# SGLang Ollama Integration

Ollama API compatibility for SGLang, plus a Smart Router for intelligent routing between local and remote models.

## Features

1. **Ollama-compatible API** - Use Ollama CLI/library with SGLang backend
2. **Smart Router** - Route simple tasks locally, complex tasks to powerful remote models

---

## Quick Start: Ollama API

### 1. Start SGLang Server (GPU)

```bash
ssh user@gpu-server
git clone -b feature/ollama-api https://github.com/alisonshao/sglang.git
cd sglang && pip install -e "python[all]"

# Start with any HuggingFace model
python -m sglang.launch_server \
    --model <YOUR_MODEL> \
    --port 30001 \
    --host 0.0.0.0

# Example:
python -m sglang.launch_server \
    --model Qwen/Qwen2.5-1.5B-Instruct \
    --port 30001 \
    --host 0.0.0.0
```

### 2. Connect from Local Machine

```bash
# SSH tunnel if behind firewall
ssh -L 30001:localhost:30001 user@gpu-server -N &

# Use Ollama CLI with any model name
OLLAMA_HOST=http://localhost:30001 ollama list
OLLAMA_HOST=http://localhost:30001 ollama run "<MODEL_NAME>"
```

### 3. Ollama Python Library

```python
import ollama

client = ollama.Client(host='http://localhost:30001')
response = client.chat(
    model='<MODEL_NAME>',
    messages=[{'role': 'user', 'content': 'Hello!'}]
)
print(response['message']['content'])
```

---

## Smart Router

Routes requests between local Ollama and remote SGLang based on task complexity.

### Setup

**Terminal 1: Local Ollama**
```bash
ollama serve
ollama pull <LOCAL_MODEL>  # e.g., llama3.2, mistral, phi3
```

**Terminal 2: Remote SGLang (GPU)**
```bash
ssh user@gpu-server
python -m sglang.launch_server --model <REMOTE_MODEL> --port 30001 --host 0.0.0.0
```

**Terminal 3: Smart Router**
```bash
ssh -L 30001:localhost:30001 user@gpu-server -N &
python python/sglang/srt/entrypoints/ollama/smart_router.py
```

### Configuration

**All models are configurable** - use whatever models fit your needs:

```python
from sglang.srt.entrypoints.ollama.smart_router import SmartRouter

router = SmartRouter(
    # Local Ollama (fast, for simple tasks)
    local_host="http://localhost:11434",
    local_model="llama3.2",  # or: mistral, phi3, gemma2, etc.

    # Remote SGLang (powerful, for complex tasks)
    remote_host="http://localhost:30001",
    remote_model="Qwen/Qwen2.5-1.5B-Instruct",  # or: any HuggingFace model

    # LLM Judge (for ambiguous cases)
    judge_model="llama3.2",  # uses local model by default
    use_llm_judge=True,
)
```

### Routing Logic

**Stage 1: Rule-based (fast)**
| Condition | Route | Confidence |
|-----------|-------|------------|
| Greetings ("hello", "hi") | Local | 95% |
| Code keywords ("python", "debug") | Remote | 90% |
| Math ("calculate", "solve") | Remote | 90% |
| Long prompts (>500 chars) | Remote | 90% |

**Stage 2: LLM Judge** (when confidence < 70%)
- Asks local model: "Is this SIMPLE or COMPLEX?"

### Usage

```python
# Auto-routing
response = router.chat("Hello!", verbose=True)
# -> Routes to Local

response = router.chat("Write quicksort in Python", verbose=True)
# -> Routes to Remote

# Force routing
response = router.chat("question", force_local=True)
response = router.chat("question", force_remote=True)

# Streaming
for chunk in router.chat_stream("Tell me a story"):
    print(chunk['message']['content'], end='')
```

---

## Value

- **Ollama**: Simple CLI/API developers already know
- **SGLang**: High-performance inference
- **Smart Router**: Fast local for simple, powerful remote for complex
