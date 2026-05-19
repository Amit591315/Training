# Ollama Setup Notes

This file records how Ollama was installed and how to run it locally on Ubuntu 22.04.

## Install Method Used

The official installer script tried to download `.tgz` and failed with `404`, so a manual user-level install was used.

### 1. Download package

```bash
cd /tmp
curl -L --fail -C - --output ollama-linux-amd64.tar.zst \
  https://ollama.com/download/ollama-linux-amd64.tar.zst
```

### 2. Extract into user directory

```bash
mkdir -p "$HOME/.local/lib/ollama" "$HOME/.local/bin"
tar --zstd -xf /tmp/ollama-linux-amd64.tar.zst -C "$HOME/.local/lib/ollama"
ln -sf "$HOME/.local/lib/ollama/bin/ollama" "$HOME/.local/bin/ollama"
```

### 3. Add to PATH (persist)

```bash
grep -q 'export PATH="$HOME/.local/bin:$PATH"' "$HOME/.bashrc" || \
  echo 'export PATH="$HOME/.local/bin:$PATH"' >> "$HOME/.bashrc"
source "$HOME/.bashrc"
```

## Start Ollama Server

```bash
ollama serve
```

Server endpoint:

- `http://127.0.0.1:11434`

## Pull a Model

```bash
ollama pull llama3.2:3b
```

## Run the Model

```bash
ollama run llama3.2:3b "Reply with exactly: OLLAMA_OK"
```

## Useful Checks

```bash
ollama --version
ollama list
curl -s http://127.0.0.1:11434/api/tags
```

## Installed Models

| Model | Size | Notes |
|---|---|---|
| `llama3.2:3b` | 2.0 GB | Default model, faster, good for chat |
| `llama2:latest` | 3.8 GB | Larger 7B model, more capable |

Pull additional models with:

```bash
ollama pull <model-name>
```

Browse available models at https://ollama.com/library

---

## Python Chat Script (`ollama_chat.py`)

An interactive multi-turn chat script is included at `ollama_chat.py`.
It uses only Python standard library — no extra packages needed.

### Run

```bash
# Default model (llama3.2:3b)
python ollama_chat.py

# Choose a specific model
python ollama_chat.py --model llama2

# List locally installed models
python ollama_chat.py --list
```

### In-session commands

| Command | Effect |
|---|---|
| `/exit` or `/quit` | End the chat session |
| `/clear` | Wipe conversation history (fresh context) |
| `/model <name>` | Switch to a different model (clears history) |
| `/history` | Print the full conversation so far |

### Features

- Streaming output — tokens print as they arrive from the model
- Multi-turn memory — full conversation history is sent each turn so the model remembers context
- Auto-fallback — if the requested model is not installed, it uses the first available one
- No dependencies — uses only `urllib`, `json`, and `argparse` from the standard library
