"""
Interactive chat with a local Ollama model.

Usage:
    python ollama_chat.py                   # uses default model (llama3.2:3b)
    python ollama_chat.py --model llama2    # pick a different model
    python ollama_chat.py --list            # list available models

Commands while chatting:
    /exit  or  /quit   — end the session
    /clear             — clear conversation history
    /model <name>      — switch model mid-session
    /history           — print full conversation so far
"""

import argparse
import json
import sys
from urllib.request import urlopen, Request
from urllib.error import URLError

OLLAMA_BASE = "http://127.0.0.1:11434"
DEFAULT_MODEL = "llama3.2:3b"


# ---------------------------------------------------------------------------
# API helpers
# ---------------------------------------------------------------------------

def api_get(path: str) -> dict:
    try:
        with urlopen(f"{OLLAMA_BASE}{path}", timeout=10) as r:
            return json.loads(r.read())
    except URLError:
        print("\nERROR: Could not reach Ollama server. Is it running?\n"
              "  Start it with:  ollama serve", file=sys.stderr)
        sys.exit(1)


def list_models() -> list[str]:
    data = api_get("/api/tags")
    return [m["name"] for m in data.get("models", [])]


def stream_chat(model: str, messages: list[dict]) -> str:
    """Send messages to /api/chat and stream tokens as they arrive."""
    payload = json.dumps({
        "model": model,
        "messages": messages,
        "stream": True,
    }).encode()

    req = Request(
        f"{OLLAMA_BASE}/api/chat",
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    full_reply = []
    try:
        with urlopen(req, timeout=120) as resp:
            for raw_line in resp:
                line = raw_line.strip()
                if not line:
                    continue
                chunk = json.loads(line)
                token = chunk.get("message", {}).get("content", "")
                if token:
                    print(token, end="", flush=True)
                    full_reply.append(token)
                if chunk.get("done"):
                    break
    except URLError as exc:
        print(f"\nERROR: {exc}", file=sys.stderr)
        sys.exit(1)

    print()  # newline after streaming ends
    return "".join(full_reply)


# ---------------------------------------------------------------------------
# Main chat loop
# ---------------------------------------------------------------------------

def chat_loop(model: str) -> None:
    history: list[dict] = []

    print(f"\n{'='*55}")
    print(f"  Ollama Chat  —  model: {model}")
    print(f"  Commands: /exit  /clear  /model <name>  /history")
    print(f"{'='*55}\n")

    while True:
        try:
            user_input = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nBye!")
            break

        if not user_input:
            continue

        # ---- built-in commands ----
        if user_input.lower() in ("/exit", "/quit"):
            print("Bye!")
            break

        if user_input.lower() == "/clear":
            history.clear()
            print("[conversation history cleared]\n")
            continue

        if user_input.lower().startswith("/model "):
            new_model = user_input.split(None, 1)[1].strip()
            available = list_models()
            if new_model not in available:
                print(f"[model '{new_model}' not found. available: {', '.join(available)}]\n")
            else:
                model = new_model
                history.clear()
                print(f"[switched to {model}, history cleared]\n")
            continue

        if user_input.lower() == "/history":
            if not history:
                print("[no history yet]\n")
            else:
                for msg in history:
                    role = msg["role"].upper()
                    print(f"  [{role}] {msg['content']}")
                print()
            continue

        # ---- normal message ----
        history.append({"role": "user", "content": user_input})
        print(f"\nAssistant: ", end="", flush=True)
        reply = stream_chat(model, history)
        history.append({"role": "assistant", "content": reply})
        print()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Interactive Ollama chat")
    parser.add_argument(
        "--model", default=DEFAULT_MODEL,
        help=f"Model to use (default: {DEFAULT_MODEL})",
    )
    parser.add_argument(
        "--list", action="store_true",
        help="List locally available models and exit",
    )
    args = parser.parse_args()

    if args.list:
        models = list_models()
        if models:
            print("Available models:")
            for m in models:
                prefix = "  * " if m == args.model else "    "
                print(f"{prefix}{m}")
        else:
            print("No models found. Run: ollama pull llama3.2:3b")
        return

    available = list_models()
    if args.model not in available:
        print(f"Model '{args.model}' is not available locally.")
        if available:
            print(f"Available models: {', '.join(available)}")
            print(f"Defaulting to:    {available[0]}\n")
            args.model = available[0]
        else:
            print("No models installed. Run: ollama pull llama3.2:3b")
            sys.exit(1)

    chat_loop(args.model)


if __name__ == "__main__":
    main()
