"""
Smart Router: Automatically routes requests between local Ollama and remote SGLang.

Routes complex tasks (code, reasoning, long prompts) to powerful remote models,
and simple tasks to local models for faster response.

Usage:
    from sglang.srt.entrypoints.ollama.smart_router import SmartRouter

    router = SmartRouter(
        local_host="http://localhost:11434",
        remote_host="http://sglang-server:30001",
    )
    response = router.chat("Hello!")
"""

import re
from typing import Optional

import ollama


class SmartRouter:
    """Routes requests between local Ollama and remote SGLang based on task complexity."""

    def __init__(
        self,
        local_host: str = "http://localhost:11434",
        remote_host: str = "http://localhost:30001",
        local_model: str = "llama3.2",
        remote_model: str = "Qwen/Qwen2.5-1.5B-Instruct",
        token_threshold: int = 500,
    ):
        """
        Initialize the smart router.

        Args:
            local_host: URL of local Ollama server
            remote_host: URL of remote SGLang server
            local_model: Model name for local Ollama
            remote_model: Model name for remote SGLang
            token_threshold: Character count threshold for routing to remote
        """
        self.local_client = ollama.Client(host=local_host)
        self.remote_client = ollama.Client(host=remote_host)
        self.local_model = local_model
        self.remote_model = remote_model
        self.token_threshold = token_threshold

        # Keywords that indicate complex tasks
        self.code_keywords = [
            "code", "program", "function", "debug", "implement", "algorithm",
            "script", "python", "javascript", "java", "c++", "rust", "golang",
            "refactor", "optimize", "bug", "error", "exception"
        ]
        self.reasoning_keywords = [
            "analyze", "reason", "explain why", "compare", "evaluate",
            "critique", "pros and cons", "advantages", "disadvantages",
            "trade-off", "implications", "consequences"
        ]
        self.math_keywords = [
            "calculate", "solve", "equation", "formula", "proof", "theorem",
            "integral", "derivative", "matrix", "vector", "probability"
        ]

    def should_use_remote(self, prompt: str) -> tuple[bool, str]:
        """
        Determine if the prompt should be routed to remote SGLang.

        Args:
            prompt: User's input prompt

        Returns:
            Tuple of (should_use_remote, reason)
        """
        prompt_lower = prompt.lower()

        # Rule 1: Long prompts -> remote
        if len(prompt) > self.token_threshold:
            return True, f"Long prompt (>{self.token_threshold} chars)"

        # Rule 2: Code-related tasks -> remote
        if any(kw in prompt_lower for kw in self.code_keywords):
            return True, "Code-related task"

        # Rule 3: Reasoning/analysis tasks -> remote
        if any(kw in prompt_lower for kw in self.reasoning_keywords):
            return True, "Reasoning/analysis task"

        # Rule 4: Math/science -> remote
        if any(kw in prompt_lower for kw in self.math_keywords):
            return True, "Math/science task"

        # Rule 5: Contains code blocks -> remote
        if "```" in prompt or re.search(r"def |class |function |import |from ", prompt):
            return True, "Contains code"

        # Rule 6: Multi-step instructions -> remote
        if prompt.count("\n") > 3 or prompt.count(". ") > 5:
            return True, "Multi-step task"

        # Default: use local
        return False, "Simple task"

    def chat(
        self,
        prompt: str,
        messages: Optional[list] = None,
        verbose: bool = False,
        force_local: bool = False,
        force_remote: bool = False,
    ) -> dict:
        """
        Route the request and get response.

        Args:
            prompt: User's input (used if messages is None)
            messages: Full message history (overrides prompt if provided)
            verbose: Print routing decision
            force_local: Force use of local model
            force_remote: Force use of remote model

        Returns:
            Response dict with 'content', 'model', 'location', 'reason' keys
        """
        # Build messages
        if messages is None:
            messages = [{"role": "user", "content": prompt}]
            check_prompt = prompt
        else:
            # Use the last user message for routing decision
            check_prompt = ""
            for msg in reversed(messages):
                if msg.get("role") == "user":
                    check_prompt = msg.get("content", "")
                    break

        # Determine routing
        if force_remote:
            use_remote, reason = True, "Forced remote"
        elif force_local:
            use_remote, reason = False, "Forced local"
        else:
            use_remote, reason = self.should_use_remote(check_prompt)

        if use_remote:
            client = self.remote_client
            model = self.remote_model
            location = "Remote SGLang"
        else:
            client = self.local_client
            model = self.local_model
            location = "Local Ollama"

        if verbose:
            print(f"[Router] {location} | {reason} | Model: {model}")

        try:
            response = client.chat(model=model, messages=messages)
            return {
                "content": response["message"]["content"],
                "model": model,
                "location": location,
                "reason": reason,
            }
        except Exception as e:
            # Fallback to the other option
            if verbose:
                print(f"[Router] {location} failed: {e}, falling back...")

            fallback_client = self.remote_client if not use_remote else self.local_client
            fallback_model = self.remote_model if not use_remote else self.local_model
            fallback_location = "Remote SGLang" if not use_remote else "Local Ollama"

            response = fallback_client.chat(model=fallback_model, messages=messages)
            return {
                "content": response["message"]["content"],
                "model": fallback_model,
                "location": fallback_location,
                "reason": f"Fallback from {location}",
            }

    def chat_stream(
        self,
        prompt: str,
        messages: Optional[list] = None,
        verbose: bool = False,
        force_local: bool = False,
        force_remote: bool = False,
    ):
        """
        Route the request and stream response.

        Yields:
            Response chunks
        """
        if messages is None:
            messages = [{"role": "user", "content": prompt}]
            check_prompt = prompt
        else:
            check_prompt = ""
            for msg in reversed(messages):
                if msg.get("role") == "user":
                    check_prompt = msg.get("content", "")
                    break

        if force_remote:
            use_remote, reason = True, "Forced remote"
        elif force_local:
            use_remote, reason = False, "Forced local"
        else:
            use_remote, reason = self.should_use_remote(check_prompt)

        if use_remote:
            client = self.remote_client
            model = self.remote_model
            location = "Remote SGLang"
        else:
            client = self.local_client
            model = self.local_model
            location = "Local Ollama"

        if verbose:
            print(f"[Router] {location} | {reason} | Model: {model}")

        for chunk in client.chat(model=model, messages=messages, stream=True):
            yield chunk


def main():
    """Interactive demo of the smart router."""
    print("=" * 50)
    print("Smart Router: Local Ollama <-> Remote SGLang")
    print("=" * 50)
    print("\nRouting rules:")
    print("  - Long prompts (>500 chars) -> Remote")
    print("  - Code/programming tasks -> Remote")
    print("  - Reasoning/analysis -> Remote")
    print("  - Math/science -> Remote")
    print("  - Simple chat -> Local")
    print("\nType 'quit' to exit\n")

    router = SmartRouter(
        local_host="http://localhost:11434",
        remote_host="http://localhost:30001",
        local_model="llama3.2",
        remote_model="Qwen/Qwen2.5-1.5B-Instruct",
    )

    messages = []
    while True:
        try:
            user_input = input("You: ").strip()
            if user_input.lower() in ["quit", "exit", "q"]:
                print("Goodbye!")
                break
            if not user_input:
                continue

            messages.append({"role": "user", "content": user_input})
            result = router.chat(prompt=user_input, messages=messages, verbose=True)
            print(f"\nAssistant: {result['content']}\n")
            messages.append({"role": "assistant", "content": result["content"]})

        except KeyboardInterrupt:
            print("\nGoodbye!")
            break
        except Exception as e:
            print(f"Error: {e}\n")


if __name__ == "__main__":
    main()
