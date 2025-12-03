"""
Smart Router: Automatically routes requests between local Ollama and remote SGLang.

Routes complex tasks (code, reasoning, long prompts) to powerful remote models,
and simple tasks to local models for faster response.

Features:
- Hardcoded rules for fast classification
- LLM-based routing for ambiguous cases
- Automatic fallback between backends

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

    # Classification prompt for LLM judge
    CLASSIFICATION_PROMPT = """Classify this user request into one category. Reply with ONLY the category name.

Categories:
- SIMPLE: Greetings, small talk, simple questions, translations, definitions
- COMPLEX: Code, math, analysis, reasoning, multi-step tasks, long explanations

User request: "{prompt}"

Category:"""

    def __init__(
        self,
        local_host: str = "http://localhost:11434",
        remote_host: str = "http://localhost:30001",
        local_model: str = "llama3.2",
        remote_model: str = "Qwen/Qwen2.5-1.5B-Instruct",
        judge_model: Optional[str] = None,
        judge_host: Optional[str] = None,
        token_threshold: int = 500,
        use_llm_judge: bool = True,
    ):
        """
        Initialize the smart router.

        Args:
            local_host: URL of local Ollama server
            remote_host: URL of remote SGLang server
            local_model: Model name for local Ollama
            remote_model: Model name for remote SGLang
            judge_model: Model for LLM-based classification (default: same as local_model)
            judge_host: Host for judge model (default: same as local_host)
            token_threshold: Character count threshold for routing to remote
            use_llm_judge: Whether to use LLM for ambiguous cases
        """
        self.local_client = ollama.Client(host=local_host)
        self.remote_client = ollama.Client(host=remote_host)
        self.local_model = local_model
        self.remote_model = remote_model
        self.token_threshold = token_threshold
        self.use_llm_judge = use_llm_judge

        # Judge model configuration
        self.judge_model = judge_model or local_model
        self.judge_host = judge_host or local_host
        self.judge_client = ollama.Client(host=self.judge_host)

        # Keywords that indicate complex tasks (high confidence)
        self.code_keywords = [
            "code", "program", "function", "debug", "implement", "algorithm",
            "script", "python", "javascript", "java", "c++", "rust", "golang",
            "refactor", "optimize", "bug", "error", "exception", "compile"
        ]
        self.reasoning_keywords = [
            "analyze", "reason", "explain why", "compare", "evaluate",
            "critique", "pros and cons", "advantages", "disadvantages",
            "trade-off", "implications", "consequences", "argue"
        ]
        self.math_keywords = [
            "calculate", "solve", "equation", "formula", "proof", "theorem",
            "integral", "derivative", "matrix", "vector", "probability"
        ]

        # Keywords that indicate simple tasks (high confidence)
        self.simple_keywords = [
            "hello", "hi", "hey", "thanks", "thank you", "bye", "goodbye",
            "how are you", "what's up", "good morning", "good night",
            "what is your name", "who are you"
        ]

    def _classify_with_rules(self, prompt: str) -> tuple[Optional[bool], str, float]:
        """
        Classify using hardcoded rules.

        Returns:
            Tuple of (use_remote, reason, confidence)
            use_remote is None if uncertain
        """
        prompt_lower = prompt.lower().strip()

        # High confidence: Simple greetings
        if any(kw in prompt_lower for kw in self.simple_keywords):
            return False, "Simple greeting/chat", 0.95

        # High confidence: Long prompts
        if len(prompt) > self.token_threshold:
            return True, f"Long prompt (>{self.token_threshold} chars)", 0.9

        # High confidence: Code-related tasks
        if any(kw in prompt_lower for kw in self.code_keywords):
            return True, "Code-related task", 0.9

        # High confidence: Contains code blocks
        if "```" in prompt or re.search(r"def |class |function |import |from ", prompt):
            return True, "Contains code", 0.95

        # High confidence: Reasoning/analysis tasks
        if any(kw in prompt_lower for kw in self.reasoning_keywords):
            return True, "Reasoning/analysis task", 0.85

        # High confidence: Math/science
        if any(kw in prompt_lower for kw in self.math_keywords):
            return True, "Math/science task", 0.9

        # Medium confidence: Multi-step instructions
        if prompt.count("\n") > 3 or prompt.count(". ") > 5:
            return True, "Multi-step task", 0.7

        # Low confidence: Short simple question
        if len(prompt) < 50 and "?" in prompt and prompt.count(" ") < 10:
            return False, "Short simple question", 0.6

        # Uncertain - needs LLM judge
        return None, "Uncertain", 0.5

    def _classify_with_llm(self, prompt: str, verbose: bool = False) -> tuple[bool, str]:
        """
        Use LLM to classify the prompt.

        Returns:
            Tuple of (use_remote, reason)
        """
        try:
            classification_prompt = self.CLASSIFICATION_PROMPT.format(
                prompt=prompt[:500]  # Limit prompt length for classification
            )

            response = self.judge_client.chat(
                model=self.judge_model,
                messages=[{"role": "user", "content": classification_prompt}],
                options={"temperature": 0, "num_predict": 10}  # Deterministic, short response
            )

            result = response["message"]["content"].strip().upper()

            if verbose:
                print(f"[Router] LLM Judge says: {result}")

            if "COMPLEX" in result:
                return True, "LLM classified as complex"
            else:
                return False, "LLM classified as simple"

        except Exception as e:
            if verbose:
                print(f"[Router] LLM Judge failed: {e}, defaulting to local")
            return False, "LLM judge failed, defaulting to simple"

    def should_use_remote(
        self, prompt: str, verbose: bool = False
    ) -> tuple[bool, str]:
        """
        Determine if the prompt should be routed to remote SGLang.

        Uses a two-stage approach:
        1. Fast hardcoded rules for high-confidence cases
        2. LLM classification for ambiguous cases

        Args:
            prompt: User's input prompt
            verbose: Print debug information

        Returns:
            Tuple of (should_use_remote, reason)
        """
        # Stage 1: Rule-based classification
        use_remote, reason, confidence = self._classify_with_rules(prompt)

        if verbose:
            print(f"[Router] Rule-based: {reason} (confidence: {confidence:.0%})")

        # High confidence decision
        if use_remote is not None and confidence >= 0.7:
            return use_remote, reason

        # Stage 2: LLM classification for uncertain cases
        if self.use_llm_judge and (use_remote is None or confidence < 0.7):
            if verbose:
                print("[Router] Using LLM judge for uncertain case...")
            return self._classify_with_llm(prompt, verbose)

        # Fallback to rule-based decision or default
        if use_remote is not None:
            return use_remote, reason
        return False, "Default to local"

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
            use_remote, reason = self.should_use_remote(check_prompt, verbose)

        if use_remote:
            client = self.remote_client
            model = self.remote_model
            location = "Remote SGLang"
        else:
            client = self.local_client
            model = self.local_model
            location = "Local Ollama"

        if verbose:
            print(f"[Router] -> {location} | {reason} | Model: {model}")

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
            use_remote, reason = self.should_use_remote(check_prompt, verbose)

        if use_remote:
            client = self.remote_client
            model = self.remote_model
            location = "Remote SGLang"
        else:
            client = self.local_client
            model = self.local_model
            location = "Local Ollama"

        if verbose:
            print(f"[Router] -> {location} | {reason} | Model: {model}")

        for chunk in client.chat(model=model, messages=messages, stream=True):
            yield chunk


def main():
    """Interactive demo of the smart router."""
    print("=" * 60)
    print("Smart Router: Local Ollama <-> Remote SGLang")
    print("=" * 60)
    print("\nRouting strategy:")
    print("  1. Fast rule-based classification (keywords, length)")
    print("  2. LLM judge for ambiguous cases")
    print("\nRoutes to Remote SGLang:")
    print("  - Code/programming tasks")
    print("  - Reasoning/analysis")
    print("  - Math/science")
    print("  - Long/multi-step tasks")
    print("\nRoutes to Local Ollama:")
    print("  - Simple greetings")
    print("  - Short questions")
    print("  - Basic chat")
    print("\nType 'quit' to exit\n")

    router = SmartRouter(
        local_host="http://localhost:11434",
        remote_host="http://localhost:30001",
        local_model="llama3.2",
        remote_model="Qwen/Qwen2.5-1.5B-Instruct",
        use_llm_judge=True,
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
