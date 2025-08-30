#!/usr/bin/env python3
"""
Script to test conversation history functionality.

This script tests how conversation history is maintained and utilized
in follow-up questions, ensuring context is preserved across interactions.
"""

import sys
import time
from pathlib import Path
from typing import List, Dict, Any
import logging
import argparse

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from core.rag.conversation import build_conversation_history, estimate_tokens
from cli.chat_cli_interface import StreamingCallback
from scripts.test_cli_streaming import MockStreamingProcessor, MockRAGCallback

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

def test_conversation_history():
    """Test conversation history maintenance and follow-up questions."""
    print("\n=== Testing Conversation History ===\n")

    # Initial history
    history = []

    # Create a mock processor
    processor = MockStreamingProcessor(
        retrieval_delay=0.2,  # Fast retrieval for testing
        generation_delay=0.05  # Fast generation for testing
    )

    # Define a conversation flow with initial question and follow-ups
    conversation = [
        "What features does the product have?",
        "How do I set it up?",
        "What about troubleshooting?",
        "Can you provide more details about the API?"
    ]

    # Process each question in sequence, maintaining history
    for i, question in enumerate(conversation):
        print(f"\n[Turn {i+1}] User: {question}\n")
        print("Assistant:")

        # Create callback for this turn
        callback = MockRAGCallback()

        # Process the query with history context
        result = processor.process_query(question, callback)
        answer = result["answer"]

        # Print a newline after generation
        print()

        # Update history
        history.append({"role": "user", "content": question})
        history.append({"role": "assistant", "content": answer})

        # Show history stats
        token_count = sum(estimate_tokens(msg["content"]) for msg in history)
        print(f"\nHistory size: {len(history)} messages, ~{token_count} tokens")

        # Test history trimming
        max_tokens = 100  # Artificially low for testing
        trimmed = build_conversation_history(history, max_tokens)
        print(f"After trimming to {max_tokens} tokens: {len(trimmed)} messages")

        # Small delay between turns
        time.sleep(1)

    return history

def test_max_history_tokens():
    """Test history trimming with different token limits."""
    print("\n=== Testing History Token Limits ===\n")

    # Create a long history
    history = []
    for i in range(20):
        user_msg = f"This is test question {i} with some additional words to increase the token count."
        assistant_msg = f"This is a detailed answer to question {i} with sufficient length to simulate a real response. It contains multiple sentences to ensure it has a reasonable token count for testing purposes."

        history.append({"role": "user", "content": user_msg})
        history.append({"role": "assistant", "content": assistant_msg})

    # Test different token limits
    limits = [50, 100, 200, 500, 1000]

    total_tokens = sum(estimate_tokens(msg["content"]) for msg in history)
    print(f"Full history: {len(history)} messages, ~{total_tokens} tokens\n")

    for limit in limits:
        trimmed = build_conversation_history(history, limit)
        remaining_tokens = sum(estimate_tokens(msg["content"]) for msg in trimmed)
        print(f"Limit {limit} tokens: {len(trimmed)} messages, ~{remaining_tokens} tokens")

    return history

def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Test conversation functionality")
    parser.add_argument("--test", choices=["history", "tokens", "all"], default="all",
                      help="Which test to run: 'history', 'tokens', or 'all'")

    args = parser.parse_args()

    if args.test in ["history", "all"]:
        test_conversation_history()

    if args.test in ["tokens", "all"]:
        test_max_history_tokens()

if __name__ == "__main__":
    main()
