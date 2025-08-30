#!/usr/bin/env python3
"""
Batch process questions and save answers to an Excel file.

This script demonstrates RAG usage by running multiple questions against
the vector store and saving the results to an Excel file for analysis.
"""
import sys
import argparse
import pandas as pd
from pathlib import Path
from tqdm import tqdm

# Add project root to path
PROJ_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJ_ROOT))

# Import from refactored modules
from core.config import refresh_openai_client
from core.rag_processor import process_rag_query, format_rag_response


def process_questions(questions, index_dir, top_k=5):
    """Process a list of questions and return their answers."""
    results = []

    # Ensure we have a fresh token
    refresh_openai_client()

    # Process each question
    for question in tqdm(questions, desc="Processing questions"):
        try:
            # This replaces the old query_cli function
            rag_result = process_rag_query(
                question=question,
                top_k=top_k,
                index_dir=index_dir
            )

            # Format the response
            answer = format_rag_response(rag_result)

            # Save both the raw result and the formatted answer
            results.append({
                "question": question,
                "answer": answer,
                "sources": ", ".join([d.get("guide-name", "Unknown") for d in rag_result.get("contexts", [])])
            })

        except Exception as e:
            print(f"Error processing question '{question}': {e}")
            results.append({
                "question": question,
                "answer": f"ERROR: {e}",
                "sources": ""
            })

    return results


def main():
    parser = argparse.ArgumentParser(description="Process questions in batch mode and save to Excel")
    parser.add_argument("--index", type=str, default="vector_store/vector_store_all",
                        help="Path to vector store index directory")
    parser.add_argument("--output", type=str, default="question_answers.xlsx",
                        help="Output Excel file path")
    parser.add_argument("--questions", type=str, help="Path to text file with questions (one per line)")
    args = parser.parse_args()

    # Resolve index path
    from core.config.paths import project_path
    index_dir = project_path(args.index)

    if not Path(index_dir).exists():
        sys.exit(f"❌ Error: Index directory not found: {index_dir}")

    # Get questions from file or use defaults
    if args.questions and Path(args.questions).exists():
        with open(args.questions, 'r', encoding='utf-8') as f:
            questions = [line.strip() for line in f if line.strip()]
    else:
        # Default sample questions
        questions = [
            "How do I reset my password",
            "how do I create an event",
            "How I change my password",
            "how to I add bidding rules to my event",
            "why are my are my suppliers not able to see my event in the event table view?",
            "how do I create a new user",
            "how do I change my timeline",
            "how do I move my suppliers timeline",
            "what bidding rules should I add to my event",
            "How do I define the cost formula for my event",
            "How many suppliers are in my event",
            "How many suppliers have not responded to xx",
            "how do I create a report that shoes me per item the number of offers received",
            "Tell me what my event deadline should be",
            "Tell me which evaluators have not submitted their evaluations",
            "how many suppliers have submitted offers and who?",
            "Suggest the test for my supplier invitation message",
            "How do I write a scenario with minimum spend awarded per supplier",
            "what are fact sheets",
            "Fact Sheet Data into Multiple Choice Item Fields/ how do I populate a multiple choice field from a fact sheet",
            "How do I configure discounts in my event",
            "How do I manage vendor/supplier name changes",
            "How do I standardize supplier entities",
            "What is the limit for multiple choice fields",
            "I’m looking to write a rule to only award a non-incumbent supplier if the total benefit vs not introducing that supplier exceeds a threshold (e.g. $500K). Decision needs to be across the total scenario, not an item by item benefi",
            "what does this xxx infeasibility message mean",
            "how do I create approval chains",
            "How do I enable bids in currencies different from the event currency",
            "I would like to create a scenario in CSO where a supplier is still awarded a part of the package. Is there a rule to overrule the fact that a package should be awarded completely?",
            "As some suppliers have previously filled out this questionnaire, we are exploring the possibility of pre-filling their answers. The pre-filled answers should be editable if needed.",
            "example of an invitation tags",
        ]

    print(f"📝 Processing {len(questions)} questions...")
    results = process_questions(questions, index_dir)

    # Save to Excel
    df = pd.DataFrame(results)
    output_path = Path(args.output)
    df.to_excel(output_path, index=False)
    print(f"✅ Results saved to {output_path}")


if __name__ == "__main__":
    main()
