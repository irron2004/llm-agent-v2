"""Convert test questions from TypeScript to JSONL format.

Usage:
    python scripts/golden_set/convert_questions.py
    python scripts/golden_set/convert_questions.py --output data/golden_set/queries.jsonl
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

# Project root
ROOT = Path(__file__).resolve().parents[2]


def parse_typescript_questions(ts_path: Path) -> list[dict]:
    """Parse TypeScript test questions file.

    Args:
        ts_path: Path to test-questions.ts file.

    Returns:
        List of question dictionaries.
    """
    content = ts_path.read_text(encoding="utf-8")

    # Extract the array content between TEST_QUESTIONS: TestQuestion[] = [ ... ];
    match = re.search(
        r"TEST_QUESTIONS:\s*TestQuestion\[\]\s*=\s*\[(.*?)\];",
        content,
        re.DOTALL,
    )
    if not match:
        raise ValueError("Could not find TEST_QUESTIONS array in file")

    array_content = match.group(1)

    # Parse each object in the array
    questions = []
    # Match each object block { ... }
    object_pattern = re.compile(r"\{([^{}]+)\}", re.DOTALL)

    for obj_match in object_pattern.finditer(array_content):
        obj_str = obj_match.group(1)

        # Extract fields
        question = {}

        # id
        id_match = re.search(r'id:\s*["\']([^"\']+)["\']', obj_str)
        if id_match:
            question["id"] = id_match.group(1)

        # question text
        q_match = re.search(r'question:\s*["\'](.+?)["\'](?:,|\n)', obj_str, re.DOTALL)
        if q_match:
            question["question"] = q_match.group(1).strip()

        # category
        cat_match = re.search(r'category:\s*["\']([^"\']+)["\']', obj_str)
        if cat_match:
            question["category"] = cat_match.group(1)

        # difficulty
        diff_match = re.search(r'difficulty:\s*["\']([^"\']+)["\']', obj_str)
        if diff_match:
            question["difficulty"] = diff_match.group(1)

        if "id" in question and "question" in question:
            questions.append(question)

    return questions


def save_jsonl(questions: list[dict], output_path: Path) -> None:
    """Save questions to JSONL file.

    Args:
        questions: List of question dictionaries.
        output_path: Output file path.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        for q in questions:
            f.write(json.dumps(q, ensure_ascii=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert TypeScript test questions to JSONL"
    )
    parser.add_argument(
        "--input",
        type=str,
        default="frontend/src/features/retrieval-test/data/test-questions.ts",
        help="Input TypeScript file",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data/golden_set/queries.jsonl",
        help="Output JSONL file",
    )

    args = parser.parse_args()

    ts_path = ROOT / args.input
    output_path = ROOT / args.output

    if not ts_path.exists():
        print(f"ERROR: Input file not found: {ts_path}")
        sys.exit(1)

    print(f"Reading questions from {ts_path}...")
    questions = parse_typescript_questions(ts_path)
    print(f"Found {len(questions)} questions")

    print(f"Saving to {output_path}...")
    save_jsonl(questions, output_path)

    print("\nQuestions saved:")
    for q in questions:
        print(f"  {q['id']}: {q.get('category', 'N/A')} / {q.get('difficulty', 'N/A')}")
        print(f"       {q['question'][:60]}...")

    print(f"\nDone! Saved {len(questions)} questions to {output_path}")


if __name__ == "__main__":
    main()
