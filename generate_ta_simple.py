"""
generate_ta_simple.py

Generates ta-simple.md (Tamil-medium simplified English) for every docs/en.md
in the phases directory. Resumable: skips files that already have ta-simple.md.

Usage:
    python3 generate_ta_simple.py
    python3 generate_ta_simple.py --dry-run       # show what would be processed
    python3 generate_ta_simple.py --limit 5       # process only first N files
    python3 generate_ta_simple.py --phase 01      # only process phase 01-*
"""

import os
import sys
import time
import argparse
import anthropic
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

SYSTEM_PROMPT = """You are an expert Tamil-medium teacher explaining AI/ML/CS concepts to students who learned in Tamil medium schools.

Your job: take an English lesson and rewrite it in SIMPLE English that a Tamil medium student can understand.

Rules:
1. Use very simple English words. Avoid complex academic English.
2. Add Tamil words in parentheses for key technical terms (e.g., "vector (வெக்டர் - எண் பட்டியல்)")
3. Use everyday Tamil analogies and examples (shops, rice, auto rickshaw, cricket, etc.)
4. Keep ALL code blocks exactly as-is — never change code.
5. Keep ALL math formulas — just explain them in simple words before/after.
6. Replace complex English phrases with short simple ones:
   - "demonstrates" → "shows"
   - "implementation" → "how to build"
   - "conceptually" → "in simple idea"
   - "parameterize" → "control with a setting"
7. Add a "Key Terms" table at the end with: English | Tamil word | One-line simple meaning
8. Structure: same sections as original, but each section starts with a 1-sentence Tamil-style plain summary.
9. DO NOT translate to pure Tamil — write in simple English with Tamil hints.
10. Keep it educational and complete — don't skip any concept from the original.

Output format: Valid Markdown. Start with a note: "<!-- Tamil-medium simplified version of en.md -->"
"""

def translate_file(client: anthropic.Anthropic, en_path: Path) -> str:
    content = en_path.read_text(encoding="utf-8")

    message = client.messages.create(
        model="claude-haiku-4-5-20251001",
        max_tokens=8192,
        system=SYSTEM_PROMPT,
        messages=[
            {
                "role": "user",
                "content": f"Here is the lesson to simplify:\n\n{content}",
            }
        ],
    )
    return message.content[0].text


def find_all_en_files(base: Path) -> list[Path]:
    return sorted(base.rglob("docs/en.md"))


def main():
    parser = argparse.ArgumentParser(description="Generate ta-simple.md for each en.md")
    parser.add_argument("--dry-run", action="store_true", help="Show files to process without generating")
    parser.add_argument("--limit", type=int, default=None, help="Max number of files to process")
    parser.add_argument("--phase", type=str, default=None, help="Filter by phase prefix e.g. '01'")
    args = parser.parse_args()

    base = Path(__file__).parent / "phases"
    all_files = find_all_en_files(base)

    if args.phase:
        all_files = [f for f in all_files if f"/{args.phase}-" in str(f)]

    # Only files that don't yet have ta-simple.md
    pending = [f for f in all_files if not (f.parent / "ta-simple.md").exists()]

    if args.limit:
        pending = pending[: args.limit]

    total_all = len(all_files)
    total_done = total_all - len([f for f in all_files if not (f.parent / "ta-simple.md").exists()])
    total_pending = len(pending)

    print(f"\n📚 Total lessons     : {total_all}")
    print(f"✅ Already done      : {total_done}")
    print(f"⏳ To generate       : {total_pending}")

    if args.dry_run:
        print("\n-- DRY RUN: files that would be processed --")
        for f in pending:
            print(f"  {f.relative_to(base.parent)}")
        return

    if total_pending == 0:
        print("\nAll ta-simple.md files already exist. Nothing to do.")
        return

    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        print("ERROR: ANTHROPIC_API_KEY not found in environment or .env file")
        sys.exit(1)

    client = anthropic.Anthropic(api_key=api_key)

    print("\nStarting generation...\n")
    errors = []

    for i, en_path in enumerate(pending, 1):
        ta_path = en_path.parent / "ta-simple.md"
        rel = en_path.relative_to(base.parent)
        print(f"[{i:3d}/{total_pending}] {rel} ...", end=" ", flush=True)

        try:
            result = translate_file(client, en_path)
            ta_path.write_text(result, encoding="utf-8")
            print("✓")
        except anthropic.RateLimitError:
            print("⚠ rate limit — waiting 60s")
            time.sleep(60)
            try:
                result = translate_file(client, en_path)
                ta_path.write_text(result, encoding="utf-8")
                print("  ✓ (retry)")
            except Exception as e2:
                print(f"  ✗ FAILED: {e2}")
                errors.append((str(rel), str(e2)))
        except Exception as e:
            print(f"✗ ERROR: {e}")
            errors.append((str(rel), str(e)))

        # Small delay to avoid hitting rate limits
        if i < total_pending:
            time.sleep(0.5)

    print(f"\n{'='*60}")
    print(f"Done. Generated: {total_pending - len(errors)}/{total_pending}")
    if errors:
        print(f"Failed ({len(errors)}):")
        for path, err in errors:
            print(f"  {path}: {err}")


if __name__ == "__main__":
    main()
