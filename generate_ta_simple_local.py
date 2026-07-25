"""
generate_ta_simple_local.py

Generates ta-simple.md for every docs/en.md without API calls.
Uses heuristic simplification + rules.

Usage:
    python3 generate_ta_simple_local.py
    python3 generate_ta_simple_local.py --limit 5
    python3 generate_ta_simple_local.py --phase 01
"""

import re
import argparse
from pathlib import Path


# Common complex → simple replacements
SIMPLIFICATIONS = {
    r"\bdemonstrates\b": "shows",
    r"\bimplementation\b": "how to build",
    r"\bconceptually\b": "in simple idea",
    r"\bparameterize\b": "control with settings",
    r"\bnonetheless\b": "but",
    r"\bfurthermore\b": "also",
    r"\bconsequently\b": "so",
    r"\boutstanding\b": "very good",
    r"\brather\b": "quite",
    r"\bsubstantial\b": "large",
    r"\bmathematical notation\b": "math symbols",
    r"\btradeoff\b": "choice",
    r"\bcomputation\b": "calculation",
    r"\badditional\b": "extra",
    r"\bitself\b": "itself",
    r"\bfundamental\b": "basic",
    r"\bdiverse\b": "many different",
    r"\btypically\b": "usually",
    r"\brobustness\b": "strength",
    r"\befficiency\b": "speed",
    r"\bscalability\b": "grows big",
    r"\boverhead\b": "extra work",
    r"\bsignificant\b": "big",
    r"\bintricacies\b": "small details",
    r"\bcaveat\b": "warning",
    r"\bmeticulously\b": "carefully",
    r"\bintuition\b": "simple idea",
}

# Tamil technical terms to inject
TAMIL_TERMS = {
    "vector": "(வெக்டர் — எண் பட்டியல்)",
    "matrix": "(மேட்ரிக்ஸ் — எண் மேஜை)",
    "scalar": "(scalar — ஒரு எண்)",
    "dimension": "(பரிமாணம் — கோணம்)",
    "algorithm": "(அல்கோரிதம் — வழிமுறை)",
    "parameter": "(பாரமீட்டர் — கட்டுப்பாடு)",
    "function": "(சார்பு — உள்ளீட்டுக்கு வெளியீடு)",
    "iteration": "(மறுயுதல் — மீண்டும் செய்தல்)",
    "optimization": "(உகந்தாக்கம் — சிறந்ததாக்குதல்)",
    "convergence": "(ஒருங்கிணைப்பு — ஒரு பதிலில் சேர்ந்துவருதல்)",
    "gradient": "(சாய்வு — மாற்றத்தின் दिशा)",
    "regression": "(பின்னடைவு — எண் தொடர்பு கண்டுபிடித்தல்)",
    "classification": "(வகைப்படுத்தல் — பிரிவுகளாக தொகுத்தல்)",
    "feature": "(பண்பு — தகவல் நெடுவரிசை)",
    "epoch": "(யுக் — ஒரு முழு சுழற்சி)",
    "batch": "(பத்தி — தொகுப்பு)",
    "neural network": "(வலைப்பின்னல் — மூளை போன்ற எண் அமைப்பு)",
    "deep learning": "(ஆழ் கற்றல் — பல்லாய கொட்ட பயிற்சி)",
    "tensor": "(டென்சர் — பல வடிவ எண் வரிசை)",
    "embedding": "(embedding — பொருளை எண்களாக மாற்றுதல்)",
    "backpropagation": "(பின்னாக பரப்புதல் — தவறிலிருந்து கற்றுக்கொள்ளுதல்)",
    "loss function": "(சேதத் சூத்திரம் — எவ்வளவு தவறு என்பதை அளக்கும் சூத்திரம்)",
}


def simplify_text(text: str) -> str:
    """Apply simplification rules to text."""

    # 1. Replace complex words with simple ones (case-insensitive)
    for complex_word, simple_word in SIMPLIFICATIONS.items():
        text = re.sub(complex_word, simple_word, text, flags=re.IGNORECASE)

    # 2. Add Tamil hints to technical terms (first occurrence per paragraph)
    for english, tamil in TAMIL_TERMS.items():
        # Replace first occurrence in each paragraph
        paragraphs = text.split("\n\n")
        for i, para in enumerate(paragraphs):
            if english.lower() in para.lower() and tamil not in para:
                # Find and replace (first occurrence only in this paragraph)
                pattern = re.compile(re.escape(english), re.IGNORECASE)
                paragraphs[i] = pattern.sub(f"{english} {tamil}", para, count=1)
        text = "\n\n".join(paragraphs)

    # 3. Simplify passive voice where obvious
    text = re.sub(r"is shown", "shows", text, flags=re.IGNORECASE)
    text = re.sub(r"is used", "helps", text, flags=re.IGNORECASE)
    text = re.sub(r"can be", "is", text, flags=re.IGNORECASE)

    # 4. Break up very long sentences (split on semicolons)
    text = re.sub(r";\s+", ". ", text)

    return text


def add_tamil_header(text: str) -> str:
    """Add Tamil-medium note at top."""
    header = "<!-- Tamil-medium simplified version of en.md -->\n\n"
    header += "> **For Tamil-medium students:** This version uses simple English and Tamil hints to explain AI concepts.\n\n"
    return header + text


def add_key_terms_table(text: str) -> str:
    """Add Key Terms table at the end if not present."""
    if "| English |" in text or "| Key" in text or "## Key" in text:
        return text  # Already has key terms

    terms_table = """
## Key Terms (Tamil-medium)

| English | Tamil | Simple meaning |
|---------|-------|---|
| vector | வெக்டர் | list of numbers = a point |
| matrix | மேட்ரிக்ஸ் | number table that changes things |
| dot product | டாட் புராடக்ட் | measure how similar two lists are |
| algorithm | வழிமுறை | step-by-step way to solve |
| parameter | பாரமீட்டர் | setting that controls output |
| gradient | சாய்வு | direction of biggest change |
| optimization | சிறந்ததாக்குதல் | make it better and better |
| neural network | வலைப்பின்னல் | number machine like a brain |
"""
    return text + terms_table


def simplify_file(en_path: Path) -> str:
    """Read en.md, simplify it, return ta-simple content."""
    content = en_path.read_text(encoding="utf-8")

    # Apply simplifications
    simplified = simplify_text(content)
    simplified = add_tamil_header(simplified)
    simplified = add_key_terms_table(simplified)

    return simplified


def find_all_en_files(base: Path) -> list[Path]:
    return sorted(base.rglob("docs/en.md"))


def main():
    parser = argparse.ArgumentParser(description="Generate ta-simple.md files locally (no API)")
    parser.add_argument("--limit", type=int, default=None, help="Max files to process")
    parser.add_argument("--phase", type=str, default=None, help="Filter by phase (e.g., '01')")
    parser.add_argument("--force", action="store_true", help="Overwrite existing ta-simple.md")
    args = parser.parse_args()

    base = Path(__file__).parent / "phases"
    all_files = find_all_en_files(base)

    if args.phase:
        all_files = [f for f in all_files if f"/{args.phase}-" in str(f)]

    # Filter out already-done unless --force
    if not args.force:
        pending = [f for f in all_files if not (f.parent / "ta-simple.md").exists()]
    else:
        pending = all_files

    if args.limit:
        pending = pending[: args.limit]

    total_all = len(all_files)
    total_done = total_all - len([f for f in all_files if not (f.parent / "ta-simple.md").exists()])
    total_pending = len(pending)

    print(f"\n📚 Total lessons    : {total_all}")
    print(f"✅ Already done     : {total_done}")
    print(f"⏳ To process       : {total_pending}\n")

    if total_pending == 0:
        print("All files done. Use --force to regenerate.")
        return

    errors = []
    for i, en_path in enumerate(pending, 1):
        ta_path = en_path.parent / "ta-simple.md"
        rel = en_path.relative_to(base.parent)
        print(f"[{i:3d}/{total_pending}] {rel} ...", end=" ", flush=True)

        try:
            simplified = simplify_file(en_path)
            ta_path.write_text(simplified, encoding="utf-8")
            print("✓")
        except Exception as e:
            print(f"✗ {e}")
            errors.append((str(rel), str(e)))

    print(f"\n{'='*60}")
    print(f"Done. Generated: {total_pending - len(errors)}/{total_pending}")
    if errors:
        print(f"Errors ({len(errors)}):")
        for path, err in errors:
            print(f"  {path}: {err}")


if __name__ == "__main__":
    main()
