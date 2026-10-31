import csv
import json
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, "src")
import rag  # noqa: E402

REFUSAL_PHRASES = [
    "don't know", "do not know", "not in the", "no information",
    "cannot find", "can't find", "not mentioned", "not covered",
]


def normalise(text):
    text = text.lower()
    for old, new in (("\u00ef", "i"), ("\u2019", "'"), ("-", " ")):
        text = text.replace(old, new)
    return text


def main():
    label = sys.argv[1] if len(sys.argv) > 1 else "run"
    questions_path = sys.argv[2] if len(sys.argv) > 2 else "eval/questions.json"
    questions = json.loads(Path(questions_path).read_text(encoding="utf-8"))

    answer_hits = 0
    source_hits = 0
    source_total = 0
    print(f"Running {len(questions)} questions (label: {label})\n")

    for q in questions:
        answer, metas = rag.ask(q["question"])
        norm = normalise(answer)

        if q.get("should_refuse"):
            answer_ok = any(p in norm for p in REFUSAL_PHRASES)
            source_ok = None
        else:
            missing = [k for k in q["keywords"] if k not in norm]
            answer_ok = not missing
            source_ok = any(
                q["lecture"] in m["source"] and m["page"] in q["pages"]
                for m in metas
            )
            source_total += 1
            source_hits += int(source_ok)

        answer_hits += int(answer_ok)
        src = "n/a" if source_ok is None else ("OK" if source_ok else "MISS")
        extra = "" if answer_ok or q.get("should_refuse") else f"  missing: {missing}"
        print(f"Q{q['id']:>2}  answer: {'PASS' if answer_ok else 'FAIL'}  source: {src}{extra}")

    total = len(questions)
    print(f"\nAnswer accuracy: {answer_hits}/{total} ({100 * answer_hits / total:.0f}%)")
    print(f"Source retrieval: {source_hits}/{source_total} ({100 * source_hits / source_total:.0f}%)")

    results_file = Path("eval/results.csv")
    new_file = not results_file.exists()
    with results_file.open("a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if new_file:
            writer.writerow(["date", "label", "answer_correct", "source_correct", "total"])
        writer.writerow([
            datetime.now().strftime("%Y-%m-%d %H:%M"), label,
            answer_hits, source_hits, total,
        ])


if __name__ == "__main__":
    main()
