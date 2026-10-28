"""
Score rectified articles against the 10 human-rectified gold examples.

Edits are computed as word-level diffs from the cleaned AI article:
    gold edits = diff(AI -> gold),  pred edits = diff(AI -> prediction)

Metrics:
    exact recall    gold edits reproduced exactly (same span, same replacement)
    located recall  gold edits whose span our edits touched (fix may differ in wording)
    precision       our edits that overlap some gold edit (the rest are unneeded changes)
    similarity      character-level similarity of prediction to gold

Usage:
    python eval.py                     # score rectified_articles/ against eval/gold/
    python eval.py --pred some_dir     # score another output folder
    python eval.py --baseline          # score the unedited (cleaned) AI text, i.e. the floor
    python eval.py -v                  # also print every missed / extra edit
    python eval.py --hints             # all 104 articles: is each annotated wrong text still present?

The --hints check is free (no LLM) and covers every article, not just the 10 gold ones. Hints are
incomplete, so 100% here is necessary for "all errors fixed" but does not prove it.
"""

import argparse
import difflib
import json
import re
import sys
from pathlib import Path

from rectification_system import clean_article, read_text, split_annotations

GOLD_DIR = Path("eval/gold")
TOKEN_RE = re.compile(r"\s+|\w+|[^\w\s]")


def tokenize(text: str) -> list[str]:
    return TOKEN_RE.findall(text.replace("\r\n", "\n"))


def diff_edits(src: list[str], dst: list[str]) -> list[tuple[int, int, str]]:
    """Edits as (start, end, replacement) over src token indices."""
    sm = difflib.SequenceMatcher(None, src, dst, autojunk=False)
    edits = []
    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        old, new = "".join(src[i1:i2]), "".join(dst[j1:j2])
        # Whitespace-only changes (e.g. stray double spaces in gold) aren't factual edits
        if tag != "equal" and (old.strip() or new.strip()):
            edits.append((i1, i2, new))
    return edits


def overlaps(a: tuple, b: tuple) -> bool:
    # Pure insertions have start == end; treat them as touching that position
    a1, a2 = a[0], max(a[1], a[0] + 1)
    b1, b2 = b[0], max(b[1], b[0] + 1)
    return a1 < b2 and b1 < a2


def score_article(ai: str, gold: str, pred: str) -> dict:
    ai_t = tokenize(ai)
    gold_e = diff_edits(ai_t, tokenize(gold))
    pred_e = diff_edits(ai_t, tokenize(pred))
    exact = [g for g in gold_e if g in pred_e]
    located = [g for g in gold_e if any(overlaps(g, p) for p in pred_e)]
    useful = [p for p in pred_e if any(overlaps(p, g) for g in gold_e)]
    span = lambda e: "".join(ai_t[e[0]:e[1]])
    return {
        "gold": len(gold_e), "pred": len(pred_e),
        "exact": len(exact), "located": len(located), "useful": len(useful),
        "similarity": difflib.SequenceMatcher(None, pred, gold, autojunk=False).ratio(),
        "missed": [(span(g), g[2]) for g in gold_e if g not in located],
        "extra": [(span(p), p[2]) for p in pred_e if p not in useful],
    }


def check_hints(pred_dir: str, mapping: dict, verbose: bool) -> None:
    """For every annotated error quoted verbatim in the AI article, check it is gone from the output."""
    fixed = still = unchecked = missing = 0
    for article_id, entry in mapping.items():
        pred_path = Path(pred_dir) / f"{article_id}.txt"
        if not pred_path.exists():
            missing += 1
            continue
        ai, hints = split_annotations(read_text(entry["ai_generated_file"]))
        pred = read_text(str(pred_path))
        for hint in hints:
            wrong = str(hint.get("error", "")).strip().strip("\"'")
            if not wrong or wrong not in ai:
                unchecked += 1  # descriptive hint, not a verbatim quote
            elif wrong in pred:
                still += 1
                if verbose:
                    print(f"  STILL  {article_id}: {wrong[:90]!r}")
            else:
                fixed += 1
    total = fixed + still
    print(f"Annotated errors removed: {fixed}/{total} ({100 * fixed / total:.1f}%)" if total else "No checkable hints.")
    print(f"Not checkable (descriptive hints): {unchecked}   Articles without output: {missing}")


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    parser = argparse.ArgumentParser(description="Score rectified articles against gold examples.")
    parser.add_argument("--pred", default="rectified_articles", help="folder with predicted articles")
    parser.add_argument("--baseline", action="store_true", help="score the cleaned AI text with no edits")
    parser.add_argument("-v", "--verbose", action="store_true", help="print missed and extra edits")
    parser.add_argument("--hints", action="store_true", help="check annotated errors are gone, all articles")
    args = parser.parse_args()

    with open("article_mapping.json", "r") as f:
        mapping = {a["article_id"]: a for a in json.load(f)}
    if args.hints:
        check_hints(args.pred, mapping, args.verbose)
        return

    totals = {"gold": 0, "pred": 0, "exact": 0, "located": 0, "useful": 0}
    sims = []
    print(f"{'article':<12} {'gold':>4} {'pred':>4} {'exact':>5} {'loc':>4} {'useful':>6} {'sim':>7}")
    for gold_path in sorted(GOLD_DIR.glob("*.txt")):
        article_id = gold_path.stem
        ai = clean_article(read_text(mapping[article_id]["ai_generated_file"]))
        pred_path = Path(args.pred) / gold_path.name
        if args.baseline:
            pred = ai
        elif pred_path.exists():
            pred = read_text(str(pred_path))
        else:
            print(f"{article_id:<12} missing prediction, skipped")
            continue
        s = score_article(ai, read_text(str(gold_path)), pred)
        for k in totals:
            totals[k] += s[k]
        sims.append(s["similarity"])
        print(f"{article_id:<12} {s['gold']:>4} {s['pred']:>4} {s['exact']:>5} {s['located']:>4} "
              f"{s['useful']:>6} {s['similarity']:>7.4f}")
        if args.verbose:
            for old, new in s["missed"]:
                print(f"    MISSED  {old!r} -> {new!r}")
            for old, new in s["extra"]:
                print(f"    EXTRA   {old!r} -> {new!r}")

    if not sims:
        print("No articles scored.")
        return
    pct = lambda a, b: f"{100 * a / b:.1f}%" if b else "n/a"
    print("-" * 48)
    print(f"Exact recall:   {pct(totals['exact'], totals['gold'])}  ({totals['exact']}/{totals['gold']})")
    print(f"Located recall: {pct(totals['located'], totals['gold'])}  ({totals['located']}/{totals['gold']})")
    print(f"Precision:      {pct(totals['useful'], totals['pred'])}  ({totals['useful']}/{totals['pred']})")
    print(f"Mean similarity to gold: {sum(sims) / len(sims):.4f}")


if __name__ == "__main__":
    main()
