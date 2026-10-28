"""
Article Rectification System

Pipeline per article (any failure falls back to the cleaned article, so an output is always produced):
  1. Clean    strip the leaked "Error Annotations" block, keep it as hints
  2. Detect   an LLM call returns surgical find/replace edits as JSON
  3. Validate each edit must be found in the article and be backed by the source
  4. Apply    edits are spliced in by Python; the LLM never rewrites the article
  5. Verify   a second LLM call undoes unsupported edits and adds fixes pass 1 missed
"""

from __future__ import annotations

import difflib
import hashlib
import json
import os
import re
import time
from pathlib import Path

from dotenv import load_dotenv
from litellm import completion

load_dotenv()

CACHE_DIR = Path(".cache/llm")  # set LLM_CACHE=0 to disable
LOG_DIR = Path(".cache/logs")
REASONING_EFFORT = os.getenv("LLM_REASONING_EFFORT", "medium")
MAX_FIND_CHARS = 600
MAX_DELETE_CHARS = 150  # longer deletions remove whole sentences, which human editors don't do

usage = {"calls": 0, "cached": 0, "prompt_tokens": 0, "completion_tokens": 0}


# ---------------------------------------------------------------- cleaning

# Seen as "**Error Annotations:**" and "## Error Annotations:"
ANNOTATION_RE = re.compile(r"^[ \t]*(?:\*\*|#+[ \t]*)?Error Annotations", re.MULTILINE | re.IGNORECASE)
_PAIR_RE = re.compile(r'"error"\s*:\s*"((?:[^"\\]|\\.)*)"\s*,\s*"correction"\s*:\s*"((?:[^"\\]|\\.)*)"', re.DOTALL)


def read_text(path: str) -> str:
    with open(path, "r", encoding="utf-8", newline="") as f:  # newline='' keeps CRLF as-is
        return f.read()


def write_text(path: str, text: str) -> None:
    with open(path, "w", encoding="utf-8", newline="") as f:
        f.write(text)


def split_annotations(text: str) -> tuple[str, list[dict]]:
    """Return (clean article body, annotation hints). Hints are untrusted and often incomplete."""
    match = ANNOTATION_RE.search(text)
    if not match:
        return text.rstrip(), []
    block = text[match.start():]
    raw = block[block.find("["):block.rfind("]") + 1]
    try:
        data = json.loads(raw)
        return text[:match.start()].rstrip(), [h for h in data if isinstance(h, dict)]
    except (json.JSONDecodeError, TypeError):
        # Malformed JSON: pull out error/correction pairs with a regex
        hints = [{"error": e.replace('\\"', '"'), "correction": c.replace('\\"', '"')} for e, c in _PAIR_RE.findall(raw)]
        return text[:match.start()].rstrip(), hints


def clean_article(text: str) -> str:
    return split_annotations(text)[0]


# ---------------------------------------------------------------- LLM client

def chat(messages: list[dict]) -> str:
    """One LLM call with an on-disk cache and retries with backoff."""
    key = json.dumps([os.getenv("LLM_MODEL_NAME"), REASONING_EFFORT, messages], ensure_ascii=False)
    path = CACHE_DIR / (hashlib.sha256(key.encode("utf-8")).hexdigest() + ".json")
    use_cache = os.getenv("LLM_CACHE", "1") != "0"
    if use_cache and path.exists():
        usage["cached"] += 1
        return json.loads(path.read_text(encoding="utf-8"))["content"]
    for attempt in range(4):
        try:
            response = completion(
                model=os.getenv("LLM_MODEL_NAME"), messages=messages,
                api_key=os.getenv("LLM_API_KEY"), api_base=os.getenv("LLM_API_BASE"),
                # Later retries use low effort: if reasoning used up the output cap, retrying the same way fails again
                reasoning_effort=REASONING_EFFORT if attempt < 2 else "low", temperature=0, timeout=240,
                max_tokens=16000,  # reasoning tokens count here; the default cap truncates to an empty reply
                drop_params=True,  # a provider that rejects a parameter drops it instead of failing the call
            )
            content = (response.choices[0].message.content or "").strip()
            if not content:
                raise ValueError("empty response")
            break
        except Exception as e:
            if attempt == 3:
                raise RuntimeError(f"LLM call failed after 4 attempts: {e}")
            time.sleep(2 ** attempt * 2)
    usage["calls"] += 1
    if getattr(response, "usage", None):
        usage["prompt_tokens"] += response.usage.prompt_tokens or 0
        usage["completion_tokens"] += response.usage.completion_tokens or 0
    if use_cache:
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({"content": content}, ensure_ascii=False), encoding="utf-8")
    return content


def extract_json(text: str):
    """Parse the JSON object in a model reply, tolerating code fences and chatter."""
    text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text.strip())
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        start, end = text.find("{"), text.rfind("}")
        if start == -1 or end <= start:
            raise ValueError("no JSON object in response")
        return json.loads(text[start:end + 1])


def chat_json(messages: list[dict]) -> dict:
    """chat() that must return a JSON object; one corrective retry on bad JSON."""
    reply = chat(messages)
    try:
        return extract_json(reply)
    except ValueError:  # includes json.JSONDecodeError
        return extract_json(chat(messages + [
            {"role": "assistant", "content": reply},
            {"role": "user", "content": "That was not valid JSON (check escaping of quotes inside strings). "
                                        "Reply with only the corrected JSON object."},
        ]))


# ---------------------------------------------------------------- patcher

_CHAR_CLASSES = {**dict.fromkeys('"“”', '["“”]'), **dict.fromkeys("'‘’", "['‘’]"), **dict.fromkeys("-–—", "[-–—]")}
_NUMBER_RE = re.compile(r"\d[\d,.]*")


def normalize(text: str) -> str:
    text = text.translate(str.maketrans({"“": '"', "”": '"', "‘": "'", "’": "'", "–": "-", "—": "-"}))
    return re.sub(r"\s+", " ", text).strip().lower()


def locate(article: str, find: str) -> list[tuple[int, int]]:
    """Spans of `find`: every exact occurrence, else matches ignoring quote/dash/space style, else one fuzzy match."""
    spans = [m.span() for m in re.finditer(re.escape(find), article)]
    if not spans:
        loose = "".join(r"\s+" if c.isspace() else _CHAR_CLASSES.get(c, re.escape(c)) for c in find.strip())
        spans = [m.span() for m in re.finditer(re.sub(r"(\\s\+)+", r"\\s+", loose), article)]
    if spans or len(find) < 8:
        return spans
    # Fuzzy: a near-identical window for a `find` the model mistyped (e.g. 'Rajeep' for 'Rajeev')
    n = len(find)
    good = sorted(((difflib.SequenceMatcher(None, find, article[i:i + n], autojunk=False).ratio(), i)
                   for i in range(len(article) - n + 1) if article[i].lower() == find[0].lower()), reverse=True)
    good = [(r, i) for r, i in good if r >= 0.9]
    # Must be unambiguous: no other good window outside the best one
    if not good or any(abs(i - good[0][1]) >= n for _, i in good[1:]):
        return []
    return [(good[0][1], good[0][1] + n)]


def quote_supported(quote: str, source_norm: str) -> bool:
    """The quote, or each piece of it the model elided with '...', must (nearly) appear in the source."""
    def found(fragment):
        m = difflib.SequenceMatcher(None, fragment, source_norm, autojunk=False).find_longest_match(
            0, len(fragment), 0, len(source_norm))
        return fragment in source_norm or m.size / len(fragment) >= 0.7
    fragments = [f.strip(" .,;:\"'") for f in re.split(r"\.\.\.|…", normalize(quote))]
    fragments = [f for f in fragments if len(f) >= 4]
    return bool(fragments) and all(found(f) for f in fragments)


def new_words_supported(find: str, replace: str, source_norm: str) -> bool:
    """Every word the edit introduces appears somewhere in the source."""
    old = set(re.findall(r"\w+", normalize(find)))
    new = [w for w in re.findall(r"\w+", normalize(replace)) if w not in old and len(w) > 2]
    return bool(new) and all(re.search(rf"\b{re.escape(w)}\b", source_norm) for w in new)


def numbers_supported(find: str, replace: str, source_digits: str) -> bool:
    """Every number introduced by `replace` must appear somewhere in the source."""
    old = {n.replace(",", "").rstrip(".") for n in _NUMBER_RE.findall(find)}
    new = {n.replace(",", "").rstrip(".") for n in _NUMBER_RE.findall(replace)}
    return all(n in old or n in source_digits for n in new)


def is_cosmetic(find: str, replace: str) -> bool:
    """True when the edit only changes punctuation, spacing, case or currency notation."""
    squash = lambda s: re.sub(r"[\W_]+|\brs\b|\binr\b", "", normalize(s))
    return squash(find) == squash(replace)


def validate_edits(article: str, source: str, edits: list) -> tuple[list[tuple[int, int, str]], list[str], list[dict]]:
    """Return (spans to apply as (start, end, replacement), rejection reasons, accepted edits).

    A fix whose `find` occurs several times is applied to every occurrence.
    """
    source_norm, source_digits = normalize(source), source.replace(",", "")
    accepted, rejected, kept = [], [], []
    for edit in (e for e in edits if isinstance(e, dict)):
        find, replace = str(edit.get("find") or ""), str(edit.get("replace") or "")
        spans = locate(article, find) if find.strip() else []
        if not find.strip() or find == replace:
            reason = "no-op"
        elif replace.strip() and is_cosmetic(find, replace):
            reason = "cosmetic only"
        elif len(find) > MAX_FIND_CHARS or (not replace.strip() and len(find) > MAX_DELETE_CHARS):
            reason = "too long"
        elif not spans:
            reason = "not found"
        elif len(spans) > 1 and len(find.strip()) < 4:
            reason = "too short to replace everywhere"
        elif replace.strip() and not (quote_supported(str(edit.get("source_quote") or ""), source_norm)
                                      or new_words_supported(find, replace, source_norm)):
            reason = "quote not in source"
        elif not numbers_supported(find, replace, source_digits):
            reason = "new number not in source"
        elif any(s[0] < e[1] and e[0] < s[1] for s in spans for e in accepted):
            reason = "overlaps another edit"
        else:
            accepted.extend((s[0], s[1], replace) for s in spans)
            kept.append(edit)
            continue
        rejected.append(f"{reason}: {find[:60]!r} -> {replace[:60]!r}")
    return accepted, rejected, kept


def apply_edits(article: str, spans: list[tuple[int, int, str]]) -> str:
    for start, end, replace in sorted(spans, reverse=True):
        before, after = article[:start], article[end:]
        # Deleting a clause: don't leave a double space or a space before punctuation
        if not replace and before.endswith(" ") and (after.startswith(" ") or after[:1] in ".,;:!?"):
            before = before[:-1]
        article = before + replace + after
    return article


# ---------------------------------------------------------------- pipeline

DETECT_SYSTEM = """You are a meticulous fact-checking copy editor working like a surgeon.
You receive a SOURCE (ground truth) and an ARTICLE written from it. The ARTICLE contains injected factual errors.
Your job: find every factual error and propose the SMALLEST possible edit that makes the ARTICLE agree with the SOURCE.

What counts as an error (check every sentence, every heading, every quote):
- numbers, amounts, currencies, percentages, statistics, scores, ages, counts, orders of magnitude
- dates, days of the week, years, times, durations, sequence (first/second/before/after)
- names of people, places, organisations, products, platforms, titles and roles, acronym expansions
- who did/said what (attribution), who it was addressed to, relationships between entities
- negations and flipped meaning ("banned" vs "not banned", "won" vs "reached the semi-final")
- sentiment and characterisation ("condemned" vs "celebrated", "narrowly beat" vs "hammered", "above-par" vs "below-par")
- figures inside headings and parentheses (e.g. bowling figures "4/20", scores, rankings) - check them too
- invented details that conflict with the SOURCE (an extra event, claim, place or date): delete just that clause

IMPORTANT - do NOT delete or change content only because the SOURCE does not mention it. The ARTICLE may contain
correct background (earlier records, history, context) that comes from elsewhere. Only edit a claim when the SOURCE
contradicts it, or when it is a specific detail clearly planted to mislead (e.g. a wrong event/venue/date attached to
something the SOURCE describes differently). When in doubt, leave the text unchanged.

How to edit (this matters as much as finding the error):
- Change ONLY the wrong words. Keep every correct word, the sentence structure and the tone exactly as they are.
- Use the SOURCE's own term for the correct fact (e.g. "Instagram", "300", "Lahore").
- To remove an invented clause, set "replace" to "" and include the surrounding punctuation in "find" so the sentence stays grammatical.
- If the error needs a small grammar adjustment (a/an, singular/plural, a connecting word), include it in the same edit.
- Never fix style, wording, grammar or spelling that is not a factual error. Never add information that was not in the ARTICLE unless it replaces a wrong fact.
- A different but compatible word is NOT an error: synonyms, "actor"/"actress", "Instagram Stories"/"social media stories",
  "smartphones"/"phones", a more specific term that the SOURCE also supports. Leave such wording exactly as it is.
- Check every number separately: in "4/20" both the wickets and the runs can be wrong; copy the exact figure from the SOURCE.
- Names: only fix a name when it refers to a different person/thing ("Sanjay Sood" vs "Sonu Sood", "Mallika" vs "Malaika").
  Transliteration variants ("Badnaam"/"Badnam"), short forms ("Salman-Arbaaz") and missing surnames are NOT errors.
- Prefer correcting the wrong words over deleting. Never delete a whole sentence: if a sentence states a wrong fact,
  replace just the wrong part with the SOURCE's fact (e.g. "winning the World Cup" -> "reaching the World Cup semi-final").
- Do not "correct" derived numbers (e.g. a target of 202 means the team scored 201 - both can be right).

The "find" string rules:
- Copy it CHARACTER-FOR-CHARACTER from the ARTICLE (same quotes, dashes, capitals, spacing).
- Keep it short, but long enough to occur exactly ONCE in the ARTICLE (add a neighbouring word if needed).
- If the same wrong fact appears several times, one edit is enough: it is applied to every occurrence.

HINTS may be provided: a list of suspected errors. Treat them as unverified leads: check each against the SOURCE,
ignore hints the SOURCE does not support, and derive the correction wording from the SOURCE (a hint's suggested correction
may be worded differently or be wrong). Hints are INCOMPLETE - always check the whole ARTICLE yourself.

Reply with ONLY a JSON object, no commentary:
{"edits": [{"find": "<exact text from ARTICLE>", "replace": "<corrected text>", "source_quote": "<short verbatim quote from SOURCE proving the fix, or empty for a deletion>", "reason": "<few words>"}]}
If there are no errors, reply {"edits": []}."""


VERIFY_SYSTEM = """You are the second, independent fact-checker in a surgical editing pipeline.
You receive a SOURCE (ground truth), an EDITED ARTICLE, and the list of EDITS a first editor already applied to it.

Do two things:

1. Review each applied edit against the WHOLE SOURCE (it may contain several articles). Reject it (by its number) only
   if the original text was actually correct, or the new text states something the SOURCE contradicts. Do not reject an
   edit just because it is worded differently from how you would word it. If unsure, keep it.

2. Read the EDITED ARTICLE sentence by sentence, headings and parentheses included, and find factual errors that are STILL
   there: wrong numbers, figures (e.g. "4/20"), dates, years, names, places, roles, attributions, negations, flipped
   sentiment or characterisation ("narrowly beat" vs "hammered", "above-par" vs "below-par"), or details that conflict with the SOURCE.
   Do NOT flag content just because the SOURCE does not mention it - only what the SOURCE contradicts.
   A different but compatible word (synonym, "actor"/"actress", "Instagram"/"social media") is NOT an error.
   Check every number separately: in "4/20" both the wickets and the runs can be wrong.
   Spelling/transliteration variants and short forms of names are NOT errors. Never delete a whole sentence;
   replace just the wrong part with the SOURCE's fact.
   For each remaining error give the SMALLEST possible edit, changing only the wrong words and using the SOURCE's own term.
   "find" must be copied character-for-character from the EDITED ARTICLE; one edit fixes every occurrence.

HINTS (unverified, possibly incomplete) list errors that were suspected in the original article; check any that still look unfixed.

Reply with ONLY a JSON object:
{"reject": [<edit numbers to undo>],
 "edits": [{"find": "<exact text from EDITED ARTICLE>", "replace": "<corrected text>", "source_quote": "<short verbatim quote from SOURCE, or empty for a deletion>", "reason": "<few words>"}]}
Use empty lists when there is nothing to reject or add."""


def ask(system: str, source: str, article_label: str, article: str, hints: list[dict], extra: str = "") -> dict:
    hint_lines = "\n".join(f"- suspected: {h['error']} | suggested: {h.get('correction', '')}"
                           for h in hints if h.get("error")) or "(none)"
    user = (f"SOURCE:\n<<<\n{source.strip()}\n>>>\n\n{article_label}:\n<<<\n{article}\n>>>\n\n"
            f"{extra}HINTS:\n{hint_lines}")
    result = chat_json([{"role": "system", "content": system}, {"role": "user", "content": user}])
    return result if isinstance(result, dict) else {}


def as_list(value) -> list:
    return value if isinstance(value, list) else []


def run(ai_generated_content: str, source_content: str, article_id: str | None = None) -> str:
    """Rectify an AI-generated article against its source. Never raises."""
    article, hints = split_annotations(ai_generated_content)
    log = {}
    try:
        # Pass 1: detect
        log["proposed"] = as_list(ask(DETECT_SYSTEM, source_content, "ARTICLE", article, hints).get("edits"))
        spans, log["rejected_pass1"], kept = validate_edits(article, source_content, log["proposed"])
        result = apply_edits(article, spans)
    except Exception as e:
        print(f"  ! fallback to cleaned article ({e})")
        log["error"] = str(e)
        result = None

    if result is not None:
        try:
            # Pass 2: verify the applied edits and catch what pass 1 missed
            listing = "\n".join(f"{i}. {e.get('find')!r} -> {e.get('replace')!r}" for i, e in enumerate(kept, 1))
            reply = ask(VERIFY_SYSTEM, source_content, "EDITED ARTICLE", result, hints,
                        f"EDITS ALREADY APPLIED:\n{listing or '(none)'}\n\n")
            undo = {int(n) - 1 for n in as_list(reply.get("reject"))
                    if str(n).strip().isdigit() and 1 <= int(n) <= len(kept)}
            log["undone"] = [kept[i] for i in sorted(undo)]
            if undo:
                spans, _, kept = validate_edits(article, source_content,
                                                [e for i, e in enumerate(kept) if i not in undo])
                result = apply_edits(article, spans)
            spans, log["rejected_pass2"], log["added"] = validate_edits(result, source_content,
                                                                        as_list(reply.get("edits")))
            result = apply_edits(result, spans)
        except Exception as e:
            # Keep the pass-1 result if verification fails
            print(f"  ! verify pass skipped ({e})")
            log["verify_error"] = str(e)

    if article_id:
        LOG_DIR.mkdir(parents=True, exist_ok=True)
        (LOG_DIR / f"{article_id}.json").write_text(json.dumps(log, ensure_ascii=False, indent=2), encoding="utf-8")
    return article if result is None else result
