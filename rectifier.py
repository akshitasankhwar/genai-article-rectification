import json
import argparse
import os
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from rectification_system import clean_article, read_text, run, usage, write_text

# Windows consoles default to cp1252, which can't print the ✓ / ✗ markers
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

# Articles processed in parallel (each one makes its own LLM calls)
WORKERS = int(os.getenv("RECTIFIER_WORKERS", "4"))

def get_article_mapping(article_id: str):
    # Load article mapping to get file paths
    with open('article_mapping.json', 'r') as f:
        articles = json.load(f)

    # Find the article by ID
    article_data = next((a for a in articles if a['article_id'] == article_id), None)
    if not article_data:
        raise ValueError(f"Article {article_id} not found in mapping")

    return article_data

def get_ai_generated_article(article_id: str):
    # Read the AI-generated article
    _mapping = get_article_mapping(article_id)
    return read_text(_mapping['ai_generated_file'])

def get_source_article(article_id: str):
    return read_text(get_article_mapping(article_id)['source_file'])

def save_rectified_article(article_id: str, rectified_content: str):
    mapping = get_article_mapping(article_id)
    fpath = mapping['rectified_file']

    # Ensure output directory exists
    output_path = Path(fpath)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    write_text(fpath, rectified_content)

def rectify_article(article_id: str):
    """
    Rectify an AI-generated article.

    Args:
        article_id: ID of the article (e.g., 'article_001')

    Returns:
        str: The rectified article content
    """

    ai_generated_content = get_ai_generated_article(article_id)

    # PLUG YOUR CUSTOM RECTIFIER HERE
    rectified_content = run(ai_generated_content, get_source_article(article_id), article_id)
    ###################################

    save_rectified_article(article_id, rectified_content)

    print(f"✓ Rectified {article_id}")
    return rectified_content


def safe_rectify(article_id: str):
    """rectify_article() that always leaves an output file, even if something unexpected fails."""
    try:
        rectify_article(article_id)
        return True
    except Exception as e:
        print(f"✗ Error processing {article_id}: {str(e)}")
        try:
            save_rectified_article(article_id, clean_article(get_ai_generated_article(article_id)))
            print(f"  wrote cleaned original for {article_id}")
        except Exception as e2:
            print(f"  could not write fallback for {article_id}: {e2}")
        return False


def process(articles: list):
    """Rectify articles in parallel; returns the number that succeeded."""
    ids = [a['article_id'] for a in articles]
    ok = 0
    with ThreadPoolExecutor(max_workers=max(1, WORKERS)) as pool:
        futures = [pool.submit(safe_rectify, article_id) for article_id in ids]
        for done, future in enumerate(as_completed(futures), 1):
            ok += future.result()
            print(f"  progress: {done}/{len(ids)}")
    return ok


def test_rectifier(count: int):
    """
    Test the rectification system on a subset of articles.

    Args:
        count: Number of articles to test (default: 16)
    """
    # Load article mapping
    with open('article_mapping.json', 'r') as f:
        articles = json.load(f)

    # Test on first 'count' articles
    process(articles[:count])


def rectify_all():
    """
    Generate rectified articles for all articles in the mapping (104).
    """
    # Load article mapping
    with open('article_mapping.json', 'r') as f:
        articles = json.load(f)

    total = len(articles)
    ok = process(articles)

    print(f"\n{'='*50}")
    print(f"Completed! Processed {total} articles ({ok} without errors).")
    print(f"{'='*50}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Rectify AI-generated articles by fixing errors and inaccuracies."
    )
    parser.add_argument(
        'command',
        choices=['test', 'rectify-all'],
        help='Command to execute: "test" to process first 16 articles, "rectify-all" to process all 104 articles'
    )
    parser.add_argument(
        '--count',
        type=int,
        default=16,
        help='Number of articles to test (only applicable for "test" command, default: 16)'
    )

    args = parser.parse_args()

    if args.command == 'test':
        print(f"Testing rectification system on first {args.count} articles...")
        test_rectifier(count=args.count)
    elif args.command == 'rectify-all':
        print("Processing all articles...")
        rectify_all()

    print(f"LLM usage: {usage['calls']} calls ({usage['cached']} cached), "
          f"{usage['prompt_tokens']} prompt + {usage['completion_tokens']} completion tokens")
