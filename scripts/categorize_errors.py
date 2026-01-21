#!/usr/bin/env python3
"""
Edge Case Categorization Tool

Interactive tool to categorize EDGE CASES (systematic model failures) into types.
Categories loaded from config/edge-case-evaluation.json.

Requires: Run prepare_edge_cases.py first to create categorization_sample.json

Usage:
    python scripts/categorize_errors.py

Controls:
    1-N = Select category (numbered by config order)
    x = Skip (not a clear edge case / doesn't fit categories)
    q = Quit and save progress

Inputs:
    data/raw/edge_cases/categorization_sample.json - From prepare_edge_cases.py

Outputs:
    data/raw/edge_cases/<category>.csv - One CSV per category with columns:
        text, sentiment, num_models_failed, failed_models, notes
    data/raw/edge_cases/categorization_progress.json - Progress tracking
"""

import argparse
import csv
import json
import os
from datetime import datetime

# Import shared config loader
from edge_case_config import load_config, wilson_ci


def load_json(filepath):
    """Load JSON file."""
    with open(filepath, 'r') as f:
        return json.load(f)


def load_progress(progress_file):
    """Load categorization progress."""
    if os.path.exists(progress_file):
        with open(progress_file, 'r') as f:
            return json.load(f)
    return {
        'categorized': {},  # hash -> category or 'skipped'
        'started_at': datetime.now().isoformat(),
        'last_updated': None
    }


def save_progress(progress, progress_file):
    """Save categorization progress."""
    progress['last_updated'] = datetime.now().isoformat()
    with open(progress_file, 'w') as f:
        json.dump(progress, f, indent=2)


def clear_screen():
    """Clear terminal screen."""
    os.system('clear' if os.name != 'nt' else 'cls')


def truncate_text(text, max_lines=15, max_chars=1200):
    """Truncate text for display."""
    if len(text) > max_chars:
        text = text[:max_chars] + "..."
    lines = text.split('\n')
    if len(lines) > max_lines:
        lines = lines[:max_lines] + ['... (truncated)']
    return '\n'.join(lines)


def format_notes(error):
    """Format notes field for CSV output."""
    confidences = []
    for model in error['failed_models']:
        conf = error['confidences'].get(model, '?')
        pred = error['predictions'].get(model, '?')
        confidences.append(f"{model}:{pred}@{conf}")

    return f"Failed on: {', '.join(error['failed_models'])}. Predictions: {'; '.join(confidences)}"


def display_error(error, index, total, progress, categories, category_descriptions):
    """Display an error for categorization."""
    clear_screen()

    # Calculate stats
    categorized = len([v for v in progress['categorized'].values() if v != 'skipped'])
    skipped = len([v for v in progress['categorized'].values() if v == 'skipped'])
    remaining = total - len(progress['categorized'])

    # Category counts
    cat_counts = {cat: 0 for cat in categories.values()}
    for verdict in progress['categorized'].values():
        if verdict in cat_counts:
            cat_counts[verdict] += 1

    print("=" * 70)
    print(f"EDGE CASE CATEGORIZATION | {categorized} categorized, {skipped} skipped, {remaining} remaining")
    print("=" * 70)

    # Show current category counts
    counts_str = " | ".join([f"{cat[:3].upper()}:{count}" for cat, count in cat_counts.items()])
    print(f"Categories: {counts_str}")
    print()

    # Error metadata
    print(f"Domain: {error['domain']}")
    print(f"Ground Truth: {error['actual_sentiment']}")
    print(f"Models Failed ({error['num_models_failed']}): {', '.join(error['failed_models'][:5])}")
    if len(error['failed_models']) > 5:
        print(f"  ... and {len(error['failed_models']) - 5} more")
    print()

    # The text
    print("-" * 70)
    print("REVIEW TEXT:")
    print("-" * 70)
    print(truncate_text(error['text']))
    print("-" * 70)
    print()

    # Category options
    print("Select category:")
    print()
    for key, cat in categories.items():
        desc = category_descriptions.get(cat, '')
        print(f"  [{key}] {cat.upper()}")
        print(f"      {desc}")
        print()

    print("  [x] Skip (doesn't fit any category clearly)")
    print("  [q] Quit and save")
    print()


def export_csvs(all_errors, progress, edge_cases_dir, categories):
    """Export categorized errors to CSV files with full traceability."""
    # Build error lookup
    errors_by_hash = {e['hash']: e for e in all_errors}

    # Group by category
    by_category = {cat: [] for cat in categories.values()}

    for hash_id, category in progress['categorized'].items():
        if category == 'skipped':
            continue
        if category in by_category and hash_id in errors_by_hash:
            by_category[category].append(errors_by_hash[hash_id])

    # Write CSVs with extended columns for traceability
    for category, errors in by_category.items():
        filepath = os.path.join(edge_cases_dir, f'{category}.csv')

        with open(filepath, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            # Extended header with signal strength and failed models for audit trail
            writer.writerow(['text', 'sentiment', 'num_models_failed', 'failed_models', 'notes'])

            for error in errors:
                writer.writerow([
                    error['text'],
                    error['actual_sentiment'],
                    error['num_models_failed'],
                    ';'.join(error['failed_models']),
                    format_notes(error)
                ])

        print(f"  Exported {len(errors)} to {filepath}")

    return by_category


def print_summary(progress, categories, confidence_level=0.95):
    """Print categorization summary with confidence interval guidance."""
    print("\n" + "=" * 50)
    print("CATEGORIZATION SUMMARY")
    print("=" * 50)

    cat_counts = {cat: 0 for cat in categories.values()}
    skipped = 0

    for verdict in progress['categorized'].values():
        if verdict == 'skipped':
            skipped += 1
        elif verdict in cat_counts:
            cat_counts[verdict] += 1

    total_categorized = sum(cat_counts.values())

    print(f"Total categorized: {total_categorized}")
    print(f"Skipped: {skipped}")
    print()
    print(f"Category breakdown (with {confidence_level:.0%} CI margin of error):")
    for cat, count in cat_counts.items():
        # Use Wilson CI for better small-sample estimation
        _, _, moe = wilson_ci(count // 2, count, confidence_level) if count > 0 else (0, 1, 0.5)
        print(f"  {cat}: {count} samples (accuracy +/- {moe:.0%})")
    print()


def main():
    parser = argparse.ArgumentParser(description='Categorize edge case errors')
    parser.add_argument('--config', type=str, default=None,
                        help='Path to config file (default: config/edge-case-evaluation.json)')
    args = parser.parse_args()

    # Load config
    try:
        config = load_config(args.config)
        print(f"Config loaded: {config}")
    except FileNotFoundError as e:
        print(f"[ERROR] {e}")
        print("Create config/edge-case-evaluation.json first")
        return 1

    # Build category mappings from config
    categories = {str(i+1): cat.id for i, cat in enumerate(config.categories)}
    category_descriptions = {cat.id: cat.description for cat in config.categories}

    # Get paths from config
    edge_cases_dir = config.directories.edge_cases_dir
    input_file = os.path.join(edge_cases_dir, 'categorization_sample.json')
    progress_file = os.path.join(edge_cases_dir, 'categorization_progress.json')
    confidence_level = config.evaluation.confidence_level

    print("=" * 60)
    print("EDGE CASE CATEGORIZATION")
    print("=" * 60)
    print(f"Config version: {config.version}")
    print()

    if not os.path.exists(input_file):
        print(f"[ERROR] Sample file not found: {input_file}")
        print("Run 'python scripts/prepare_edge_cases.py' first.")
        return 1

    # Load data
    all_errors = load_json(input_file)
    progress = load_progress(progress_file)

    print(f"Loaded {len(all_errors)} filtered errors")
    print(f"Already categorized: {len(progress['categorized'])}")

    # Filter to uncategorized
    pending = [e for e in all_errors if e['hash'] not in progress['categorized']]

    if not pending:
        print("\nAll errors have been categorized!")
        print("\nExporting CSVs...")
        by_category = export_csvs(all_errors, progress, edge_cases_dir, categories)
        print_summary(progress, categories, confidence_level)
        return 0

    print(f"Remaining to categorize: {len(pending)}")
    print()
    print("Category guide:")
    for cat, desc in category_descriptions.items():
        print(f"  - {cat}: {desc}")
    print()

    valid_choices = list(categories.keys())
    input("Press Enter to start categorizing...")

    for i, error in enumerate(pending):
        display_error(error, i, len(all_errors), progress, categories, category_descriptions)

        while True:
            choice = input("Category: ").strip().lower()

            if choice in categories:
                progress['categorized'][error['hash']] = categories[choice]
                save_progress(progress, progress_file)
                break
            elif choice == 'x':
                progress['categorized'][error['hash']] = 'skipped'
                save_progress(progress, progress_file)
                break
            elif choice == 'q':
                save_progress(progress, progress_file)
                print(f"\nProgress saved. {len(progress['categorized'])} processed.")

                # Export what we have so far
                print("\nExporting CSVs with current progress...")
                by_category = export_csvs(all_errors, progress, edge_cases_dir, categories)
                print_summary(progress, categories, confidence_level)
                return 0
            else:
                print(f"Invalid choice. Enter {'-'.join(valid_choices)}, x, or q")

    # All done
    print("\nCategorization complete!")
    print("\nExporting CSVs...")
    by_category = export_csvs(all_errors, progress, edge_cases_dir, categories)
    print_summary(progress, categories, confidence_level)

    print("\nNext step: Run edge case evaluation")
    print("  ./scripts/evaluate_edge_cases.sh all")
    print()

    return 0


if __name__ == '__main__':
    exit(main())
