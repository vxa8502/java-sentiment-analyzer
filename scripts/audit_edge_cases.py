#!/usr/bin/env python3
"""
Edge Case Audit Tool
Presents error candidates one-by-one for human verification.

Usage:
    python scripts/audit_edge_cases.py [--resume]

Controls:
    1 = Label Error (model is RIGHT, ground truth is WRONG)
    2 = True Error (model is WRONG, ground truth is correct)
    3 = Ambiguous (genuinely hard to determine)
    s = Skip (come back later)
    q = Quit and save progress
"""

import json
import os
import sys
from datetime import datetime

AUDIT_SAMPLE_FILE = 'data/raw/edge_cases/audit_sample.json'
AUDIT_RESULTS_FILE = 'data/raw/edge_cases/audit_results.json'

def load_sample():
    with open(AUDIT_SAMPLE_FILE, 'r') as f:
        return json.load(f)

def load_results():
    if os.path.exists(AUDIT_RESULTS_FILE):
        with open(AUDIT_RESULTS_FILE, 'r') as f:
            return json.load(f)
    return {
        'completed': {},  # hash -> verdict
        'started_at': datetime.now().isoformat(),
        'last_updated': None
    }

def save_results(results):
    results['last_updated'] = datetime.now().isoformat()
    with open(AUDIT_RESULTS_FILE, 'w') as f:
        json.dump(results, f, indent=2)

def clear_screen():
    os.system('clear' if os.name != 'nt' else 'cls')

def truncate_text(text, max_lines=15, max_chars=1000):
    """Truncate text for display."""
    if len(text) > max_chars:
        text = text[:max_chars] + "..."
    lines = text.split('\n')
    if len(lines) > max_lines:
        lines = lines[:max_lines] + ['...']
    return '\n'.join(lines)

def display_error(error, index, total, results):
    clear_screen()

    completed = len(results['completed'])
    remaining = total - completed

    # Calculate current stats
    verdicts = list(results['completed'].values())
    label_errors = verdicts.count('label_error')
    true_errors = verdicts.count('true_error')
    ambiguous = verdicts.count('ambiguous')

    print("=" * 70)
    print(f"EDGE CASE AUDIT | Progress: {completed}/{total} ({remaining} remaining)")
    print(f"Stats: Label Errors={label_errors}, True Errors={true_errors}, Ambiguous={ambiguous}")
    print("=" * 70)
    print()

    # Error metadata
    print(f"Domain: {error['domain']}")
    print(f"Ground Truth Label: {error['actual_sentiment']}")
    print(f"Models that failed ({error['num_models_failed']}): {', '.join(error['failed_models'])}")
    print()

    # The text
    print("-" * 70)
    print("REVIEW TEXT:")
    print("-" * 70)
    print(truncate_text(error['text']))
    print("-" * 70)
    print()

    # Question
    print("Is the GROUND TRUTH LABEL correct?")
    print()
    print(f"  If '{error['actual_sentiment']}' is WRONG (model is right)  -> press 1 (Label Error)")
    print(f"  If '{error['actual_sentiment']}' is CORRECT (model failed) -> press 2 (True Error)")
    print(f"  If genuinely ambiguous/hard to tell                        -> press 3 (Ambiguous)")
    print()
    print("  [s] Skip   [q] Quit and save")
    print()

def main():
    sample = load_sample()
    results = load_results()

    print(f"Loaded {len(sample)} samples for audit")
    print(f"Already completed: {len(results['completed'])}")

    # Filter to uncompleted
    pending = [e for e in sample if e['hash'] not in results['completed']]

    if not pending:
        print("\nAll samples have been audited!")
        print_summary(results)
        return

    print(f"Remaining to audit: {len(pending)}")
    input("\nPress Enter to start auditing...")

    for i, error in enumerate(pending):
        display_error(error, i, len(sample), results)

        while True:
            choice = input("Your verdict: ").strip().lower()

            if choice == '1':
                results['completed'][error['hash']] = 'label_error'
                save_results(results)
                break
            elif choice == '2':
                results['completed'][error['hash']] = 'true_error'
                save_results(results)
                break
            elif choice == '3':
                results['completed'][error['hash']] = 'ambiguous'
                save_results(results)
                break
            elif choice == 's':
                break  # Skip, don't save
            elif choice == 'q':
                save_results(results)
                print(f"\nProgress saved. Completed {len(results['completed'])}/{len(sample)}")
                print_summary(results)
                return
            else:
                print("Invalid choice. Enter 1, 2, 3, s, or q")

    print("\nAudit complete!")
    print_summary(results)

def print_summary(results):
    verdicts = list(results['completed'].values())
    total = len(verdicts)

    if total == 0:
        print("No samples audited yet.")
        return

    label_errors = verdicts.count('label_error')
    true_errors = verdicts.count('true_error')
    ambiguous = verdicts.count('ambiguous')

    print("\n" + "=" * 50)
    print("AUDIT SUMMARY")
    print("=" * 50)
    print(f"Total audited: {total}")
    print(f"  Label Errors (model was right): {label_errors} ({100*label_errors/total:.1f}%)")
    print(f"  True Errors (model failed):     {true_errors} ({100*true_errors/total:.1f}%)")
    print(f"  Ambiguous:                      {ambiguous} ({100*ambiguous/total:.1f}%)")
    print()

    if label_errors > 0:
        print(f"Estimated label noise rate: {100*label_errors/total:.1f}%")
        print("Recommendation: Filter out likely label errors before categorizing.")

    print()

if __name__ == '__main__':
    main()
