#!/usr/bin/env python3
"""
Prepare Edge Cases for Categorization

Loads error CSVs from generate_edge_cases.sh, deduplicates, filters to
systematic failures (4+ models), and creates a stratified sample.

Usage:
    python scripts/prepare_edge_cases.py

Outputs:
    data/raw/edge_cases/categorization_sample.json - Stratified sample for categorization
    data/raw/edge_cases/categorization_stats.json  - Statistics about the sample
"""

import argparse
import csv
import hashlib
import json
import os
import random
from collections import defaultdict
from datetime import datetime

from edge_case_config import load_config


def load_and_deduplicate(candidates_dir, algorithms):
    """Load all error CSVs and deduplicate across models."""
    all_errors = {}

    for filename in os.listdir(candidates_dir):
        if not filename.endswith('_errors.csv'):
            continue

        # Parse algorithm and domain from filename
        base = filename.replace('_errors.csv', '')
        algo, domain = None, None
        for a in algorithms:
            if base.startswith(a + '_'):
                algo = a
                domain = base[len(a) + 1:]
                break

        if not algo:
            print(f"  [WARN] Could not parse: {filename}")
            continue

        model_name = f"{algo}-{domain}"
        filepath = os.path.join(candidates_dir, filename)

        count = 0
        with open(filepath, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                text = row.get('text', '').strip()
                if not text:
                    continue

                text_hash = hashlib.md5(text.encode()).hexdigest()

                if text_hash not in all_errors:
                    all_errors[text_hash] = {
                        'hash': text_hash,
                        'text': text,
                        'actual_sentiment': row.get('actual_sentiment', '').upper(),
                        'failed_models': set(),
                        'predictions': {},
                        'confidences': {},
                        'domains': set()
                    }

                all_errors[text_hash]['failed_models'].add(model_name)
                all_errors[text_hash]['predictions'][model_name] = row.get('predicted_sentiment', '').lower()
                all_errors[text_hash]['confidences'][model_name] = row.get('confidence', '')
                all_errors[text_hash]['domains'].add(domain)
                count += 1

        print(f"  {model_name}: {count} errors")

    # Convert sets to lists and compute metadata
    for error in all_errors.values():
        error['failed_models'] = sorted(list(error['failed_models']))
        error['domains'] = sorted(list(error['domains']))
        error['num_models_failed'] = len(error['failed_models'])

        # Primary domain
        domain_counts = defaultdict(int)
        for model in error['failed_models']:
            d = model.split('-')[1]
            domain_counts[d] += 1
        error['domain'] = max(domain_counts, key=domain_counts.get)

    return list(all_errors.values())


def create_stratified_sample(edge_cases, target_size, random_seed):
    """Create stratified sample by domain and error direction."""
    random.seed(random_seed)

    # Group by domain x direction
    strata = defaultdict(list)
    for error in edge_cases:
        direction = 'pos_to_neg' if error['actual_sentiment'] == 'POSITIVE' else 'neg_to_pos'
        strata[(error['domain'], direction)].append(error)

    print(f"\n  Strata ({len(strata)}):")
    for key, errors in sorted(strata.items()):
        print(f"    {key}: {len(errors)}")

    # Proportional allocation
    total = len(edge_cases)
    sample = []

    for key, errors in strata.items():
        n = max(1, round(len(errors) / total * target_size))
        n = min(n, len(errors))
        sample.extend(random.sample(errors, n))

    random.shuffle(sample)
    return sample


def main():
    parser = argparse.ArgumentParser(description='Prepare edge cases for categorization')
    parser.add_argument('--config', type=str, default=None)
    args = parser.parse_args()

    try:
        config = load_config(args.config)
        print(f"Config: {config}")
    except FileNotFoundError as e:
        print(f"[ERROR] {e}")
        return 1

    candidates_dir = config.directories.candidates_dir
    output_dir = config.directories.edge_cases_dir
    min_models = config.min_models_for_edge_case
    target_size = config.categorization_sample_size
    random_seed = config.sampling.random_seed

    print("=" * 60)
    print("PREPARE EDGE CASES")
    print("=" * 60)
    print(f"Config version: {config.version}")
    print(f"Edge case threshold: {min_models}+ models failed")
    print(f"Target sample size: {target_size}")
    print()

    # Step 1: Load and deduplicate
    print("Step 1: Loading and deduplicating errors...")
    if not os.path.exists(candidates_dir):
        print(f"[ERROR] Not found: {candidates_dir}")
        print("Run './scripts/generate_edge_cases.sh all' first")
        return 1

    all_errors = load_and_deduplicate(candidates_dir, config.algorithms)
    print(f"\n  Total unique errors: {len(all_errors)}")

    # Step 2: Filter to edge cases
    print(f"\nStep 2: Filtering to edge cases ({min_models}+ models)...")
    edge_cases = [e for e in all_errors if e['num_models_failed'] >= min_models]
    print(f"  Edge cases: {len(edge_cases)}")
    print(f"  Excluded: {len(all_errors) - len(edge_cases)}")

    if not edge_cases:
        print("[ERROR] No edge cases found")
        return 1

    # Stats
    by_domain = defaultdict(int)
    by_direction = defaultdict(int)
    for e in edge_cases:
        by_domain[e['domain']] += 1
        by_direction[e['actual_sentiment']] += 1

    print(f"\n  By domain: {dict(by_domain)}")
    print(f"  By direction: {dict(by_direction)}")

    # Step 3: Create sample
    print(f"\nStep 3: Creating stratified sample (target={target_size})...")
    sample = create_stratified_sample(edge_cases, target_size, random_seed)
    print(f"\n  Sample size: {len(sample)}")

    # Step 4: Save
    print("\nStep 4: Saving...")
    os.makedirs(output_dir, exist_ok=True)

    sample_file = os.path.join(output_dir, 'categorization_sample.json')
    with open(sample_file, 'w') as f:
        json.dump(sample, f, indent=2)
    print(f"  {sample_file}")

    stats = {
        'generated_at': datetime.now().isoformat(),
        'config_version': config.version,
        'total_errors': len(all_errors),
        'edge_cases': len(edge_cases),
        'sample_size': len(sample),
        'min_models_failed': min_models,
        'by_domain': dict(by_domain)
    }
    stats_file = os.path.join(output_dir, 'categorization_stats.json')
    with open(stats_file, 'w') as f:
        json.dump(stats, f, indent=2)
    print(f"  {stats_file}")

    print("\n" + "=" * 60)
    print("READY FOR CATEGORIZATION")
    print("=" * 60)
    print(f"  {len(sample)} samples to categorize")
    print()
    print("Next: python scripts/categorize_errors.py")
    print()

    return 0


if __name__ == '__main__':
    exit(main())
