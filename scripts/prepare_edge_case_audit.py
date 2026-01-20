#!/usr/bin/env python3
"""
Edge Case Audit Preparation Pipeline

This script prepares error candidates for human audit by:
1. Loading all error CSVs from ErrorAnalyzer
2. Deduplicating across models (same text may fail on multiple models)
3. Computing statistics and signal strength
4. Creating a stratified sample for human audit

All parameters loaded from config/edge-case-evaluation.json for reproducibility.

Usage:
    python scripts/prepare_edge_case_audit.py [--sample-pct 2.0] [--min-sample 100] [--max-sample 200]

Outputs:
    data/raw/edge_cases/deduplicated_errors.json  - All unique errors with metadata
    data/raw/edge_cases/audit_sample.json         - Stratified sample for audit
    data/raw/edge_cases/audit_stats.json          - Statistics about the error population
"""

import argparse
import csv
import hashlib
import json
import os
import random
from collections import defaultdict
from datetime import datetime
from pathlib import Path

# Import shared config loader
from edge_case_config import load_config, EdgeCaseConfig


def parse_error_filename(filename, algorithms):
    """Extract algorithm and domain from error CSV filename."""
    base = filename.replace('_errors.csv', '')

    for algo in algorithms:
        if base.startswith(algo + '_'):
            domain = base[len(algo) + 1:]
            return algo, domain

    return None, None


def load_all_errors(candidates_dir, algorithms):
    """Load and deduplicate errors across all model error files."""
    all_errors = {}
    file_stats = {}

    for filename in os.listdir(candidates_dir):
        if not filename.endswith('_errors.csv'):
            continue

        algo, domain = parse_error_filename(filename, algorithms)
        if not algo:
            print(f"  [WARN] Could not parse filename: {filename}")
            continue

        model_name = f"{algo}-{domain}"
        filepath = os.path.join(candidates_dir, filename)

        file_error_count = 0
        with open(filepath, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                text = row.get('text', '').strip()
                if not text:
                    continue

                # Create hash for deduplication
                text_hash = hashlib.md5(text.encode()).hexdigest()

                if text_hash not in all_errors:
                    all_errors[text_hash] = {
                        'hash': text_hash,
                        'text': text,
                        'actual_sentiment': row.get('actual_sentiment', '').upper(),
                        'failed_models': set(),  # Use set to prevent duplicates
                        'predictions': {},
                        'confidences': {},
                        'domains': set()
                    }

                all_errors[text_hash]['failed_models'].add(model_name)  # Set.add() prevents duplicates
                all_errors[text_hash]['predictions'][model_name] = row.get('predicted_sentiment', '').lower()
                all_errors[text_hash]['confidences'][model_name] = row.get('confidence', '')
                all_errors[text_hash]['domains'].add(domain)
                file_error_count += 1

        file_stats[model_name] = file_error_count
        print(f"  Loaded {file_error_count} errors from {model_name}")

    return all_errors, file_stats


def compute_error_metadata(all_errors):
    """Add computed fields to each error."""
    for h, error in all_errors.items():
        # Convert sets to sorted lists for JSON serialization and deterministic output
        error['failed_models'] = sorted(list(error['failed_models']))
        error['domains'] = sorted(list(error['domains']))
        error['num_models_failed'] = len(error['failed_models'])

        # Determine primary domain (most common among failing models)
        domain_counts = defaultdict(int)
        for model in error['failed_models']:
            domain = model.split('-')[1]
            domain_counts[domain] += 1
        error['domain'] = max(domain_counts, key=domain_counts.get)

    return all_errors


def compute_statistics(all_errors):
    """Compute population statistics for documentation."""
    stats = {
        'generated_at': datetime.now().isoformat(),
        'total_unique_errors': len(all_errors),
        'by_num_models_failed': defaultdict(int),
        'by_domain': defaultdict(int),
        'by_error_direction': defaultdict(int),
        'by_signal_strength': {
            'high_signal_3plus': 0,
            'medium_signal_1_2': 0
        }
    }

    for error in all_errors.values():
        n = error['num_models_failed']
        stats['by_num_models_failed'][str(n)] += 1

        if n >= 3:
            stats['by_signal_strength']['high_signal_3plus'] += 1
        else:
            stats['by_signal_strength']['medium_signal_1_2'] += 1

        for domain in error['domains']:
            stats['by_domain'][domain] += 1

        direction = f"{error['actual_sentiment']}_to_predicted"
        stats['by_error_direction'][direction] += 1

    # Convert defaultdicts to regular dicts for JSON
    stats['by_num_models_failed'] = dict(stats['by_num_models_failed'])
    stats['by_domain'] = dict(stats['by_domain'])
    stats['by_error_direction'] = dict(stats['by_error_direction'])

    return stats


def create_stratified_sample(all_errors, sample_pct, min_sample, max_sample, random_seed, high_signal_min):
    """
    Create a stratified sample for human audit.

    Stratification:
    - 50% high-signal (high_signal_min+ models failed)
    - 50% medium-signal (fewer models failed)
    - Within each: balanced by domain and error direction
    """
    random.seed(random_seed)

    # Calculate target sample size
    total_errors = len(all_errors)
    target_size = int(total_errors * (sample_pct / 100))
    target_size = max(min_sample, min(max_sample, target_size))

    print(f"\n  Target sample size: {target_size}")
    print(f"    (Based on {sample_pct}% of {total_errors}, min={min_sample}, max={max_sample})")

    errors_list = list(all_errors.values())

    # Split by signal strength
    high_signal = [e for e in errors_list if e['num_models_failed'] >= high_signal_min]
    medium_signal = [e for e in errors_list if e['num_models_failed'] < high_signal_min]

    print(f"  High-signal pool: {len(high_signal)}")
    print(f"  Medium-signal pool: {len(medium_signal)}")

    # Target 50/50 split
    high_target = target_size // 2
    medium_target = target_size - high_target

    sample = []

    # Sample from high-signal, stratified by domain and direction
    sample.extend(stratified_sample_from_pool(high_signal, high_target, 'high-signal'))

    # Sample from medium-signal, stratified by domain and direction
    sample.extend(stratified_sample_from_pool(medium_signal, medium_target, 'medium-signal'))

    # Shuffle to avoid bias during review
    random.shuffle(sample)

    return sample, {
        'target_size': target_size,
        'actual_size': len(sample),
        'high_signal_count': len([s for s in sample if s['num_models_failed'] >= high_signal_min]),
        'medium_signal_count': len([s for s in sample if s['num_models_failed'] < high_signal_min]),
        'sample_pct': sample_pct,
        'random_seed': random_seed,
        'high_signal_threshold': high_signal_min
    }


def stratified_sample_from_pool(pool, target_count, pool_name):
    """Sample from a pool with stratification by domain and direction."""
    if not pool:
        print(f"    {pool_name}: Empty pool, skipping")
        return []

    # Group by domain and direction
    buckets = defaultdict(list)
    for error in pool:
        key = (error['domain'], error['actual_sentiment'])
        buckets[key].append(error)

    # Calculate per-bucket quota
    num_buckets = len(buckets)
    if num_buckets == 0:
        return []

    base_quota = target_count // num_buckets
    remainder = target_count % num_buckets

    sample = []
    for i, (key, errors) in enumerate(sorted(buckets.items())):
        quota = base_quota + (1 if i < remainder else 0)
        actual = min(quota, len(errors))
        sample.extend(random.sample(errors, actual))
        print(f"    {pool_name} {key}: {actual}/{len(errors)} sampled")

    return sample


def main():
    parser = argparse.ArgumentParser(description='Prepare edge case errors for audit')
    parser.add_argument('--sample-pct', type=float, default=None,
                        help='Percentage of unique errors to sample (default: from config)')
    parser.add_argument('--min-sample', type=int, default=None,
                        help='Minimum sample size (default: from config)')
    parser.add_argument('--max-sample', type=int, default=None,
                        help='Maximum sample size (default: from config)')
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

    # Get values from config (CLI args override config)
    candidates_dir = config.directories.candidates_dir
    output_dir = config.directories.edge_cases_dir
    random_seed = config.sampling.random_seed
    algorithms = config.algorithms
    high_signal_min = config.filtering.high_signal_min_models

    # Default sample params (not in config yet, use sensible defaults)
    sample_pct = args.sample_pct if args.sample_pct is not None else 2.0
    min_sample = args.min_sample if args.min_sample is not None else 100
    max_sample = args.max_sample if args.max_sample is not None else 200

    print("=" * 60)
    print("EDGE CASE AUDIT PREPARATION")
    print("=" * 60)
    print(f"Timestamp: {datetime.now().isoformat()}")
    print(f"Config version: {config.version}")
    print(f"Parameters: sample_pct={sample_pct}%, min={min_sample}, max={max_sample}")
    print(f"Random seed: {random_seed}")
    print(f"High-signal threshold: {high_signal_min}+ models")
    print()

    # Step 1: Load all errors
    print("Step 1: Loading error candidates from all models...")
    if not os.path.exists(candidates_dir):
        print(f"  [ERROR] Candidates directory not found: {candidates_dir}")
        print("  Run './scripts/generate_edge_cases.sh all' first")
        return 1

    all_errors, file_stats = load_all_errors(candidates_dir, algorithms)
    print(f"\n  Total raw error lines: {sum(file_stats.values())}")
    print(f"  Unique errors (deduplicated): {len(all_errors)}")

    if not all_errors:
        print("  [ERROR] No errors found. Check that models are trained and error CSVs exist.")
        return 1

    # Step 2: Compute metadata
    print("\nStep 2: Computing error metadata...")
    all_errors = compute_error_metadata(all_errors)

    # Step 3: Compute statistics
    print("\nStep 3: Computing population statistics...")
    stats = compute_statistics(all_errors)

    print(f"  By number of models failed:")
    for n, count in sorted(stats['by_num_models_failed'].items(), key=lambda x: int(x[0])):
        print(f"    {n} model(s): {count}")

    print(f"  By domain:")
    for domain, count in sorted(stats['by_domain'].items()):
        print(f"    {domain}: {count}")

    print(f"  By error direction:")
    for direction, count in sorted(stats['by_error_direction'].items()):
        print(f"    {direction}: {count}")

    # Step 4: Create stratified sample
    print("\nStep 4: Creating stratified audit sample...")
    sample, sample_stats = create_stratified_sample(
        all_errors, sample_pct, min_sample, max_sample, random_seed, high_signal_min
    )

    stats['sample'] = sample_stats
    stats['config_version'] = config.version

    # Step 5: Save outputs
    print("\nStep 5: Saving outputs...")

    os.makedirs(output_dir, exist_ok=True)

    # Save deduplicated errors (convert to list for JSON)
    dedup_file = os.path.join(output_dir, 'deduplicated_errors.json')
    with open(dedup_file, 'w') as f:
        json.dump(list(all_errors.values()), f, indent=2)
    print(f"  Saved: {dedup_file} ({len(all_errors)} errors)")

    # Save audit sample
    sample_file = os.path.join(output_dir, 'audit_sample.json')
    with open(sample_file, 'w') as f:
        json.dump(sample, f, indent=2)
    print(f"  Saved: {sample_file} ({len(sample)} samples)")

    # Save statistics
    stats_file = os.path.join(output_dir, 'audit_stats.json')
    stats['file_stats'] = file_stats
    with open(stats_file, 'w') as f:
        json.dump(stats, f, indent=2)
    print(f"  Saved: {stats_file}")

    # Summary
    print("\n" + "=" * 60)
    print("AUDIT PREPARATION COMPLETE")
    print("=" * 60)
    print(f"  Unique errors: {len(all_errors)}")
    print(f"  Audit sample:  {len(sample)} ({sample_stats['high_signal_count']} high-signal, {sample_stats['medium_signal_count']} medium-signal)")
    print()
    print("Next step: Run the audit tool")
    print("  python scripts/audit_edge_cases.py")
    print()

    return 0


if __name__ == '__main__':
    exit(main())
