#!/usr/bin/env python3
"""
Categorization Sample Preparation Pipeline

This script prepares a stratified sample of EDGE CASES for manual categorization.

Sample size is computed from config using the formula:
    sample_size = min_samples_per_category * num_categories * distribution_buffer

This ensures each category has statistically sufficient samples for valid
confidence intervals, with buffer for uneven distribution.

IMPORTANT: Only errors meeting the edge case definition are included.
An edge case is a SYSTEMATIC model failure where multiple algorithms failed,
not a one-off model quirk. The threshold is defined in config as:
    edge_case_definition.min_models_failed (default: 3)

All parameters loaded from config/edge-case-evaluation.json for reproducibility.

Usage:
    python scripts/prepare_categorization_sample.py

Inputs:
    data/raw/edge_cases/deduplicated_errors.json - All unique errors from deduplication

Outputs:
    data/raw/edge_cases/categorization_sample.json - Stratified sample of edge cases
    data/raw/edge_cases/categorization_stats.json  - Statistics about population and sample
"""

import argparse
import json
import os
import random
from collections import defaultdict
from datetime import datetime

# Import shared config loader
from edge_case_config import load_config


def load_json(filepath):
    """Load JSON file."""
    with open(filepath, 'r') as f:
        return json.load(f)


def filter_to_edge_cases(errors, min_models_failed):
    """
    Filter errors to only include edge cases (systematic failures).

    An edge case is an error where min_models_failed or more algorithms
    failed on the same text. Single-model failures are excluded as they
    represent model-specific quirks, not systematic difficulty.
    """
    edge_cases = [e for e in errors if e['num_models_failed'] >= min_models_failed]
    excluded = len(errors) - len(edge_cases)
    return edge_cases, excluded


def compute_population_stats(all_errors, edge_cases, min_models_failed):
    """Compute statistics about the full population and edge case subset."""
    stats = {
        'all_verified_errors': len(all_errors),
        'edge_cases': len(edge_cases),
        'excluded_single_model_failures': len(all_errors) - len(edge_cases),
        'edge_case_threshold': min_models_failed,
        'by_num_models_failed': defaultdict(int),
        'by_domain': defaultdict(int),
        'by_error_direction': defaultdict(int),
    }

    for error in edge_cases:
        n = error['num_models_failed']
        stats['by_num_models_failed'][str(n)] += 1
        stats['by_domain'][error['domain']] += 1
        direction = f"{error['actual_sentiment']}_to_predicted"
        stats['by_error_direction'][direction] += 1

    # Convert defaultdicts for JSON
    stats['by_num_models_failed'] = dict(stats['by_num_models_failed'])
    stats['by_domain'] = dict(stats['by_domain'])
    stats['by_error_direction'] = dict(stats['by_error_direction'])

    return stats


def create_stratified_sample(edge_cases, target_size, random_seed):
    """
    Create a stratified sample from edge cases for categorization.

    Stratification dimensions:
    - Domain (imdb, amazon, yelp)
    - Error direction (pos->neg vs neg->pos)

    Note: No longer stratifying by signal strength since all samples
    are already high-signal (meeting the edge case threshold).
    """
    random.seed(random_seed)

    # Group edge cases into strata: domain x direction
    strata = defaultdict(list)
    for error in edge_cases:
        domain = error['domain']
        direction = 'pos_to_neg' if error['actual_sentiment'] == 'POSITIVE' else 'neg_to_pos'
        key = (domain, direction)
        strata[key].append(error)

    print(f"\n  Strata distribution ({len(strata)} strata):")
    for key, errors_in_stratum in sorted(strata.items()):
        print(f"    {key}: {len(errors_in_stratum)}")

    # Allocate samples proportionally, with minimum 1 per stratum
    total_edge_cases = len(edge_cases)
    sample = []
    allocation = {}

    for key, errors_in_stratum in strata.items():
        # Proportional allocation
        proportion = len(errors_in_stratum) / total_edge_cases
        n = max(1, round(proportion * target_size))
        n = min(n, len(errors_in_stratum))  # Can't sample more than available
        allocation[str(key)] = {'available': len(errors_in_stratum), 'sampled': n}

        sampled = random.sample(errors_in_stratum, n)
        sample.extend(sampled)

    # Shuffle to avoid ordering bias during review
    random.shuffle(sample)

    # Compute sample stats
    sample_stats = {
        'target_size': target_size,
        'actual_size': len(sample),
        'random_seed': random_seed,
        'allocation_by_stratum': allocation,
        'sample_by_domain': defaultdict(int),
        'sample_by_direction': {'pos_to_neg': 0, 'neg_to_pos': 0},
        'sample_by_num_models': defaultdict(int),
    }

    for error in sample:
        sample_stats['sample_by_domain'][error['domain']] += 1
        sample_stats['sample_by_num_models'][str(error['num_models_failed'])] += 1
        if error['actual_sentiment'] == 'POSITIVE':
            sample_stats['sample_by_direction']['pos_to_neg'] += 1
        else:
            sample_stats['sample_by_direction']['neg_to_pos'] += 1

    sample_stats['sample_by_domain'] = dict(sample_stats['sample_by_domain'])
    sample_stats['sample_by_num_models'] = dict(sample_stats['sample_by_num_models'])

    return sample, sample_stats


def main():
    parser = argparse.ArgumentParser(description='Prepare categorization sample from edge cases')
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

    # Get values from config
    edge_cases_dir = config.directories.edge_cases_dir
    random_seed = config.sampling.random_seed
    min_models_failed = config.min_models_for_edge_case
    target_size = config.categorization_sample_size

    # File paths
    dedup_errors_file = os.path.join(edge_cases_dir, 'deduplicated_errors.json')
    sample_file = os.path.join(edge_cases_dir, 'categorization_sample.json')
    stats_file = os.path.join(edge_cases_dir, 'categorization_stats.json')

    print("=" * 60)
    print("EDGE CASE CATEGORIZATION SAMPLE PREPARATION")
    print("=" * 60)
    print(f"Timestamp: {datetime.now().isoformat()}")
    print(f"Config version: {config.version}")
    print()
    print(f"Sample size formula: min_samples_per_category * num_categories * distribution_buffer")
    print(f"  = {config.sampling.min_samples_per_category} * {len(config.categories)} * {config.sampling.distribution_buffer}")
    print(f"  = {target_size}")
    print()
    print(f"Random seed: {random_seed}")
    print(f"Edge case definition: {min_models_failed}+ models must fail")
    print(f"Rationale: {config.edge_case_definition.rationale}")
    print()

    # Check for required file
    if not os.path.exists(dedup_errors_file):
        print(f"[ERROR] Deduplicated errors not found: {dedup_errors_file}")
        print("Run 'python scripts/prepare_edge_case_audit.py' first.")
        return 1

    # Load deduplicated errors
    print("Step 1: Loading deduplicated errors...")
    all_errors = load_json(dedup_errors_file)
    print(f"  Loaded {len(all_errors)} unique errors")

    # Filter to edge cases only
    print(f"\nStep 2: Filtering to edge cases ({min_models_failed}+ models failed)...")
    edge_cases, excluded = filter_to_edge_cases(all_errors, min_models_failed)
    print(f"  Edge cases: {len(edge_cases)}")
    print(f"  Excluded (single-model failures): {excluded}")

    if not edge_cases:
        print("\n[ERROR] No edge cases found. Check min_models_failed threshold in config.")
        return 1

    # Compute population stats
    print("\nStep 3: Computing edge case statistics...")
    pop_stats = compute_population_stats(all_errors, edge_cases, min_models_failed)

    print(f"  By domain:")
    for domain, count in sorted(pop_stats['by_domain'].items()):
        print(f"    {domain}: {count}")

    print(f"  By models failed:")
    for n, count in sorted(pop_stats['by_num_models_failed'].items(), key=lambda x: int(x[0])):
        print(f"    {n} models: {count}")

    # Create stratified sample
    print(f"\nStep 4: Creating stratified sample (target={target_size})...")
    sample, sample_stats = create_stratified_sample(edge_cases, target_size, random_seed)

    print(f"\n  Sample created: {len(sample)} edge cases")
    print(f"  By domain: {sample_stats['sample_by_domain']}")
    print(f"  By models failed: {sample_stats['sample_by_num_models']}")
    print(f"  By direction: {sample_stats['sample_by_direction']}")

    # Save outputs
    print("\nStep 5: Saving outputs...")

    with open(sample_file, 'w') as f:
        json.dump(sample, f, indent=2)
    print(f"  Saved: {sample_file} ({len(sample)} samples)")

    full_stats = {
        'generated_at': datetime.now().isoformat(),
        'config_version': config.version,
        'edge_case_definition': {
            'min_models_failed': min_models_failed,
            'rationale': config.edge_case_definition.rationale
        },
        'sampling_formula': {
            'formula': 'min_samples_per_category * num_categories * distribution_buffer',
            'min_samples_per_category': config.sampling.min_samples_per_category,
            'num_categories': len(config.categories),
            'distribution_buffer': config.sampling.distribution_buffer,
            'computed_target': target_size
        },
        'population': pop_stats,
        'sample': sample_stats
    }

    with open(stats_file, 'w') as f:
        json.dump(full_stats, f, indent=2)
    print(f"  Saved: {stats_file}")

    # Summary
    print("\n" + "=" * 60)
    print("EDGE CASE SAMPLE READY")
    print("=" * 60)
    print(f"  All verified errors: {len(all_errors)}")
    print(f"  Edge cases ({min_models_failed}+ models): {len(edge_cases)}")
    print(f"  Sample for categorization: {len(sample)}")
    print()
    print("Next step: Categorize the sample")
    print("  python scripts/categorize_errors.py --sample")
    print()

    return 0


if __name__ == '__main__':
    exit(main())
