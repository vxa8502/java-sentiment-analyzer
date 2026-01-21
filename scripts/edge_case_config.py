#!/usr/bin/env python3
"""
Edge Case Evaluation Configuration Loader

Loads config from config/edge-case-evaluation.json.
Shared format with Java EdgeCaseConfig.java for reproducibility.

Usage:
    from edge_case_config import load_config, get_config

    config = load_config()  # Load from default path
    config = load_config('/path/to/config.json')  # Load from specific path

    # Access config values
    print(config.sampling.pilot_size)
    print(config.categories)
"""

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

DEFAULT_CONFIG_PATH = "config/edge-case-evaluation.json"
ENV_CONFIG_PATH = "EDGE_CASE_CONFIG"

# Global cached config
_cached_config = None


@dataclass
class Directories:
    edge_cases_dir: str = "data/raw/edge_cases"
    candidates_dir: str = "data/raw/edge_cases/candidates"
    models_dir: str = "models"


@dataclass
class Sampling:
    min_samples_per_category: int = 40
    target_margin_of_error: float = 0.15
    distribution_buffer: float = 1.25
    random_seed: int = 42


@dataclass
class EdgeCaseDefinition:
    """Defines what qualifies as an edge case vs a one-off model error."""
    min_models_failed: int = 3
    rationale: str = "Errors where 3+ algorithms failed indicate systematic difficulty"


@dataclass
class Filtering:
    label_error_threshold: float = 0.25
    high_signal_min_models: int = 3
    very_conservative_min_models: int = 4


@dataclass
class Evaluation:
    confidence_level: float = 0.95
    ci_method: str = "wilson"
    min_samples_for_inclusion: int = 30


@dataclass
class Category:
    id: str
    name: str
    description: str


@dataclass
class EdgeCaseConfig:
    version: str = "1.0.0"
    directories: Directories = field(default_factory=Directories)
    edge_case_definition: EdgeCaseDefinition = field(default_factory=EdgeCaseDefinition)
    sampling: Sampling = field(default_factory=Sampling)
    filtering: Filtering = field(default_factory=Filtering)
    evaluation: Evaluation = field(default_factory=Evaluation)
    categories: List[Category] = field(default_factory=list)
    algorithms: List[str] = field(default_factory=list)
    domains: List[str] = field(default_factory=list)

    @property
    def category_ids(self) -> List[str]:
        return [c.id for c in self.categories]

    @property
    def min_models_for_edge_case(self) -> int:
        """Convenience accessor for the edge case threshold."""
        return self.edge_case_definition.min_models_failed

    @property
    def categorization_sample_size(self) -> int:
        """
        Compute target sample size for categorization based on statistical requirements.

        Formula: min_samples_per_category * num_categories * distribution_buffer

        - min_samples_per_category: ensures each category has enough samples for valid CI
        - num_categories: total categories to distribute samples across
        - distribution_buffer: accounts for uneven category distribution (default 1.25)
        """
        return int(
            self.sampling.min_samples_per_category
            * len(self.categories)
            * self.sampling.distribution_buffer
        )

    def validate(self):
        """Validate config has required fields."""
        if not self.categories:
            raise ValueError("Missing or empty 'categories' in config")
        if not self.algorithms:
            raise ValueError("Missing or empty 'algorithms' in config")
        if not self.domains:
            raise ValueError("Missing or empty 'domains' in config")
        if self.edge_case_definition.min_models_failed < 1:
            raise ValueError("edge_case_definition.min_models_failed must be >= 1")

    def __str__(self):
        return (f"EdgeCaseConfig[version={self.version}, "
                f"edge_case_min_models={self.min_models_for_edge_case}, "
                f"categories={len(self.categories)}, "
                f"algorithms={len(self.algorithms)}, "
                f"domains={len(self.domains)}]")


def _parse_config(data: dict) -> EdgeCaseConfig:
    """Parse JSON dict into EdgeCaseConfig dataclass."""

    directories = Directories(
        edge_cases_dir=data.get("directories", {}).get("edge_cases_dir", "data/raw/edge_cases"),
        candidates_dir=data.get("directories", {}).get("candidates_dir", "data/raw/edge_cases/candidates"),
        models_dir=data.get("directories", {}).get("models_dir", "models"),
    )

    edge_case_def_data = data.get("edge_case_definition", {})
    edge_case_definition = EdgeCaseDefinition(
        min_models_failed=edge_case_def_data.get("min_models_failed", 3),
        rationale=edge_case_def_data.get("rationale", "Errors where 3+ algorithms failed indicate systematic difficulty"),
    )

    sampling = Sampling(
        min_samples_per_category=data.get("sampling", {}).get("min_samples_per_category", 40),
        target_margin_of_error=data.get("sampling", {}).get("target_margin_of_error", 0.15),
        distribution_buffer=data.get("sampling", {}).get("distribution_buffer", 1.25),
        random_seed=data.get("sampling", {}).get("random_seed", 42),
    )

    filtering = Filtering(
        label_error_threshold=data.get("filtering", {}).get("label_error_threshold", 0.25),
        high_signal_min_models=data.get("filtering", {}).get("high_signal_min_models", 3),
        very_conservative_min_models=data.get("filtering", {}).get("very_conservative_min_models", 4),
    )

    evaluation = Evaluation(
        confidence_level=data.get("evaluation", {}).get("confidence_level", 0.95),
        ci_method=data.get("evaluation", {}).get("ci_method", "wilson"),
        min_samples_for_inclusion=data.get("evaluation", {}).get("min_samples_for_inclusion", 30),
    )

    categories = [
        Category(id=c["id"], name=c["name"], description=c["description"])
        for c in data.get("categories", [])
    ]

    return EdgeCaseConfig(
        version=data.get("_version", "1.0.0"),
        directories=directories,
        edge_case_definition=edge_case_definition,
        sampling=sampling,
        filtering=filtering,
        evaluation=evaluation,
        categories=categories,
        algorithms=data.get("algorithms", []),
        domains=data.get("domains", []),
    )


def load_config(path: Optional[str] = None) -> EdgeCaseConfig:
    """
    Load config from file.

    Priority:
    1. Explicit path argument
    2. EDGE_CASE_CONFIG environment variable
    3. Default path (config/edge-case-evaluation.json)
    """
    global _cached_config

    if path is None:
        path = os.environ.get(ENV_CONFIG_PATH, DEFAULT_CONFIG_PATH)

    config_path = Path(path)

    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_path, 'r') as f:
        data = json.load(f)

    config = _parse_config(data)
    config.validate()

    _cached_config = config
    return config


def get_config() -> EdgeCaseConfig:
    """
    Get cached config or load from default path.
    """
    global _cached_config
    if _cached_config is None:
        return load_config()
    return _cached_config


def get_config_path() -> Path:
    """Get the path to the config file."""
    path = os.environ.get(ENV_CONFIG_PATH, DEFAULT_CONFIG_PATH)
    return Path(path)


# Wilson score confidence interval calculation (matches Java implementation)
def wilson_ci(successes: int, trials: int, confidence: float = 0.95) -> tuple:
    """
    Calculate Wilson score confidence interval.
    Better than normal approximation for small samples.

    Returns: (lower, upper, margin_of_error)
    """
    if trials == 0:
        return (0.0, 1.0, 0.5)

    z = 1.96 if confidence == 0.95 else 1.645  # 95% or 90%
    p = successes / trials
    n = trials

    denominator = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denominator
    spread = z * ((p * (1 - p) + z * z / (4 * n)) / n) ** 0.5 / denominator

    lower = max(0, center - spread)
    upper = min(1, center + spread)
    margin = (upper - lower) / 2

    return (lower, upper, margin)


if __name__ == "__main__":
    # Test config loading
    try:
        config = load_config()
        print(f"Config loaded successfully: {config}")
        print(f"  Edge cases dir: {config.directories.edge_cases_dir}")
        print(f"  Edge case threshold: {config.min_models_for_edge_case}+ models failed")
        print(f"  Edge case rationale: {config.edge_case_definition.rationale}")
        print(f"  Min samples/category: {config.sampling.min_samples_per_category}")
        print(f"  Distribution buffer: {config.sampling.distribution_buffer}")
        print(f"  Categorization sample size: {config.categorization_sample_size}")
        print(f"  Confidence level: {config.evaluation.confidence_level}")
        print(f"  Categories: {', '.join(config.category_ids)}")
        print(f"  Algorithms: {', '.join(config.algorithms)}")
        print(f"  Domains: {', '.join(config.domains)}")

        # Test Wilson CI
        print("\nWilson CI test (40 correct out of 50):")
        lower, upper, margin = wilson_ci(40, 50)
        print(f"  Accuracy: 80.0%")
        print(f"  95% CI: [{lower:.1%}, {upper:.1%}]")
        print(f"  Margin of error: +/- {margin:.1%}")

    except Exception as e:
        print(f"Failed to load config: {e}")
        exit(1)
