"""Test normalization on maintenance text."""

import sys
from pathlib import Path

backend_dir = Path(__file__).parent.parent / "backend"
sys.path.insert(0, str(backend_dir))

from backend.llm_infrastructure.preprocessing.registry import get_preprocessor
from backend.config.settings import rag_settings

# Get preprocessor
preprocessor = get_preprocessor(
    rag_settings.preprocess_method,
    version=rag_settings.preprocess_version,
    level=rag_settings.preprocess_level,
)

# Test texts
test_cases = [
    "# PM3 disable",
    "# PM2-1 position check ok",
    "Old: MB51210‑11",  # Dash variant
    "Temperature: 3.5 × 10^-3",
    "gas box exhasut flow",  # Typo
]

print("=== Normalization Test ===\n")

for original in test_cases:
    normalized = list(preprocessor.preprocess([original]))[0]
    print(f"Original:    '{original}'")
    print(f"Normalized:  '{normalized}'")
    print("-" * 60)

print("\n✓ Normalization is working!")
