"""Quick test for txt parser."""

import sys
from pathlib import Path

backend_dir = Path(__file__).parent.parent / "backend"
sys.path.insert(0, str(backend_dir))

from backend.services.ingest.txt_parser import parse_maintenance_txt

# Sample txt
txt_file = Path(__file__).parent.parent / "data" / "sample_maintenance_report.txt"

with open(txt_file, "r", encoding="utf-8") as f:
    content = f.read()

print("Parsing txt file...")
report = parse_maintenance_txt(content)

print("\n=== META ===")
for key, value in report.meta.items():
    print(f"{key}: {value}")

print("\n=== SECTIONS ===")
for section_name, section_text in report.sections.items():
    print(f"\n[{section_name}]")
    print(section_text[:200])

print("\n=== FULL TEXT (first 500 chars) ===")
print(report.full_text[:500])

print("\n✓ Parser test successful!")
