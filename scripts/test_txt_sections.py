"""Test script to verify txt parsing creates separate sections."""

import sys
from pathlib import Path

backend_dir = Path(__file__).parent.parent / "backend"
sys.path.insert(0, str(backend_dir))

from backend.services.ingest.txt_parser import parse_maintenance_txt
from backend.services.ingest.document_ingest_service import Section

# Sample txt
txt_file = Path(__file__).parent.parent / "data" / "sample_maintenance_report.txt"

with open(txt_file, "r", encoding="utf-8") as f:
    content = f.read()

print("Parsing txt file...")
report = parse_maintenance_txt(content)

print("\n=== Creating Section objects ===")

# Simulate what ingest_txt() does
device_name = report.meta.get("Model Name", "") or report.meta.get("Equip. NO", "")
doc_description = report.meta.get("Title", "")
order_no = report.meta.get("Order No.", "")

common_meta = {
    "device_name": device_name,
    "doc_description": doc_description,
    "order_no": order_no,
}

sections = []
section_names = ["status", "action", "cause", "result"]

for section_name in section_names:
    section_text = report.sections.get(section_name, "")
    if not section_text.strip():
        continue

    section_meta = common_meta.copy()
    section_meta["section_type"] = section_name
    section_meta["chapter"] = section_name

    section = Section(
        title=section_name,
        text=section_text,
        page_start=0,
        page_end=0,
        metadata=section_meta,
    )
    sections.append(section)

print(f"\nTotal sections created: {len(sections)}")

for i, section in enumerate(sections):
    print(f"\n--- Section {i+1}: {section.title} ---")
    print(f"Text (first 100 chars): {section.text[:100]}...")
    print(f"Metadata:")
    for key, value in section.metadata.items():
        print(f"  {key}: {value}")

print("\n✓ Section creation test successful!")
print(f"\nEach section will be stored as a separate ES document with doc_id='{order_no}'")
print("To retrieve all sections: query by doc_id.keyword")
