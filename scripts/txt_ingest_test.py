"""Test script for ingesting maintenance report txt files into Elasticsearch.

Usage:
    python scripts/txt_ingest_test.py [txt_file_path]

Example:
    python scripts/txt_ingest_test.py data/sample_maintenance_report.txt
"""

import sys
from pathlib import Path

# Add backend to path
backend_dir = Path(__file__).parent.parent / "backend"
sys.path.insert(0, str(backend_dir))

from backend.services.es_ingest_service import EsIngestService


def main():
    """Main test function."""
    # Get txt file path
    if len(sys.argv) > 1:
        txt_file = sys.argv[1]
    else:
        txt_file = "data/sample_maintenance_report.txt"

    txt_path = Path(txt_file)
    if not txt_path.exists():
        print(f"Error: File not found: {txt_file}")
        sys.exit(1)

    print(f"Ingesting maintenance report: {txt_path}")
    print("-" * 80)

    # Initialize service
    print("Initializing EsIngestService...")
    service = EsIngestService.from_settings()

    # Ingest txt file (doc_id is auto-generated as "myservice_{title}")
    print(f"Calling ingest_myservice_txt()...")

    try:
        result = service.ingest_myservice_txt(
            file=str(txt_path),
            lang="ko",
            tags=["test", "myservice"],
            refresh=True,  # Refresh for immediate search
            use_llm_summary=True,
        )

        print("\n" + "=" * 80)
        print("✓ Ingestion successful!")
        print("=" * 80)
        print(f"Doc ID: {result.doc_id}")
        print(f"Index: {result.index}")
        print(f"Total sections: {result.total_sections}")
        print(f"Total chunks: {result.total_chunks}")
        print(f"Indexed chunks: {result.indexed_chunks}")
        print(f"Failed chunks: {result.failed_chunks}")
        print("\nMetadata:")
        for key, value in result.metadata.items():
            if key == "sections":
                print(f"  {key}:")
                for section_name, section_text in value.items():
                    preview = section_text[:100] if section_text else ""
                    print(f"    - {section_name}: {preview}...")
            else:
                print(f"  {key}: {value}")

    except Exception as e:
        print("\n" + "=" * 80)
        print("✗ Ingestion failed!")
        print("=" * 80)
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
