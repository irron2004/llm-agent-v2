"""Batch ingest all txt files from gcb folder.

Usage:
    python scripts/batch_ingest_gcb.py
"""

import sys
import time
from pathlib import Path
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from backend.services.es_ingest_service import EsIngestService
from backend.config.settings import search_settings
from elasticsearch import Elasticsearch


def get_es_stats(es, index):
    """Get ES index statistics."""
    try:
        stats = es.indices.stats(index=index)
        doc_count = stats["_all"]["primaries"]["docs"]["count"]
        store_size = stats["_all"]["primaries"]["store"]["size_in_bytes"]
        return {
            "doc_count": doc_count,
            "size_mb": store_size / (1024 * 1024),
        }
    except Exception:
        return {"doc_count": 0, "size_mb": 0}


def get_gcb_stats(es, index):
    """Get GCB document statistics."""
    try:
        result = es.search(
            index=index,
            body={
                "query": {"term": {"doc_type.keyword": "gcb"}},
                "size": 0,
                "aggs": {
                    "by_chapter": {
                        "terms": {"field": "chapter.keyword", "size": 10}
                    }
                }
            }
        )
        total = result["hits"]["total"]["value"]
        chapters = {
            bucket["key"]: bucket["doc_count"]
            for bucket in result["aggregations"]["by_chapter"]["buckets"]
        }
        return {"total": total, "chapters": chapters}
    except Exception:
        return {"total": 0, "chapters": {}}


def get_existing_gcb_doc_ids(es, index):
    """Get set of already ingested GCB doc_ids."""
    try:
        result = es.search(
            index=index,
            body={
                "query": {"term": {"doc_type.keyword": "gcb"}},
                "size": 0,
                "aggs": {
                    "doc_ids": {
                        "terms": {"field": "doc_id", "size": 50000}
                    }
                }
            }
        )
        doc_ids = {
            bucket["key"]
            for bucket in result["aggregations"]["doc_ids"]["buckets"]
        }
        return doc_ids
    except Exception:
        return set()


def ingest_single_file(txt_file, service):
    """Ingest a single txt file. Returns (success, doc_id, error_msg, num_sections)."""
    try:
        result = service.ingest_gcb_txt(
            file=str(txt_file),
            lang="en",
            tags=["gcb", "batch"],
            refresh=False,
            use_llm_summary=True,
        )

        if result.indexed_chunks > 0:
            return (True, result.doc_id, None, result.total_sections)
        else:
            return (False, result.doc_id, "No chunks indexed", 0)

    except Exception as e:
        return (False, txt_file.stem, str(e), 0)


def main():
    """Batch ingest all GCB txt files."""
    # Source directory (Docker mount: /data/pe_agent_data, Host: /home/llm-share/datasets/pe_agent_data)
    source_dir = Path("/data/pe_agent_data/pe_preprocess_data/gcb")
    if not source_dir.exists():
        # Fallback to host path
        source_dir = Path("/home/llm-share/datasets/pe_agent_data/pe_preprocess_data/gcb")

    if not source_dir.exists():
        print(f"Error: Directory not found: {source_dir}")
        sys.exit(1)

    # Get all txt files
    txt_files = sorted(source_dir.glob("*.txt"))
    total_files = len(txt_files)

    if total_files == 0:
        print(f"No txt files found in {source_dir}")
        sys.exit(0)

    print(f"Found {total_files} GCB txt files")
    print("=" * 80)

    # Initialize service
    print("Initializing EsIngestService...")
    service = EsIngestService.from_settings()
    es = service.es
    index = service.index
    print(f"Index: {index}")

    # Initial ES stats
    print("\n📊 Initial ES Stats:")
    initial_stats = get_es_stats(es, index)
    print(f"  Total docs: {initial_stats['doc_count']}")
    print(f"  Index size: {initial_stats['size_mb']:.2f} MB")

    # Get already ingested doc_ids and filter out
    print("\n🔍 Checking existing GCB documents...")
    existing_doc_ids = get_existing_gcb_doc_ids(es, index)
    print(f"  Already ingested: {len(existing_doc_ids)} GCB files")

    # Filter out already ingested files
    txt_files = [f for f in txt_files if f.stem not in existing_doc_ids]
    files_to_process = len(txt_files)
    skipped_count = total_files - files_to_process

    print(f"  Skipping: {skipped_count} files")
    print(f"  To process: {files_to_process} files")
    print("=" * 80)

    if files_to_process == 0:
        print("\n✅ All GCB files already ingested. Nothing to do.")
        return

    # Batch processing
    success_count = 0
    failed_count = 0
    failed_files = []
    section_stats = defaultdict(int)

    start_time = time.time()
    report_interval = 50
    max_workers = 4

    print(f"\nProcessing {files_to_process} files with {max_workers} workers...")
    print("=" * 80)

    # Parallel processing with ThreadPoolExecutor
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_file = {
            executor.submit(ingest_single_file, txt_file, service): txt_file
            for txt_file in txt_files
        }

        with tqdm(total=files_to_process, desc="Ingesting GCB", unit="file") as pbar:
            processed = 0
            for future in as_completed(future_to_file):
                processed += 1
                txt_file = future_to_file[future]

                try:
                    success, doc_id, error_msg, num_sections = future.result()

                    if success:
                        success_count += 1
                        section_stats[num_sections] += 1
                    else:
                        failed_count += 1
                        failed_files.append((doc_id, error_msg or "Unknown error"))
                        if error_msg and "No chunks indexed" not in error_msg:
                            tqdm.write(f"✗ Failed: {doc_id} - {error_msg[:80]}")

                except Exception as e:
                    doc_id = txt_file.stem
                    failed_count += 1
                    failed_files.append((doc_id, str(e)))
                    tqdm.write(f"✗ Exception: {doc_id} - {e}")

                pbar.update(1)

                # Periodic progress report
                if processed % report_interval == 0:
                    elapsed = time.time() - start_time
                    gcb_stats = get_gcb_stats(es, index)
                    print(f"\n[Progress] {processed}/{files_to_process} | Success: {success_count} | GCB docs: {gcb_stats['total']} | Elapsed: {elapsed:.1f}s")

    # Refresh index once at the end
    print("\nRefreshing index...")
    service.es.indices.refresh(index=index)

    # Final summary
    total_time = time.time() - start_time
    print("\n" + "=" * 80)
    print("🎉 GCB BATCH INGESTION COMPLETE")
    print("=" * 80)
    print(f"Total files:     {total_files}")
    print(f"Skipped:         {skipped_count} (already ingested)")
    print(f"Processed:       {files_to_process}")
    print(f"✓ Success:       {success_count}")
    print(f"✗ Failed:        {failed_count}")
    print(f"⏱️  Total time:    {total_time:.1f}s ({total_time/60:.1f}min)")
    if files_to_process > 0:
        print(f"⚡ Avg speed:     {total_time/files_to_process:.2f}s/file")

    # Final ES stats
    print("\n" + "-" * 80)
    print("📊 Final ES Statistics:")
    print("-" * 80)
    final_stats = get_es_stats(es, index)
    print(f"Total docs:      {final_stats['doc_count']}")
    print(f"Index size:      {final_stats['size_mb']:.2f} MB")
    print(f"Docs added:      {final_stats['doc_count'] - initial_stats['doc_count']}")

    gcb_stats = get_gcb_stats(es, index)
    print(f"\nGCB docs:        {gcb_stats['total']}")
    if gcb_stats['chapters']:
        print("Sections breakdown:")
        for chapter, count in sorted(gcb_stats['chapters'].items()):
            print(f"  - {chapter:>12}: {count:>6}")

    if failed_files:
        print("\n" + "-" * 80)
        print("⚠️  Failed files:")
        for doc_id, error in failed_files[:10]:
            print(f"  - {doc_id}: {error[:60]}")
        if len(failed_files) > 10:
            print(f"  ... and {len(failed_files) - 10} more")

    print("=" * 80)


if __name__ == "__main__":
    main()
