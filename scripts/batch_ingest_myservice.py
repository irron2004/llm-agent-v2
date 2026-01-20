"""Batch ingest all txt files from myservice_txt folder.

Usage:
    python scripts/batch_ingest_myservice.py
"""

import sys
import time
import json
from pathlib import Path
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

# Add backend to path
backend_dir = Path(__file__).parent.parent / "backend"
sys.path.insert(0, str(backend_dir))

from backend.services.es_ingest_service import EsIngestService
from backend.config.settings import search_settings
from elasticsearch import Elasticsearch


def has_content(txt_file):
    """Check if txt file has actual content (not empty sections)."""
    try:
        with open(txt_file, "r", encoding="utf-8") as f:
            content = f.read()

        # Quick check: look for completeness field
        if '"completeness": "empty"' in content:
            return False

        # Check if all sections_present are false
        if '"sections_present"' in content:
            try:
                # Extract the sections_present JSON
                import re
                match = re.search(r'"sections_present"\s*:\s*\{[^}]+\}', content)
                if match:
                    sections_str = "{" + match.group(0) + "}"
                    # Simple check: if all values are false
                    if sections_str.count("false") >= 4 and sections_str.count("true") == 0:
                        return False
            except Exception:
                pass

        return True
    except Exception:
        return True  # If error, include the file


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


def get_myservice_stats(es, index):
    """Get myservice document statistics."""
    try:
        result = es.search(
            index=index,
            body={
                "query": {"term": {"doc_type.keyword": "myservice"}},
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


def get_recent_documents(es, index, limit=3):
    """Get recent myservice documents to show original content."""
    try:
        result = es.search(
            index=index,
            body={
                "query": {"term": {"doc_type.keyword": "myservice"}},
                "size": limit,
                "sort": [{"created_at": "desc"}],
                "_source": ["doc_id", "chapter", "content", "device_name"]
            }
        )
        return result["hits"]["hits"]
    except Exception:
        return []


def print_progress_stats(es, index, success, failed, processed, total, elapsed):
    """Print detailed progress statistics."""
    print("\n" + "=" * 80)
    print(f"📊 PROGRESS UPDATE: {processed}/{total} files ({processed/total*100:.1f}%)")
    print("=" * 80)

    # Processing stats
    print(f"✓ Success:       {success:>6} files")
    print(f"✗ Failed:        {failed:>6} files")
    print(f"⏱️  Elapsed:       {elapsed:.1f}s")

    if processed > 0:
        avg_time = elapsed / processed
        remaining = total - processed
        eta = avg_time * remaining
        print(f"⚡ Avg time:      {avg_time:.2f}s/file")
        print(f"⏳ ETA:           {eta:.0f}s (~{eta/60:.1f}min)")

    # ES stats
    print("\n" + "-" * 80)
    print("🗄️  ELASTICSEARCH STATUS")
    print("-" * 80)

    es_stats = get_es_stats(es, index)
    print(f"Total docs:      {es_stats['doc_count']:>6}")
    print(f"Index size:      {es_stats['size_mb']:>6.2f} MB")

    myservice_stats = get_myservice_stats(es, index)
    print(f"\nMyservice docs:  {myservice_stats['total']:>6}")
    if myservice_stats['chapters']:
        print("Sections breakdown:")
        for chapter, count in sorted(myservice_stats['chapters'].items()):
            print(f"  - {chapter:>10}: {count:>6}")

    # Show recent documents (original content only)
    print("\n" + "-" * 80)
    print("📄 Recent Documents (Original Content)")
    print("-" * 80)

    recent_docs = get_recent_documents(es, index, limit=2)
    if recent_docs:
        for i, hit in enumerate(recent_docs, 1):
            source = hit["_source"]
            content = source.get("content", "")
            content_preview = content[:200] if len(content) > 200 else content

            print(f"\n[{i}] Doc: {source.get('doc_id', 'N/A')} | Section: {source.get('chapter', 'N/A')}")
            print(f"Device: {source.get('device_name', 'N/A')}")
            print(f"Content: {content_preview}{'...' if len(content) > 200 else ''}")
    else:
        print("No documents found yet.")

    print("=" * 80 + "\n")


def ingest_single_file(txt_file, service):
    """Ingest a single txt file. Returns (success, doc_id, error_msg)."""
    try:
        result = service.ingest_myservice_txt(
            file=str(txt_file),
            lang="ko",
            tags=["myservice", "batch"],
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
    """Batch ingest all txt files."""
    # Source directory
    source_dir = Path("/home/llm-share/datasets/pe_agent_data/pe_preprocess_data/myservice_txt")

    if not source_dir.exists():
        print(f"Error: Directory not found: {source_dir}")
        sys.exit(1)

    # Get all txt files
    all_txt_files = sorted(source_dir.glob("*.txt"))
    print(f"Found {len(all_txt_files)} txt files")

    # Filter out empty files
    print("Filtering empty files...")
    txt_files = [f for f in tqdm(all_txt_files, desc="Filtering") if has_content(f)]

    total_files = len(txt_files)
    skipped_files = len(all_txt_files) - total_files

    if total_files == 0:
        print(f"No valid txt files found (all empty)")
        sys.exit(0)

    print(f"\n✓ Valid files: {total_files}")
    print(f"⊘ Skipped (empty): {skipped_files}")
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
    print("=" * 80)

    # Batch processing
    success_count = 0
    failed_count = 0
    failed_files = []
    section_stats = defaultdict(int)

    start_time = time.time()
    report_interval = 100  # Report every 100 files
    max_workers = 8  # Number of parallel workers

    print(f"\nProcessing {total_files} files with {max_workers} workers...")
    print("=" * 80)

    # Parallel processing with ThreadPoolExecutor
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all tasks
        future_to_file = {
            executor.submit(ingest_single_file, txt_file, service): txt_file
            for txt_file in txt_files
        }

        # Process completed tasks with progress bar
        with tqdm(total=total_files, desc="Ingesting", unit="file") as pbar:
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
                            tqdm.write(f"✗ Failed: {doc_id} - {error_msg}")

                except Exception as e:
                    doc_id = txt_file.stem
                    failed_count += 1
                    failed_files.append((doc_id, str(e)))
                    tqdm.write(f"✗ Exception: {doc_id} - {e}")

                pbar.update(1)

                # Periodic progress report
                if processed % report_interval == 0 or processed == total_files:
                    elapsed = time.time() - start_time
                    print_progress_stats(es, index, success_count, failed_count, processed, total_files, elapsed)

    # Refresh index once at the end
    print("\nRefreshing index...")
    service.es.indices.refresh(index=index)

    # Final summary
    total_time = time.time() - start_time
    print("\n" + "=" * 80)
    print("🎉 BATCH INGESTION COMPLETE")
    print("=" * 80)
    print(f"Total files:     {total_files}")
    print(f"✓ Success:       {success_count}")
    print(f"✗ Failed:        {failed_count}")
    print(f"⏱️  Total time:    {total_time:.1f}s ({total_time/60:.1f}min)")
    print(f"⚡ Avg speed:     {total_time/total_files:.2f}s/file")

    # Final ES stats
    print("\n" + "-" * 80)
    print("📊 Final ES Statistics:")
    print("-" * 80)
    final_stats = get_es_stats(es, index)
    print(f"Total docs:      {final_stats['doc_count']}")
    print(f"Index size:      {final_stats['size_mb']:.2f} MB")
    print(f"Docs added:      {final_stats['doc_count'] - initial_stats['doc_count']}")
    print(f"Size increased:  {final_stats['size_mb'] - initial_stats['size_mb']:.2f} MB")

    myservice_stats = get_myservice_stats(es, index)
    print(f"\nMyservice docs:  {myservice_stats['total']}")
    if myservice_stats['chapters']:
        print("Sections breakdown:")
        for chapter, count in sorted(myservice_stats['chapters'].items()):
            print(f"  - {chapter:>10}: {count:>6}")

    # Sample documents with LLM analysis
    print("\n" + "-" * 80)
    print("📄 Sample Documents (with LLM Analysis)")
    print("-" * 80)

    try:
        sample_result = es.search(
            index=index,
            body={
                "query": {"term": {"doc_type.keyword": "myservice"}},
                "size": 2,
                "sort": [{"created_at": "desc"}],
                "_source": ["doc_id", "chapter", "content", "chunk_summary", "chunk_keywords", "device_name"]
            }
        )

        for i, hit in enumerate(sample_result["hits"]["hits"], 1):
            source = hit["_source"]
            content = source.get("content", "")[:150]
            print(f"\n[Sample {i}]")
            print(f"Doc ID:   {source.get('doc_id', 'N/A')}")
            print(f"Section:  {source.get('chapter', 'N/A')}")
            print(f"Device:   {source.get('device_name', 'N/A')}")
            print(f"Content:  {content}...")
            if source.get('chunk_summary'):
                print(f"Summary:  {source['chunk_summary']}")
            if source.get('chunk_keywords'):
                keywords = source['chunk_keywords'][:8]
                print(f"Keywords: {', '.join(keywords)}")
    except Exception as e:
        print(f"Could not fetch sample documents: {e}")

    if failed_files:
        print("\n" + "-" * 80)
        print("⚠️  Failed files:")
        for doc_id, error in failed_files[:10]:
            print(f"  - {doc_id}: {error}")
        if len(failed_files) > 10:
            print(f"  ... and {len(failed_files) - 10} more")

    print("=" * 80)


if __name__ == "__main__":
    main()
