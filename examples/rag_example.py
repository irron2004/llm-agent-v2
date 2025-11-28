"""Example: Complete RAG pipeline usage."""

from pathlib import Path
import sys

# Add backend to path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from backend.services import (
    DocumentIndexService,
    SourceDocument,
    RAGService,
)


def main():
    """Demonstrate RAG service usage."""

    # 1. Prepare sample documents
    documents = [
        SourceDocument(
            doc_id="pm1",
            text="PM2-1 chamber experienced exhaust alarm at 2024-01-15. "
                 "The issue was caused by a leak in the helium line. "
                 "Resolution: Replace helium line seal.",
        ),
        SourceDocument(
            doc_id="pm2",
            text="PM3-2 temperature exceeded normal range. "
                 "Cooling system malfunction detected. "
                 "Action taken: Clean cooling fins and replace thermal paste.",
        ),
        SourceDocument(
            doc_id="pm3",
            text="Pressure sensor reading anomaly in PM1-3. "
                 "Sensor calibration drift observed. "
                 "Solution: Recalibrate pressure sensor and update baseline.",
        ),
    ]

    # 2. Index documents
    print("📚 Indexing documents...")
    indexer = DocumentIndexService.from_settings()
    corpus = indexer.index(
        documents,
        preprocess=True,
        build_sparse=True,
        persist_dir="data/vector_stores/example",
    )
    print(f"   Indexed {len(corpus.documents)} documents\n")

    # 3. Create RAG service
    print("🤖 Initializing RAG service...")
    rag_service = RAGService.from_settings(corpus)
    print("   Ready!\n")

    # 4. Query examples
    queries = [
        "What caused the exhaust alarm in PM2-1?",
        "How do you fix temperature issues?",
        "Tell me about pressure sensor problems.",
    ]

    for i, question in enumerate(queries, 1):
        print(f"❓ Question {i}: {question}")

        # Execute RAG query
        response = rag_service.query(question, top_k=2)

        print(f"💡 Answer: {response.answer}\n")
        print(f"   📖 Retrieved {len(response.context)} documents:")
        for j, ctx in enumerate(response.context, 1):
            print(f"      [{j}] (score: {ctx.score:.3f}) {ctx.content[:80]}...")
        print()

    # 5. Simple query interface
    print("🎯 Simple query interface:")
    answer = rag_service.query_simple("What maintenance actions are needed?")
    print(f"   Answer: {answer}\n")


if __name__ == "__main__":
    main()
