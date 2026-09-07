"""
Build (or rebuild) the archive's index from the knowledge folder.

    python ingest.py            rebuild with the configured providers
    python ingest.py --check    show what would be indexed, without embedding

The Streamlit app also builds the index on first start, so running this is
optional; it is useful in CI and to see which documents are still placeholders.
"""
from __future__ import annotations

import argparse

from config.settings import load_settings
from rag.chunker import chunk_documents
from rag.loader import load_documents
from rag.pipeline import Archive


def main() -> None:
    parser = argparse.ArgumentParser(description="Index the archive's knowledge folder.")
    parser.add_argument("--check", action="store_true", help="list documents and chunk counts without embedding")
    parser.add_argument("--test-data", action="store_true", help="include knowledge/_test_data (TEST_DATA_ONLY)")
    args = parser.parse_args()

    settings = load_settings({"include_test_data": True} if args.test_data else None)
    docs = load_documents(settings.knowledge_dir, include_test_data=settings.include_test_data)

    print(f"Knowledge folder: {settings.knowledge_dir}")
    for doc in docs:
        state = "placeholder" if doc.placeholder else ("TEST_DATA_ONLY" if doc.test_data else "live")
        print(f"  {doc.path.name:<28} {state:<15} {doc.metadata.get('category', '')}")

    if args.check:
        chunks = chunk_documents(docs)
        print(f"\n{len(chunks)} chunk(s) would be indexed (placeholders excluded).")
        for chunk in chunks:
            print(f"  {chunk.id:<24} {chunk.breadcrumb}")
        return

    archive = Archive(settings)
    status = archive.rebuild()
    print(
        f"\nIndexed {status.chunks} chunk(s) from {status.live_documents} live document(s) "
        f"({status.placeholders} placeholder(s) skipped) with {status.embedding}."
    )
    print(f"Index written to {settings.index_dir}")


if __name__ == "__main__":
    main()
