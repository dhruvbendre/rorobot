"""
The archive's retrieval pipeline.

    loader       markdown documents + frontmatter metadata  ->  Document
    chunker      logical sections, not fixed windows        ->  Chunk
    embeddings   pluggable text -> vector providers
    vector_store numpy cosine index, persisted to disk
    retriever    top-k with a grounding threshold           ->  Hit
    prompts      the archive's voice and grounding rules
    llm          pluggable chat providers (openai / anthropic / none)
    pipeline     Archive: load, index, ask
"""
from .pipeline import Archive, Answer

__all__ = ["Archive", "Answer"]
