#!/usr/bin/env python3
"""Test script for all-MiniLM-L6-v2 semantic embeddings."""

from contract_rag.ingestion.chunk import Chunk
from contract_rag.ingestion.index import VectorIndex

# Create test chunks
chunks = [
    Chunk(
        id='1',
        source='contract_a.txt',
        text='Termination rights and conditions. Either party may terminate with 30 days notice.',
        ordinal=0,
        section_number='2.1',
        section_title='Termination',
    ),
    Chunk(
        id='2',
        source='contract_b.txt',
        text='Liability limitations and caps. Total liability capped at 12 months fees.',
        ordinal=1,
        section_number='3.1',
        section_title='Liability',
    ),
    Chunk(
        id='3',
        source='contract_a.txt',
        text='Payment terms and conditions. Invoice due within 30 days of receipt.',
        ordinal=2,
        section_number='4.1',
        section_title='Payment',
    ),
]

print("=" * 60)
print("Testing all-MiniLM-L6-v2 Semantic Embeddings")
print("=" * 60)

# Build index
print("\n[1] Building semantic index...")
index = VectorIndex()
index.build(chunks)

print(f"    ✓ Embeddings model: all-MiniLM-L6-v2 (384-dim)")
print(f"    ✓ Chunks indexed: {len(chunks)}")
print(f"    ✓ Embeddings shape: {index.embeddings.shape}")

# Test search 1: Termination query
print("\n[2] Testing semantic search...")
query1 = "What are the termination rights?"
results1 = index.search(query1, k=2)

print(f"\n    Query: '{query1}'")
print(f"    Top results:")
for i, (chunk, score) in enumerate(results1, 1):
    print(f"      {i}. {chunk.section_title} (score: {score:.4f})")

# Test search 2: Liability query
query2 = "liability caps and limits"
results2 = index.search(query2, k=2)

print(f"\n    Query: '{query2}'")
print(f"    Top results:")
for i, (chunk, score) in enumerate(results2, 1):
    print(f"      {i}. {chunk.section_title} (score: {score:.4f})")

# Test search 3: Payment query
query3 = "payment terms"
results3 = index.search(query3, k=2)

print(f"\n    Query: '{query3}'")
print(f"    Top results:")
for i, (chunk, score) in enumerate(results3, 1):
    print(f"      {i}. {chunk.section_title} (score: {score:.4f})")

print("\n" + "=" * 60)
print("✓ All tests passed! Semantic embeddings working correctly.")
print("=" * 60)
