=== AGENTS.md ===

# ACG Protocol

Audited Context Generation (ACG) Protocol — a dual-layer standard for veracity assurance in RAG systems. Python-based project for grounding facts and verifying reasoning.

## Directory Structure

  README.md
  Usage.md
  requirements.txt
  AGENTS.md
  docs/
    ACG_PROTOCOL.md       — Core ACG protocol specification
    UGVP_PROTOCOL.md       — Universal Grounding & Verification Protocol (Layer 1)
    RSVP_PROTOCOL.md       — Reasoning & Synthesis Verification Protocol (Layer 2)
  src/
    agent.py               — Main agent entry point / inference
    config.py              — Configuration
    evaluation_data.py     — Evaluation data helpers
    indexer.py             — Chunk + SHI indexer for URLs
    mongodb_client.py      — MongoDB client for store/retrieve
    ragas_evaluator.py     — RAGAS evaluation integration
    ugvp_protocol.py       — UGVP protocol implementation
    utils.py               — Utility functions

## Stack

Python 3.12+, MongoDB, aiohttp, google-generativeai, RAGAS

## Commands

Setup:    python -m venv venv && source venv/bin/activate && pip install -r requirements.txt
Index:    python src/indexer.py --sentences_per_chunk 5 <url>
Agent:    python src/agent.py
Tests:    pytest (if configured)
