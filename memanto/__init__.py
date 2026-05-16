"""
MemantoSaver: LangGraph checkpointer backed by Memanto's long-term memory system.

Quick start:
    pip install -r requirements.txt
    memanto agent create my-agent
    memanto agent activate my-agent
    # Then use:
    from memanto import MemantoSaver

Docs: https://docs.memanto.ai
"""

from .memanto_saver import MemantoSaver, MemantoMemoryAPI, CheckpointSerializer

__all__ = ["MemantoSaver", "MemantoMemoryAPI", "CheckpointSerializer"]
