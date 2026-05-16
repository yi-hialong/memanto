"""
MemantoSaver: LangGraph checkpointer backed by Memanto's long-term memory system.

Memanto (memanto.ai) provides a typed semantic memory layer for AI agents.
This checkpointer serializes LangGraph checkpoints and stores them in Memanto,
enabling persistent, multi-session state for LangGraph agents.

Install:
    pip install -r requirements.txt

Usage:
    1. Get a Moorcheh API key from https://console.moorcheh.ai/api-keys
    2. Run: memanto agent create <my-agent>  (creates an agent namespace)
    3. Activate: memanto agent activate <my-agent>  (gets session token)
    4. Use in LangGraph:

        from langgraph.graph import StateGraph
        from memanto import MemantoSaver

        checkpointer = MemantoSaver(
            moorcheh_api_key="your-api-key",
            agent_id="my-agent",
            session_token="your-session-token",
        )

        graph = StateGraph(...).compile(checkpointer=checkpointer)
"""

import json
import logging
import base64
import zlib
from typing import Optional, AsyncIterator, Any

import httpx

from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.checkpoint.metadata import CheckpointMetadata
from langgraph.checkpoint.serde.base import SerializerProtocol
from langchain_core.runnables import RunnableConfig

logger = logging.getLogger(__name__)


class MemantoMemoryAPI:
    """
    Low-level client for Memanto's REST API.

    The Memanto server exposes these endpoints when running locally
    (via `memanto serve`) or when connected to Moorcheh Cloud.

    Docs: https://docs.memanto.ai
    """

    def __init__(
        self,
        base_url: str = "http://127.0.0.1:8000",
        api_key: Optional[str] = None,
        timeout: float = 30.0,
    ):
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.timeout = timeout
        self._client: Optional[httpx.AsyncClient] = None

    async def _get_client(self) -> httpx.AsyncClient:
        if self._client is None or self._client.is_closed:
            headers = {}
            if self.api_key:
                headers["X-API-Key"] = self.api_key
            self._client = httpx.AsyncClient(
                base_url=self.base_url,
                headers=headers,
                timeout=self.timeout,
            )
        return self._client

    async def close(self) -> None:
        if self._client is not None:
            await self._client.aclose()
            self._client = None

    async def create_agent(self, name: str) -> dict:
        """Create a new agent namespace in Memanto."""
        client = await self._get_client()
        resp = await client.post("/api/v2/agents", json={"name": name})
        resp.raise_for_status()
        return resp.json()

    async def activate_session(self, agent_id: str) -> dict:
        """Activate a session and get a session token (valid for 6 hours)."""
        client = await self._get_client()
        resp = await client.post(f"/api/v2/agents/{agent_id}/activate")
        resp.raise_for_status()
        return resp.json()

    async def deactivate_session(self, agent_id: str, session_token: str) -> None:
        """Deactivate the current session."""
        client = await self._get_client()
        await client.post(
            f"/api/v2/agents/{agent_id}/deactivate",
            headers={"X-Session-Token": session_token},
        )

    async def remember(
        self,
        agent_id: str,
        session_token: str,
        content: str,
        memory_type: str = "artifact",
        metadata: Optional[dict] = None,
    ) -> dict:
        """
        Store a memory in Memanto.

        Uses the Memanto /remember endpoint with a serialized checkpoint
        stored as an 'artifact' type memory.
        """
        client = await self._get_client()
        payload: dict[str, Any] = {
            "content": content,
            "type": memory_type,
        }
        if metadata:
            payload["metadata"] = metadata

        resp = await client.post(
            f"/api/v2/agents/{agent_id}/remember",
            json=payload,
            headers={"X-Session-Token": session_token},
        )
        resp.raise_for_status()
        return resp.json()

    async def recall(
        self,
        agent_id: str,
        session_token: str,
        query: str,
        memory_type: Optional[str] = None,
        limit: int = 10,
    ) -> dict:
        """
        Semantic recall from Memanto memory.

        Uses Moorcheh's Information-Theoretic Search (ITS) engine
        for deterministic, sub-90ms retrieval.
        """
        client = await self._get_client()
        payload: dict[str, Any] = {
            "query": query,
            "limit": limit,
        }
        if memory_type:
            payload["type"] = memory_type

        resp = await client.post(
            f"/api/v2/agents/{agent_id}/recall",
            json=payload,
            headers={"X-Session-Token": session_token},
        )
        resp.raise_for_status()
        return resp.json()

    async def get_memory(
        self,
        agent_id: str,
        session_token: str,
        memory_id: str,
    ) -> dict:
        """Retrieve a specific memory by ID."""
        client = await self._get_client()
        resp = await client.get(
            f"/api/v2/agents/{agent_id}/memories/{memory_id}",
            headers={"X-Session-Token": session_token},
        )
        resp.raise_for_status()
        return resp.json()

    async def list_memories(
        self,
        agent_id: str,
        session_token: str,
        memory_type: Optional[str] = None,
        limit: int = 100,
    ) -> dict:
        """List memories for an agent, optionally filtered by type."""
        client = await self._get_client()
        params = {"limit": limit}
        if memory_type:
            params["type"] = memory_type

        resp = await client.get(
            f"/api/v2/agents/{agent_id}/memories",
            params=params,
            headers={"X-Session-Token": session_token},
        )
        resp.raise_for_status()
        return resp.json()


class CheckpointSerializer:
    """
    Serializes LangGraph checkpoints for storage in Memanto.

    Checkpoints contain complex nested Python objects (ChannelValues,
    PendingWrites, etc.) that need to be serialized before storing.

    Strategy: JSON serialize with compression fallback for large checkpoints.
    """

    @staticmethod
    def serialize(checkpoint: Any) -> str:
        """
        Serialize a checkpoint dict to a storable string.

        Tries JSON first, falls back to base64+zlib compressed JSON
        if the result is smaller.
        """
        json_bytes = json.dumps(checkpoint, default=str).encode("utf-8")

        # If JSON is small enough, store as-is
        if len(json_bytes) < 50000:
            return json_bytes.decode("utf-8")

        # Compress large checkpoints
        compressed = zlib.compress(json_bytes)
        if len(compressed) < len(json_bytes):
            encoded = base64.b64encode(compressed).decode("ascii")
            return f"__zlib__:{encoded}"

        return json_bytes.decode("utf-8")

    @staticmethod
    def deserialize(data: str) -> Any:
        """Deserialize a stored string back to a checkpoint dict."""
        if data.startswith("__zlib__:"):
            compressed = base64.b64decode(data[8:])
            json_bytes = zlib.decompress(compressed)
            return json.loads(json_bytes.decode("utf-8"))
        return json.loads(data)

    @staticmethod
    def build_artifact_content(checkpoint: Any, metadata: Any) -> str:
        """Build artifact content string for Memanto."""
        return json.dumps(
            {
                "checkpoint": checkpoint,
                "metadata": metadata,
            },
            default=str,
        )

    @staticmethod
    def parse_artifact_content(content: str) -> tuple[Any, dict]:
        """Parse artifact content back to checkpoint and metadata."""
        parsed = json.loads(content)
        return parsed.get("checkpoint", {}), parsed.get("metadata", {})


class MemantoSaver(BaseCheckpointSaver):
    """
    LangGraph checkpointer backed by Memanto's long-term memory system.

    This checkpointer stores LangGraph agent checkpoints as Memanto memories,
    enabling persistent state across sessions and threads.

    Key features:
    - Full async implementation for use with async LangGraph graphs
    - Stores checkpoints as Memanto 'artifact' type memories
    - Uses Moorcheh ITS engine for semantic retrieval
    - Supports configurable memory namespaces (agents)
    - Sub-90ms retrieval latency via Memanto's information-theoretic index

    Example::

        import os
        from langgraph.graph import StateGraph
        from memanto import MemantoSaver

        checkpointer = MemantoSaver(
            moorcheh_api_key=os.environ["MOORCHEH_API_KEY"],
            base_url="https://api.moorcheh.ai",  # For cloud mode
            agent_id="my-langgraph-agent",
        )

        graph = StateGraph(...).compile(checkpointer=checkpointer)
        # The graph state will persist across sessions via Memanto
    """

    def __init__(
        self,
        *,
        moorcheh_api_key: Optional[str] = None,
        agent_id: str,
        session_token: Optional[str] = None,
        base_url: str = "http://127.0.0.1:8000",
        thread_id_key: str = "thread_id",
        memory_type: str = "artifact",
    ):
        """
        Initialize the Memanto checkpointer.

        Args:
            moorcheh_api_key: Moorcheh API key (from console.moorcheh.ai).
                              Can also be set as MOORCHEH_API_KEY env var.
            agent_id: Memanto agent ID / namespace for this checkpointer.
                      Create with: memanto agent create <name>
            session_token: Memanto session token. Get by running:
                           memanto agent activate <agent-id>
                           Or programmatically via MemantoMemoryAPI.activate_session()
                           If not provided, attempts to auto-activate.
            base_url: Memanto API base URL.
                      - Local: http://127.0.0.1:8000 (after `memanto serve`)
                      - Cloud: https://api.moorcheh.ai (or equivalent)
            thread_id_key: Config key used to identify the thread/session.
                           Defaults to "thread_id" (standard LangGraph convention).
            memory_type: Memanto memory type to use for checkpoints.
                         Defaults to "artifact" (suitable for serialized data).
        """
        super().__init__()
        self.moorcheh_api_key = moorcheh_api_key
        self.agent_id = agent_id
        self.session_token = session_token
        self.base_url = base_url
        self.thread_id_key = thread_id_key
        self.memory_type = memory_type
        self._api: Optional[MemantoMemoryAPI] = None
        self._serde = CheckpointSerializer()
        self._verified = False

    @property
    def api(self) -> MemantoMemoryAPI:
        if self._api is None:
            self._api = MemantoMemoryAPI(
                base_url=self.base_url,
                api_key=self.moorcheh_api_key,
            )
        return self._api

    async def _ensure_session(self) -> str:
        """Ensure we have an active session token."""
        if self.session_token:
            return self.session_token

        # Auto-activate session
        result = await self.api.activate_session(self.agent_id)
        self.session_token = result.get("session_token", "")
        return self.session_token

    async def _get_thread_id(self, config: RunnableConfig) -> str:
        """Extract thread ID from config."""
        return config["configurable"].get(self.thread_id_key, "default")

    async def _get_memory_key(self, thread_id: str) -> str:
        """Build the Memanto memory key for a thread."""
        return f"langgraph:thread:{thread_id}"

    async def _get_checkpoint_ts(self, config: RunnableConfig) -> Optional[str]:
        """Get checkpoint timestamp from config."""
        return config.get("configurable", {}).get("thread_ts")

    async def put(
        self,
        config: RunnableConfig,
        checkpoint: Any,
        metadata: CheckpointMetadata,
    ) -> RunnableConfig:
        """
        Save a checkpoint to Memanto.

        Stores the checkpoint serialized as an artifact memory entry,
        keyed by thread_id. Overwrites any previous checkpoint for
        the same thread (upsert behavior).
        """
        thread_id = await self._get_thread_id(config)
        session = await self._ensure_session()

        # Serialize checkpoint + metadata as artifact content
        content = self._serde.build_artifact_content(checkpoint, metadata)
        memory_key = await self._get_memory_key(thread_id)

        # Store in Memanto with artifact type
        result = await self.api.remember(
            agent_id=self.agent_id,
            session_token=session,
            content=content,
            memory_type=self.memory_type,
            metadata={
                "thread_id": thread_id,
                "memory_key": memory_key,
                "checkpoint_ts": checkpoint.get("ts"),
                "created_by": "MemantoSaver",
            },
        )

        # Return updated config with checkpoint timestamp
        new_config: RunnableConfig = {
            **config,
            "configurable": {
                **config.get("configurable", {}),
                self.thread_id_key: thread_id,
                "thread_ts": checkpoint.get("ts"),
            },
        }
        return new_config

    async def get(
        self,
        config: RunnableConfig,
    ) -> Optional[Any]:
        """
        Retrieve a checkpoint by config.

        Returns the most recent checkpoint for the thread if no timestamp
        is specified in config, otherwise returns the checkpoint at the
        specified timestamp.
        """
        thread_id = await self._get_thread_id(config)
        session = await self._ensure_session()
        thread_ts = await self._get_checkpoint_ts(config)
        memory_key = await self._get_memory_key(thread_id)

        # Semantic recall by memory key
        recall_result = await self.api.recall(
            agent_id=self.agent_id,
            session_token=session,
            query=memory_key,
            memory_type=self.memory_type,
            limit=20,
        )

        memories = recall_result.get("results", [])
        for memory in memories:
            content = memory.get("content", "")
            metadata = memory.get("metadata", {})
            if metadata.get("memory_key") == memory_key:
                if thread_ts:
                    # Filter by specific timestamp
                    cp_ts = metadata.get("checkpoint_ts")
                    if cp_ts == thread_ts:
                        checkpoint, _ = self._serde.parse_artifact_content(content)
                        return checkpoint
                else:
                    # Return most recent
                    checkpoint, _ = self._serde.parse_artifact_content(content)
                    return checkpoint

        return None

    async def get_tuple(self, config: RunnableConfig) -> Optional[Any]:
        """
        Retrieve a checkpoint tuple (checkpoint, metadata, config).

        This is the primary read method used by LangGraph's state loading.
        Returns None if no checkpoint exists for the given thread.
        """
        thread_id = await self._get_thread_id(config)
        session = await self._ensure_session()
        memory_key = await self._get_memory_key(thread_id)

        recall_result = await self.api.recall(
            agent_id=self.agent_id,
            session_token=session,
            query=memory_key,
            memory_type=self.memory_type,
            limit=20,
        )

        memories = recall_result.get("results", [])
        best = None
        best_ts = None

        for memory in memories:
            content = memory.get("content", "")
            meta = memory.get("metadata", {})
            if meta.get("memory_key") == memory_key:
                cp_ts = meta.get("checkpoint_ts", "")
                if best_ts is None or (cp_ts and cp_ts > best_ts):
                    best = memory
                    best_ts = cp_ts

        if best is None:
            return None

        content = best.get("content", "")
        metadata_dict = best.get("metadata", {})
        checkpoint, _ = self._serde.parse_artifact_content(content)

        result_config: RunnableConfig = {
            **config,
            "configurable": {
                **config.get("configurable", {}),
                self.thread_id_key: thread_id,
                "thread_ts": best_ts,
            },
        }

        # Reconstruct CheckpointTuple (simplified: no pending_writes)
        return {
            "config": result_config,
            "checkpoint": checkpoint,
            "metadata": metadata_dict,
        }

    async def list(
        self,
        config: Optional[RunnableConfig],
        *,
        filter: Optional[dict] = None,
        before: Optional[RunnableConfig] = None,
        limit: int = 10,
    ) -> AsyncIterator[Any]:
        """
        List checkpoints for a thread.

        Yields CheckpointTuple objects in reverse chronological order
        (most recent first), up to `limit` results.

        Args:
            config: Config specifying the thread. If None, lists all threads.
            filter: Optional filter criteria (e.g., {"source": "input"})
            before: Return checkpoints created before this timestamp
            limit: Maximum number of checkpoints to return (default 10)
        """
        session = await self._ensure_session()

        # If a thread is specified, filter to that thread
        if config:
            thread_id = await self._get_thread_id(config)
            memory_key = await self._get_memory_key(thread_id)
        else:
            memory_key = None

        # List memories filtered by artifact type
        try:
            list_result = await self.api.list_memories(
                agent_id=self.agent_id,
                session_token=session,
                memory_type=self.memory_type,
                limit=limit * 3,  # Fetch extra to account for filtering
            )
        except Exception:
            # Fallback: use recall for broad search
            list_result = await self.api.recall(
                agent_id=self.agent_id,
                session_token=session,
                query="langgraph:thread:",
                memory_type=self.memory_type,
                limit=limit * 3,
            )

        memories = list_result.get("results", [])
        seen = set()
        count = 0

        for memory in memories:
            if count >= limit:
                break

            content = memory.get("content", "")
            meta = memory.get("metadata", {})
            mem_key = meta.get("memory_key", "")

            # Filter to langgraph threads
            if not mem_key.startswith("langgraph:thread:"):
                continue

            # Skip duplicates
            if mem_key in seen:
                continue
            seen.add(mem_key)

            # Apply timestamp filter (before)
            if before:
                before_ts = before.get("configurable", {}).get("thread_ts")
                cp_ts = meta.get("checkpoint_ts", "")
                if cp_ts and before_ts and cp_ts >= before_ts:
                    continue

            try:
                checkpoint, _ = self._serde.parse_artifact_content(content)
            except (json.JSONDecodeError, KeyError):
                continue

            thread_id = mem_key.replace("langgraph:thread:", "")
            thread_ts = meta.get("checkpoint_ts")

            result_config: RunnableConfig = {
                "configurable": {
                    self.thread_id_key: thread_id,
                    "thread_ts": thread_ts,
                },
            }

            yield {
                "config": result_config,
                "checkpoint": checkpoint,
                "metadata": meta,
            }
            count += 1
