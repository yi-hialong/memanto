"""
Example: Using MemantoSaver with LangGraph.

Prerequisites:
1. Install dependencies:
   pip install -r requirements.txt

2. Get a Moorcheh API key from https://console.moorcheh.ai/api-keys

3. Create and activate an agent:
   memanto agent create my-langgraph-agent
   memanto agent activate my-langgraph-agent
   # Copy the session token from output

4. Set environment variable:
   export MOORCHEH_API_KEY="your-api-key"

5. Start the Memanto server:
   memanto serve
   # Keeps running in background
"""

import os

from langgraph.graph import StateGraph, START, END
from memanto import MemantoSaver


# Define state schema
class AgentState(dict):
    messages: list


# Simple example agent
def route_after_response(state: AgentState):
    messages = state.get("messages", [])
    if messages and "bye" in messages[-1].get("content", "").lower():
        return END
    return "respond"


def respond(state: AgentState):
    user_msg = state["messages"][-1]["content"]
    return {
        "messages": [
            {"role": "assistant", "content": f"Echo: {user_msg}"}
        ]
    }


def build_agent():
    builder = StateGraph(AgentState)
    builder.add_node("respond", respond)
    builder.add_edge(START, "respond")
    builder.add_conditional_edges("respond", route_after_response)

    # Initialize Memanto checkpointer
    checkpointer = MemantoSaver(
        moorcheh_api_key=os.environ.get("MOORCHEH_API_KEY"),
        agent_id="my-langgraph-agent",
        session_token=os.environ.get("MEMANTO_SESSION_TOKEN"),
        # For local Memanto server (default):
        # base_url="http://127.0.0.1:8000",
        # For Moorcheh Cloud (if available):
        # base_url="https://api.moorcheh.ai",
    )

    return builder.compile(checkpointer=checkpointer)


if __name__ == "__main__":
    graph = build_agent()

    # Run with thread_id for state persistence
    config = {"configurable": {"thread_id": "user-123-session-1"}}

    # First interaction
    result1 = graph.invoke(
        {"messages": [{"role": "user", "content": "Hello, remember me!"}]},
        config=config,
    )
    print("First response:", result1["messages"][-1])

    # Second interaction (state persists across sessions)
    result2 = graph.invoke(
        {"messages": [{"role": "user", "content": "What did I say earlier?"}]},
        config=config,
    )
    print("Second response:", result2["messages"][-1])

    # List all checkpoints for this thread
    checkpoints = list(graph.checkpointer.list(config))
    print(f"Total checkpoints: {len(checkpoints)}")
