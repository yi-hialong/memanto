from langgraph.checkpoint.base import BaseCheckpointSaver

class MemantoSaver(BaseCheckpointSaver):
    """Checkpointer que guarda estado de agentes en Memanto."""
    
    def __init__(self, client):
        super().__init__()
        self.client = client

    def put(self, config, checkpoint, metadata):
        thread_id = config["configurable"]["thread_id"]
        self.client.store(key=f"session:{thread_id}", value=checkpoint)
        return {"configurable": {"thread_id": thread_id, "thread_ts": checkpoint["ts"]}}

    def get_tuple(self, config):
        thread_id = config["configurable"]["thread_id"]
        data = self.client.retrieve(key=f"session:{thread_id}")
        if data:
            return (config, data, {})
        return None
