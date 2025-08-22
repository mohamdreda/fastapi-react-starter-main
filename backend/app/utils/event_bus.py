import json
from fastapi import WebSocket
import redis.asyncio as redis

REDIS_URL = "redis://localhost:6379"

async def publish_ws_update(run_id: str, message: dict):
    r = redis.from_url(REDIS_URL, decode_responses=False)
    await r.publish(f"workflow:{run_id}", json.dumps(message))
    await r.close()

async def workflow_ws_listener(websocket: WebSocket, run_id: str):
    r = redis.from_url(REDIS_URL, decode_responses=True)
    pubsub = r.pubsub()
    await pubsub.subscribe(f"workflow:{run_id}")
    try:
        async for msg in pubsub.listen():
            if msg["type"] == "message":
                await websocket.send_text(msg["data"])
    finally:
        await pubsub.unsubscribe(f"workflow:{run_id}")
        await r.close()
