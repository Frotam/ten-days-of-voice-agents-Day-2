import asyncio
import os
import json
from src.agent import Assistant


async def run_checkin_simulation():
    # Read last entry if available
    filename = os.path.join(os.getcwd(), "backend", "wellness_log.json")
    last_entry = {}
    if os.path.exists(filename):
        try:
            with open(filename, "r", encoding="utf-8") as f:
                data = json.load(f)
                if isinstance(data, list) and data:
                    last_entry = data[-1]
        except Exception:
            last_entry = {}

    assistant = Assistant(last_entry=last_entry)

    # Simulate filling fields one at a time (no write until confirm)
    print(await assistant.update_order(None, "mood", "a bit low"))
    print(await assistant.update_order(None, "energy", "low"))
    print(await assistant.update_order(None, "stressors", "deadline at work"))
    print(await assistant.update_order(None, "objectives", "finish report, take 10-minute walk"))

    # Now finalize (confirm) and save
    result = await assistant.finalize_checkin(None, confirm=True)
    print(result)


if __name__ == "__main__":
    asyncio.run(run_checkin_simulation())
