import logging
import os
import json
import datetime
import asyncio
import urllib.request
import urllib.error
# test
from dotenv import load_dotenv
from livekit.agents import (
    Agent,
    AgentSession,
    JobContext,
    JobProcess,
    MetricsCollectedEvent,
    RoomInputOptions,
    WorkerOptions,
    cli,
    metrics,
    tokenize,
    function_tool,
    RunContext,
)
from livekit.plugins import murf, silero, google, deepgram, noise_cancellation
from livekit.plugins.turn_detector.multilingual import MultilingualModel

logger = logging.getLogger("agent")

load_dotenv(".env.local")


class Assistant(Agent):
    def __init__(self, last_entry: dict | None = None) -> None:
        # Build a concise, grounded system prompt that references the last entry if available
        if last_entry:
            # Keep the system prompt deterministic and concise; mention the user's last energy as a cue
            last_energy = last_entry.get("energy", "")
            last_mood = last_entry.get("mood", "")
            last_ref = (
                f'At session start, briefly reference the last check-in: "Last time you said your energy was {last_energy} — how is it today?"'
            )
        else:
            last_ref = 'If no prior check-ins exist, open with a neutral: "Hi — how are you feeling today?"'

        super().__init__(
            instructions=(
                "You are a grounded, supportive health & wellness voice companion.\n"
                "Run a brief daily check-in: ask one clear question at a time. Focus on mood, energy, any stressors, and 1-3 simple intentions for the day.\n"
                "Do NOT give medical diagnoses or make clinical claims — provide compassionate, realistic, non-diagnostic support.\n"
                "If the user is feeling stressed or low over the mood ask why he is feeling sad or stressed depending over his mood"
                "Offer one small, practical suggestion after objectives are set (e.g., take a 5-minute walk, break a task into a 10-minute step, take a short break).\n"
                "At the end, provide a short recap: today's mood summary, the 1-3 objectives, and ask 'Does this sound right?'.\n"
                "Persist completed check-ins to a JSON log file, and reference the last check-in when appropriate.\n"
                + last_ref
            )
        )

        # Initialize a small check-in state that we will fill via tools
        self.checkin_state = {
            "mood": "",
            "energy": "",
            "stressors": "",
            "objectives": [],
            "summary": "",
            "datetime": "",
        }

    @function_tool
    async def update_order(self, context: RunContext, field: str, value: str):
        """Update a specific field of the current check-in.

        Args:
            field: One of 'mood', 'energy', 'stressors', 'objectives', 'summary'
            value: The value to set. For 'objectives', a comma-separated string will be split into a list.

        Returns:
            A short status string describing what changed and any remaining required fields.
        """

        allowed = {"mood", "energy", "stressors", "objectives", "summary"}
        if field not in allowed:
            return f"Unknown field '{field}'. Allowed fields: {', '.join(sorted(allowed))}."

        if field == "objectives":
            objs = [o.strip() for o in value.split(",") if o.strip()]
            if objs:
                # replace objectives with the provided list (intentional)
                self.checkin_state["objectives"] = objs
        else:
            self.checkin_state[field] = value.strip()

        # Required fields for a minimal check-in: mood and at least one objective
        missing = []
        if not self.checkin_state.get("mood"):
            missing.append("mood")
        if not self.checkin_state.get("objectives"):
            missing.append("objectives")

        if not missing:
            # All required fields present in memory; do NOT write to disk yet.
            return (
                f"Updated '{field}'. All required fields present in memory."
                " I will wait for you to confirm the recap before saving."
            )

        return f"Updated '{field}'. Missing required fields: {', '.join(missing)}."

    @function_tool
    async def get_order(self, context: RunContext):
        """Return the current check-in state as JSON-serializable dict."""
        return self.checkin_state

    @function_tool
    async def save_checkin(self, context: RunContext):
        """(Deprecated) kept for compatibility — use `finalize_checkin` instead."""
        return "Use finalize_checkin to confirm and save the check-in."

    @function_tool
    async def finalize_checkin(self, context: RunContext, confirm: bool = True):
        """Finalize the in-memory check-in: validate required fields, generate a summary and a small suggestion,
        and append the completed entry to `backend/wellness_log.json` using an atomic write and a simple lock.

        Args:
            confirm: boolean indicating user confirmation. If False, abort and return a message.

        Returns:
            A confirmation string describing the saved entry or why it was not saved.
        """

        if not confirm:
            return "Check-in confirmation not received; nothing was saved."

        # Validate required fields
        missing = []
        if not self.checkin_state.get("mood"):
            missing.append("mood")
        if not self.checkin_state.get("objectives"):
            missing.append("objectives")
        if missing:
            return f"Cannot finalize — missing required fields: {', '.join(missing)}."

        entry = {
            "datetime": datetime.datetime.now().isoformat(),
            "mood": self.checkin_state.get("mood", ""),
            "energy": self.checkin_state.get("energy", ""),
            "stressors": self.checkin_state.get("stressors", ""),
            "objectives": self.checkin_state.get("objectives", []),
        }

        # Generate summary if not provided
        if not self.checkin_state.get("summary"):
            mood = entry["mood"] or "unspecified mood"
            energy = entry["energy"] or "unspecified energy"
            objs = entry["objectives"]
            obj_text = ", ".join(objs[:3]) if objs else "no objectives"
            entry["summary"] = f"Mood: {mood}; Energy: {energy}; Objectives: {obj_text}."
        else:
            entry["summary"] = self.checkin_state.get("summary")

        # Generate a small, grounded suggestion (non-clinical)
        suggestion = "Consider a short, concrete action: take a 5-minute walk or break a task into a 10-minute step."
        energy_lower = (entry.get("energy") or "").lower()
        if "low" in energy_lower or "tired" in energy_lower:
            suggestion = "Your energy sounds low — a short 5-minute walk or a brief rest could help."
        elif entry.get("stressors"):
            suggestion = "If you're feeling stressed, try a 2-minute grounding breath or a short break."

        entry["suggestion"] = suggestion

        # Append atomically with a simple lock
        filename = os.path.join(os.getcwd(), "backend", "wellness_log.json")
        lockfile = filename + ".lock"
        tmpfile = filename + ".tmp"

        os.makedirs(os.path.dirname(filename), exist_ok=True)

        # Acquire simple lock by creating a lockfile (O_EXCL ensures atomic create)
        start = datetime.datetime.now()
        timeout_sec = 5
        got_lock = False
        while (datetime.datetime.now() - start).total_seconds() < timeout_sec:
            try:
                fd = os.open(lockfile, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
                os.close(fd)
                got_lock = True
                break
            except FileExistsError:
                # Another process likely writing; wait briefly
                await asyncio.sleep(0.05)
        if not got_lock:
            return "Could not acquire file lock to save check-in; please try again shortly."

        try:
            # Read existing data safely
            try:
                if os.path.exists(filename):
                    with open(filename, "r", encoding="utf-8") as f:
                        data = json.load(f)
                        if not isinstance(data, list):
                            data = []
                else:
                    data = []
            except Exception:
                data = []

            data.append(entry)

            # Write to temp file then atomically replace
            try:
                with open(tmpfile, "w", encoding="utf-8") as f:
                    json.dump(data, f, ensure_ascii=False, indent=2)
                os.replace(tmpfile, filename)
            finally:
                if os.path.exists(tmpfile):
                    try:
                        os.remove(tmpfile)
                    except Exception:
                        pass
        finally:
            # Release lock
            try:
                os.remove(lockfile)
            except Exception:
                pass

        # Attempt to push objectives to Notion if integration is configured
        notion_msg = ""
        try:
            notion_token = os.getenv("NOTION_TOKEN")
            notion_page = os.getenv("NOTION_TODO_PAGE_ID")
            if notion_token and notion_page and entry.get("objectives"):
                try:
                    res = await asyncio.to_thread(self._push_objectives_to_notion_sync, entry.get("objectives"))
                    notion_msg = " Notion: " + res
                except Exception as e:
                    notion_msg = f" Notion push failed: {e}"
        except Exception:
            notion_msg = ""

        # Reset in-memory checkin
        self.checkin_state = {
            "mood": "",
            "energy": "",
            "stressors": "",
            "objectives": [],
            "summary": "",
            "datetime": "",
        }

        return f"Check-in saved to {filename}. Summary: {entry['summary']} Suggestion: {suggestion}" + notion_msg

    def _push_objectives_to_notion_sync(self, objectives: list):
        """Synchronous helper to append objectives as to-do blocks to a Notion page.

        Requires environment variables: NOTION_TOKEN and NOTION_TODO_PAGE_ID (a page ID to append to).
        Uses the Notion blocks children append endpoint.
        """

        token = os.getenv("NOTION_TOKEN")
        page_id = os.getenv("NOTION_TODO_PAGE_ID")
        if not token or not page_id:
            raise RuntimeError("NOTION_TOKEN and NOTION_TODO_PAGE_ID must be set to push to Notion.")

        url = f"https://api.notion.com/v1/blocks/{page_id}/children"
        headers = {
            "Authorization": f"Bearer {token}",
            "Notion-Version": "2022-06-28",
            "Content-Type": "application/json",
        }

        children = []
        for obj in objectives:
            children.append({
                "object": "block",
                "type": "to_do",
                "to_do": {
                    "text": [{"type": "text", "text": {"content": obj}}],
                    "checked": False,
                },
            })

        payload = json.dumps({"children": children}).encode("utf-8")

        req = urllib.request.Request(url, data=payload, headers=headers, method="PATCH")
        try:
            with urllib.request.urlopen(req, timeout=10) as resp:
                resp_body = resp.read().decode("utf-8")
                return "Objectives pushed to Notion."
        except urllib.error.HTTPError as e:
            body = e.read().decode("utf-8") if hasattr(e, 'read') else str(e)
            raise RuntimeError(f"Notion API error: {e.code} {body}")
        except Exception as e:
            raise RuntimeError(f"Notion request failed: {e}")

    @function_tool
    async def get_last_checkin(self, context: RunContext):
        """Return the last saved check-in (or an empty dict)."""

        filename = os.path.join(os.getcwd(), "backend", "wellness_log.json")
        if not os.path.exists(filename):
            return {}
        try:
            with open(filename, "r", encoding="utf-8") as f:
                data = json.load(f)
                if isinstance(data, list) and data:
                    return data[-1]
        except Exception:
            return {}
        return {}


def prewarm(proc: JobProcess):
    proc.userdata["vad"] = silero.VAD.load()


async def entrypoint(ctx: JobContext):
    # Logging setup
    # Add any other context you want in all log entries here
    ctx.log_context_fields = {
        "room": ctx.room.name,
    }

    # Set up a voice AI pipeline using OpenAI, Cartesia, AssemblyAI, and the LiveKit turn detector
    session = AgentSession(
        # Speech-to-text (STT) is your agent's ears, turning the user's speech into text that the LLM can understand
        # See all available models at https://docs.livekit.io/agents/models/stt/
        stt=deepgram.STT(model="nova-3"),
        # A Large Language Model (LLM) is your agent's brain, processing user input and generating a response
        # See all available models at https://docs.livekit.io/agents/models/llm/
        llm=google.LLM(
                model="gemini-2.5-flash",
            ),
        # Text-to-speech (TTS) is your agent's voice, turning the LLM's text into speech that the user can hear
        # See all available models as well as voice selections at https://docs.livekit.io/agents/models/tts/
        tts=murf.TTS(
                voice="en-US-matthew", 
                style="Conversation",
                tokenizer=tokenize.basic.SentenceTokenizer(min_sentence_len=2),
                text_pacing=True
            ),
        # VAD and turn detection are used to determine when the user is speaking and when the agent should respond
        # See more at https://docs.livekit.io/agents/build/turns
        turn_detection=MultilingualModel(),
        vad=ctx.proc.userdata["vad"],
        # allow the LLM to generate a response while waiting for the end of turn
        # See more at https://docs.livekit.io/agents/build/audio/#preemptive-generation
        preemptive_generation=True,
    )

    # To use a realtime model instead of a voice pipeline, use the following session setup instead.
    # (Note: This is for the OpenAI Realtime API. For other providers, see https://docs.livekit.io/agents/models/realtime/))
    # 1. Install livekit-agents[openai]
    # 2. Set OPENAI_API_KEY in .env.local
    # 3. Add `from livekit.plugins import openai` to the top of this file
    # 4. Use the following session setup instead of the version above
    # session = AgentSession(
    #     llm=openai.realtime.RealtimeModel(voice="marin")
    # )

    # Metrics collection, to measure pipeline performance
    # For more information, see https://docs.livekit.io/agents/build/metrics/
    usage_collector = metrics.UsageCollector()

    @session.on("metrics_collected")
    def _on_metrics_collected(ev: MetricsCollectedEvent):
        metrics.log_metrics(ev.metrics)
        usage_collector.collect(ev.metrics)

    async def log_usage():
        summary = usage_collector.get_summary()
        logger.info(f"Usage: {summary}")

    ctx.add_shutdown_callback(log_usage)

    # # Add a virtual avatar to the session, if desired
    # # For other providers, see https://docs.livekit.io/agents/models/avatar/
    # avatar = hedra.AvatarSession(
    #   avatar_id="...",  # See https://docs.livekit.io/agents/models/avatar/plugins/hedra
    # )
    # # Start the avatar and wait for it to join
    # await avatar.start(session, room=ctx.room)

    # Start the session, which initializes the voice pipeline and warms up the models
    # Read the last check-in (if any) so the agent can reference it at session start
    last_entry = {}
    try:
        filename = os.path.join(os.getcwd(), "backend", "wellness_log.json")
        if os.path.exists(filename):
            with open(filename, "r", encoding="utf-8") as f:
                data = json.load(f)
                if isinstance(data, list) and data:
                    last_entry = data[-1]
    except Exception:
        last_entry = {}

    assistant = Assistant(last_entry=last_entry)

    await session.start(
        agent=assistant,
        room=ctx.room,
        room_input_options=RoomInputOptions(
            # For telephony applications, use `BVCTelephony` for best results
            noise_cancellation=noise_cancellation.BVC(),
        ),
    )

    # Join the room and connect to the user
    await ctx.connect()


if __name__ == "__main__":
    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint, prewarm_fnc=prewarm))
