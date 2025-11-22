import logging
import os
import json
import datetime
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
    def __init__(self) -> None:
        super().__init__(
            instructions="""You are a friendly Starbucks barista. The user will place a drink order by speaking.
            Your job is to ask concise clarifying questions until the following order state is fully filled:
            {
              "drinkType": "string",
              "size": "string",
              "milk": "string",
              "extras": ["string"],
              "name": "string"
            }
            Ask one clear question at a time to obtain missing fields. Confirm the full order once complete, then save the order to disk and tell the user it was saved.
            Keep a friendly, helpful tone and avoid complex formatting.""",
        )

        # Initialize a small order state that we will fill via tools
        self.order_state = {
            "drinkType": "",
            "size": "",
            "milk": "",
            "extras": [],
            "name": "",
        }

    @function_tool
    async def update_order(self, context: RunContext, field: str, value: str):
        """Update a specific field of the order.

        Args:
            field: One of 'drinkType', 'size', 'milk', 'extras', 'name'
            value: The value to set. For 'extras', a comma-separated string will be split into list items.

        Returns:
            A short status string describing what changed and any remaining missing fields, or a saved-file confirmation when complete.
        """

        allowed = {"drinkType", "size", "milk", "extras", "name"}
        if field not in allowed:
            return f"Unknown field '{field}'. Allowed fields: {', '.join(sorted(allowed))}."

        if field == "extras":
            # Accept a comma-separated list, or a single extra
            extras = [e.strip() for e in value.split(",") if e.strip()]
            if extras:
                self.order_state["extras"].extend(extras)
        else:
            self.order_state[field] = value.strip()

        # Determine missing fields (extras must have at least one item)
        missing = [f for f in ["drinkType", "size", "milk", "extras", "name"] if not self.order_state.get(f)]
        if self.order_state.get("extras") == [] and "extras" not in missing:
            missing.append("extras")

        if not missing:
            # Order complete: save to JSON file
            orders_dir = os.path.join(os.getcwd(), "backend", "orders")
            os.makedirs(orders_dir, exist_ok=True)
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = os.path.join(orders_dir, f"order_{timestamp}.json")
            try:
                with open(filename, "w", encoding="utf-8") as f:
                    json.dump(self.order_state, f, ensure_ascii=False, indent=2)
            except Exception as e:
                return f"Order complete but failed to save: {e}"

            return f"Order complete and saved to {filename}."

        return f"Updated '{field}'. Missing fields: {', '.join(missing)}."

    @function_tool
    async def get_order(self, context: RunContext):
        """Return the current order state as JSON-serializable dict."""
        return self.order_state


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
    await session.start(
        agent=Assistant(),
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
