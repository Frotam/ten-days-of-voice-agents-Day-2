import logging
import os
# (Removed unused imports `json` and `datetime` - not required for the Game Master agent)
import asyncio
from typing import Optional, List, Dict

from dotenv import load_dotenv

# LiveKit framework and plugins
from livekit.agents import (
    Agent,
    JobContext,
    JobProcess,
    MetricsCollectedEvent,
    RoomInputOptions,
    WorkerOptions,
    cli,
    metrics,
    tokenize,
    RunContext,
)
from livekit.agents.voice import AgentSession
from livekit.plugins import murf, silero, google, deepgram, noise_cancellation
from livekit.plugins.turn_detector.multilingual import MultilingualModel

logger = logging.getLogger("agent")

load_dotenv(".env.local")
class Assistant(Agent):
    def __init__(self) -> None:
        # Default TTS voice for the Game Master
        self.VOICE_GM = {"voice_id": "Alicia", "style": "Dramatic", "model": "Falcon"}

        # System message: Decision-Based D&D Voice Game Master
        system_message = (
            "You are a Voice Game Master (GM) running a simple, decision-based D&D-style adventure in a single universe.\n"
            "Your job is to guide the player through a story using only conversation history — no external memory or state.\n\n"
            "Universe: A light-fantasy world with forests, ruins, magic, and mysterious creatures.\n"
            "Tone: Adventurous, dramatic but beginner-friendly.\n"
            "No horror, no gore, no complex mechanics.\n\n"
            "You are the GM. You describe scenes clearly and end every message with a question like: 'What do you do?'\n"
            "Never continue the story without user input.\n\n"
            "The entire adventure must be decision-based. Every scene should offer clear choices (A/B).\n"
            "User decisions must shape the story and the story should last 8–15 turns. At the end, declare GOOD/NEUTRAL/BAD ending and explain why.\n\n"
            "Maintain continuity using the conversation: past choices, characters met, locations visited. Keep dialogue short, clear, and suitable for voice-only interaction.\n\n"
            "Structure: Start (2 choices), Middle (4–5 branching scenes), Ending (GOOD/NEUTRAL/BAD), then ask if the player wants another adventure.\n\n"
            "FIRST MESSAGE: Welcome, traveler.\nYou awaken at the edge of the Whispering Forest, a magical place filled with ancient secrets.\n\nTwo paths stretch before you:\n\nA: A sunlit trail lined with glowing blue flowers.\nB: A shadowy path where the trees whisper your name.\n\nWhat do you choose?"
        )

        super().__init__(instructions=system_message)

        # Adventure state (keeps track of conversation-driven progression)
        self.turns: List[Dict] = []
        self.current_turn: int = 0
        self.min_turns: int = 8
        self.max_turns: int = 15
        self.player_choices: List[str] = []
        self.characters_met: List[str] = []
        self.locations_visited: List[str] = []
        self.first_message = (
            "Welcome, traveler.\nYou awaken at the edge of the Whispering Forest, a magical place filled with ancient secrets.\n\nTwo paths stretch before you:\n\nA: A sunlit trail lined with glowing blue flowers.\nB: A shadowy path where the trees whisper your name.\n\nWhat do you choose?"
        )

    # Adventure helpers (kept minimal because the LLM handles narrative generation)
    def record_choice(self, choice: str) -> None:
        """Record player choices and update adventure state."""
        self.player_choices.append(choice)
        self.current_turn += 1

    def record_location(self, location: str) -> None:
        if location not in self.locations_visited:
            self.locations_visited.append(location)

    def record_character(self, name: str) -> None:
        if name not in self.characters_met:
            self.characters_met.append(name)

    # No function tools required for the D&D GM. The LLM will handle narrative and decision branching.

    async def _ensure_voice(self, context: Optional[RunContext], voice):
        """Best-effort TTS voice switching."""
        self._requested_voice = voice
        if context is None:
            return
        try:
            session = getattr(context, "session", None)
            if session is not None:
                tts = getattr(session, "tts", None)
                if tts is not None:
                    try:
                        maybe = tts.set_voice
                        if asyncio.iscoroutinefunction(maybe):
                            await maybe(voice)
                        else:
                            await asyncio.to_thread(maybe, voice)
                        return
                    except Exception:
                        try:
                            if isinstance(voice, dict):
                                setattr(tts, "voice", voice.get("voice_id"))
                            else:
                                setattr(tts, "voice", voice)
                            return
                        except Exception:
                            return
        except Exception:
            return



def prewarm(proc: JobProcess):
    proc.userdata["vad"] = silero.VAD.load()


class DynamicMurf:
    """Simple proxy wrapper around murf.TTS that allows swapping voice at runtime.

    This recreates an internal murf.TTS instance when `set_voice` is called.
    It forwards attribute access and method calls to the current instance.
    """

    def __init__(self, voice=None, **kwargs):
        # voice can be a string voice id or a dict with keys {voice_id, style, model}
        self._kwargs = dict(kwargs)
        self._voice_spec = None
        if isinstance(voice, dict):
            self._voice_spec = voice
            voice_id = voice.get("voice_id")
            mk = dict(self._kwargs)
            if "style" in voice:
                mk["style"] = voice.get("style")
            if "model" in voice:
                mk["model"] = voice.get("model")
            self._voice = voice_id
            self._impl = murf.TTS(voice=voice_id, **mk)
        else:
            self._voice = voice
            self._impl = murf.TTS(voice=voice, **self._kwargs)

    def set_voice(self, voice):
        # Accept either a string voice id or a voice spec dict
        if isinstance(voice, dict):
            voice_id = voice.get("voice_id")
            if voice_id == self._voice:
                return
            self._voice_spec = voice
            mk = dict(self._kwargs)
            if "style" in voice:
                mk["style"] = voice.get("style")
            if "model" in voice:
                mk["model"] = voice.get("model")
            self._voice = voice_id
            self._impl = murf.TTS(voice=voice_id, **mk)
        else:
            if voice == self._voice:
                return
            self._voice_spec = None
            self._voice = voice
            self._impl = murf.TTS(voice=voice, **self._kwargs)

    def __getattr__(self, item):
        return getattr(self._impl, item)


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
        tts=DynamicMurf(
            voice={"voice_id": "Alicia", "style": "Dramatic", "model": "Falcon"},
                tokenizer=tokenize.basic.SentenceTokenizer(min_sentence_len=2),
                text_pacing=True,
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
    # Instantiate the Game Master assistant
    assistant = Assistant()

    await session.start(
        agent=assistant,
        room=ctx.room,
        room_input_options=RoomInputOptions(
            # For telephony applications, use `BVCTelephony` for best results
            noise_cancellation=noise_cancellation.BVC(),
        ),
    )

    # Ensure the session TTS voice matches the assistant's voice spec
    try:
        tts = getattr(session, "tts", None)
        if tts is not None:
            maybe = getattr(tts, "set_voice", None)
            if maybe is not None:
                if asyncio.iscoroutinefunction(maybe):
                    await maybe(assistant.VOICE_GM)
                else:
                    await asyncio.to_thread(maybe, assistant.VOICE_GM)
    except Exception:
        pass

    # Join the room and connect to the user
    await ctx.connect()


if __name__ == "__main__":
    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint, prewarm_fnc=prewarm))
