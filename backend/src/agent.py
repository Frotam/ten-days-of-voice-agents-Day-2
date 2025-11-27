import logging
import os
import json
import datetime
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
    function_tool,
    RunContext,
)
from livekit.agents.voice import AgentSession
from livekit.plugins import murf, silero, google, deepgram, noise_cancellation
from livekit.plugins.turn_detector.multilingual import MultilingualModel

logger = logging.getLogger("agent")

load_dotenv(".env.local")
class Assistant(Agent):
    def __init__(self) -> None:
        # Fraud-agent voice (calm, reassuring)
        self.VOICE_BANK = {"voice_id": "Matthew", "style": "Calm", "model": "Falcon"}

        super().__init__(
            instructions=(
                "You are a calm, professional fraud department agent for a demo bank.\n"
                "Verify identity with a non-sensitive security question, read suspected transactions, ask for confirmation, and mark the case safe or fraudulent.\n"
                "Never ask for full card numbers, PINs, or passwords. Keep the interaction short and reassuring."
            )
        )

        # Fraud case state
        self.current_fraud_case: Optional[dict] = None
        self.fraud_db_path = os.path.join(os.getcwd(), "shared-data", "fraud_cases.json")
        self.bank_name = "Demo Bank"

    # ---------------- Fraud alert tools ----------------
    @function_tool
    async def load_fraud_case(self, context: RunContext, username: str):
        try:
            path = self.fraud_db_path
            if not os.path.exists(path):
                return {"error": "Fraud database not found."}
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            for entry in data:
                if (entry.get("userName") or "").lower() == (username or "").lower():
                    self.current_fraud_case = entry
                    return {"status": "ok", "case": entry}
            return {"status": "not_found"}
        except Exception as e:
            logger.error(f"Failed to load fraud cases: {e}")
            return {"error": "failed_to_load"}

    @function_tool
    async def start_fraud_call(self, context: RunContext, username: str):
        loaded = await self.load_fraud_case(context, username)
        if loaded.get("status") != "ok":
            return {"status": "error", "message": "No fraud case found for that user."}

        case = self.current_fraud_case
        bank = self.bank_name
        greeting = (
            f"Hello {case.get('userName')}, this is the fraud department at {bank}. "
            "We're contacting you about a suspicious transaction on your account."
        )
        verification_prompt = case.get("securityQuestion") or "Please confirm a small detail to verify your identity."
        return {"status": "ok", "greeting": greeting, "verification_prompt": verification_prompt}

    @function_tool
    async def verify_security(self, context: RunContext, answer: str):
        case = getattr(self, "current_fraud_case", None)
        if not case:
            return {"status": "error", "message": "No fraud case loaded."}

        expected = (case.get("securityAnswer") or "").strip().lower()
        got = (answer or "").strip().lower()
        if expected and got == expected:
            return {"status": "verified"}
        else:
            case["status"] = "verification_failed"
            case["outcome_note"] = "Verification failed during fraud call."
            await self._save_fraud_db()
            return {"status": "verification_failed", "message": "Verification failed. We cannot proceed."}

    @function_tool
    async def read_transaction(self, context: RunContext):
        case = getattr(self, "current_fraud_case", None)
        if not case:
            return {"status": "error", "message": "No fraud case loaded."}
        text = (
            f"We see a {case.get('transactionCategory')} charge of {case.get('transactionAmount')} "
            f"to {case.get('transactionName')} via {case.get('transactionSource')} on {case.get('transactionTime')} "
            f"using card {case.get('cardEnding')} in {case.get('location')}."
        )
        question = "Did you make this transaction?"
        return {"status": "ok", "text": text, "question": question}

    @function_tool
    async def resolve_fraud(self, context: RunContext, made_transaction: bool):
        case = getattr(self, "current_fraud_case", None)
        if not case:
            return {"status": "error", "message": "No fraud case loaded."}
        if made_transaction:
            case["status"] = "confirmed_safe"
            case["outcome_note"] = "Customer confirmed transaction as legitimate."
            outcome_text = "Thank you — we've marked this transaction as legitimate and no further action is required."
        else:
            case["status"] = "confirmed_fraud"
            case["outcome_note"] = "Customer reported transaction as fraudulent. Card blocked and dispute initiated (mock)."
            outcome_text = (
                "Thank you — we've marked this transaction as fraudulent. "
                "As a precaution, we've blocked the card and initiated a dispute (demo). Our team will contact you with next steps."
            )
        await self._save_fraud_db()
        return {"status": "ok", "outcome": outcome_text, "case_status": case.get("status")}

    async def _save_fraud_db(self):
        try:
            path = self.fraud_db_path
            if not os.path.exists(path):
                return False
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            updated = False
            for i, entry in enumerate(data):
                if entry.get("securityIdentifier") == self.current_fraud_case.get("securityIdentifier") or (
                    (entry.get("userName") or "").lower() == (self.current_fraud_case.get("userName") or "").lower()
                ):
                    data[i] = self.current_fraud_case
                    updated = True
                    break
            if not updated:
                data.append(self.current_fraud_case)
            tmp = path + ".tmp"
            with open(tmp, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            os.replace(tmp, path)
            logger.info("Fraud DB updated")
            return True
        except Exception as e:
            logger.error(f"Failed to save fraud DB: {e}")
            return False

    async def _ensure_voice(self, context: Optional[RunContext], voice):
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
        try:
            # Preferred: context.session.tts.set_voice (works with DynamicMurf)
            session = getattr(context, "session", None)
            if session is not None:
                tts = getattr(session, "tts", None)
                if tts is not None:
                    # call set_voice in thread if blocking
                    try:
                        # if it's async-capable, call directly
                        maybe = tts.set_voice
                        if asyncio.iscoroutinefunction(maybe):
                            await maybe(voice)
                        else:
                            # run blocking call in thread to avoid blocking event loop
                            await asyncio.to_thread(maybe, voice)
                        return
                    except Exception:
                        # fallback to attribute set
                        try:
                            # if voice is a dict, set voice attribute to voice_id
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
                voice={"voice_id": "Matthew", "style": "Conversation", "model": "Falcon"},
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
    # Instantiate the tutor assistant (no wellness check-ins)
    assistant = Assistant()

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
