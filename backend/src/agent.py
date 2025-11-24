import logging
import os
import json
import datetime
import asyncio
from typing import Optional, List, Dict
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
        # Preferred Murf voice specs (use dicts with voice_id/style/model)
        self.VOICE_MATTHEW = {"voice_id": "Matthew", "style": "Conversation", "model": "Falcon"}
        self.VOICE_ALICIA = {"voice_id": "Alicia", "style": "Conversation", "model": "Falcon"}
        self.VOICE_KEN = {"voice_id": "Ken", "style": "Conversation", "model": "Falcon"}

        super().__init__(
            instructions=(
                "You are a concise, helpful teaching assistant designed for a 'Teach-the-Tutor' learning experience.\n"
                "Support three modes: 'learn' (explain a concept), 'quiz' (ask questions), and 'teach_back' (prompt the user to explain the concept back and provide qualitative feedback).\n"
                "Do NOT provide medical or legal advice. Keep explanations short, clear, and focused on the concept summary provided by the course content file.\n"
                "When asked to switch modes, change behavior accordingly. Use small, concrete examples where helpful.\n"
            )
        )

        # Tutor-specific state
        self.mode: Optional[str] = None  # 'learn' | 'quiz' | 'teach_back'
        self.current_concept: Optional[dict] = None
        self.content: List[Dict] = []
        # Track progress through sample questions for each concept (avoid repeats until exhausted)
        self.question_progress: Dict[str, int] = {}
        # Load small course content file if present
        try:
            content_path = os.path.join(os.getcwd(), "shared-data", "day4_tutor_content.json")
            if os.path.exists(content_path):
                with open(content_path, "r", encoding="utf-8") as f:
                    self.content = json.load(f)
        except Exception:
            self.content = []

        # Mastery tracking filename
        self.mastery_file = os.path.join(os.getcwd(), "backend", "tutor_mastery.json")

    # (Removed wellness check-in functions and Notion integration — this assistant focuses on tutor modes.)

    # ------------------ Tutor tools ------------------
    @function_tool
    async def list_concepts(self, context: RunContext):
        """Return the list of available concepts (id and title)."""
        return [{"id": c.get("id"), "title": c.get("title")} for c in self.content]

    @function_tool
    async def set_mode(self, context: RunContext, mode: str):
        """Set the tutor mode: 'learn', 'quiz', or 'teach_back'."""
        mode = (mode or "").strip().lower()
        if mode not in {"learn", "quiz", "teach_back"}:
            return f"Unknown mode '{mode}'. Choose 'learn', 'quiz', or 'teach_back'."
        self.mode = mode
        # Immediately apply the voice for this mode and keep it until mode changes
        try:
            # Map modes to voice specs
            mode_voice = None
            if mode == "learn":
                mode_voice = self.VOICE_MATTHEW
            elif mode == "quiz":
                mode_voice = self.VOICE_ALICIA
            elif mode == "teach_back":
                mode_voice = self.VOICE_KEN

            if mode_voice is not None:
                # Attempt to set voice on the running session if available
                await self._apply_mode_voice(context, mode_voice)
        except Exception:
            # ignore failures; voice switching is best-effort
            pass

        return f"Mode set to {mode}."

    async def _apply_mode_voice(self, context: Optional[RunContext], voice_spec):
        """Apply the voice for the current mode and record it so we don't switch until mode changes."""
        # Store current mode voice for reference
        self._mode_voice = voice_spec
        # If we have a live context/session, set the TTS voice now
        try:
            await self._ensure_voice(context, voice_spec)
        except Exception:
            pass

    @function_tool
    async def choose_concept(self, context: RunContext, concept_id: str):
        """Select a concept by id to use in next interactions."""
        for c in self.content:
            if c.get("id") == concept_id:
                self.current_concept = c
                # Initialize question progress pointer for this concept if needed
                if concept_id not in self.question_progress:
                    self.question_progress[concept_id] = 0
                return {"status": "ok", "concept": c}
        return {"status": "error", "message": f"Concept '{concept_id}' not found."}

    @function_tool
    async def explain_concept(self, context: RunContext):
        """Return the summary text for the current concept (learn mode).

        Includes the preferred Murf voice for Learn mode.
        """
        if not self.current_concept:
            return {"error": "No concept selected. Use choose_concept first."}
        # Do NOT switch voice here — voice is mode-locked via `set_mode`
        return {"text": self.current_concept.get("summary")}

    @function_tool
    async def ask_quiz_question(self, context: RunContext):
        """Return a quiz question (sample_question) for the current concept.

        Includes preferred Murf voice for Quiz mode.
        """
        if not self.current_concept:
            return {"error": "No concept selected. Use choose_concept first."}
        # Voice must NOT be switched here — voice is mode-locked via `set_mode`.
        # Use a list of sample questions if available; ensure we don't repeat until all are used
        questions = self.current_concept.get("sample_questions") or [self.current_concept.get("sample_question")]
        concept_id = self.current_concept.get("id")
        idx = self.question_progress.get(concept_id, 0)
        if idx < len(questions):
            question = questions[idx]
            # advance pointer so next call returns the next question
            self.question_progress[concept_id] = idx + 1
            # If this is the very first quiz question (pointer was 0), include an announcement
            announcement = None
            if idx == 0 and self.mode == "quiz":
                announcement = f"Initiating quiz — your first question is: {question}"
                # Return the announcement and the question; caller can speak the announcement then the question
            return {"question": question, "remaining": len(questions) - (idx + 1), "announcement": announcement}
        else:
            # All sample questions used for this concept
            return {"question": None, "message": "You have gone through all sample questions for this concept."}

    @function_tool
    async def evaluate_quiz_answer(self, context: RunContext, user_response: str):
        """Evaluate the user's answer to the last quiz question.

        Uses a simple keyword-overlap heuristic against the concept summary. If the
        answer is considered correct, the assistant returns the next question. If
        incorrect, it explains the correct answer (using the concept summary) and
        then offers the next question. Voice must NOT be switched here; mode-locked
        voice is enforced via `set_mode`.
        """
        if not self.current_concept:
            return {"error": "No concept selected. Use choose_concept first."}

        concept_id = self.current_concept.get("id")
        # The last asked question index is pointer-1 because ask_quiz_question increments after selecting
        last_idx = self.question_progress.get(concept_id, 0) - 1
        questions = self.current_concept.get("sample_questions") or [self.current_concept.get("sample_question")]
        if last_idx < 0 or last_idx >= len(questions):
            return {"error": "No quiz question has been asked yet. Call ask_quiz_question first."}

        # Simple keyword extraction from the concept summary
        reference = (self.current_concept.get("summary") or "").lower()
        response = (user_response or "").lower()

        def keywords(text: str):
            import re

            words = re.findall(r"\b[a-z]{4,}\b", text)
            return set(words)

        ref_keys = keywords(reference)
        resp_keys = keywords(response)
        if not ref_keys:
            score = 0.0
        else:
            matched = len(ref_keys & resp_keys)
            score = matched / max(1, len(ref_keys))

        correct_threshold = 0.5
        if score >= correct_threshold:
            # Correct: prepare next question
            next_q = await self.ask_quiz_question(context)
            if next_q.get("question"):
                return {"correct": True, "message": "Correct! Here's your next question:", "next": next_q}
            else:
                return {"correct": True, "message": "Correct! You have completed the questions for this concept.", "next": next_q}
        else:
            # Incorrect: explain using the concept summary, then proceed to next question
            explanation = self.current_concept.get("summary")
            next_q = await self.ask_quiz_question(context)
            return {"correct": False, "explanation": explanation, "next": next_q}

    @function_tool
    async def evaluate_teach_back(self, context: RunContext, user_response: str):
        """Evaluate a user's teach-back response with a simple keyword-overlap heuristic,
        save a mastery record, and return qualitative feedback.
        Preferred Murf voice for teach_back return is Ken.
        """
        if not self.current_concept:
            return {"error": "No concept selected. Use choose_concept first."}

        reference = (self.current_concept.get("summary") or "").lower()
        response = (user_response or "").lower()

        # Simple keyword extraction: remove punctuation, split, take unique words longer than 3 chars
        def keywords(text: str):
            import re

            words = re.findall(r"\b[a-z]{4,}\b", text)
            return set(words)

        ref_keys = keywords(reference)
        resp_keys = keywords(response)
        if not ref_keys:
            score = 0.0
        else:
            matched = len(ref_keys & resp_keys)
            score = matched / max(1, len(ref_keys))

        # Map score to qualitative feedback
        if score >= 0.75:
            feedback = "Great explanation — you covered the main ideas clearly."
        elif score >= 0.4:
            feedback = "A decent start — you touched on some key points. Try elaborating one or two parts."
        else:
            feedback = "Thanks for trying — I'd like you to include more of the core ideas. Would you try explaining X next?"

        # Save mastery record
        record = {
            "datetime": datetime.datetime.now().isoformat(),
            "concept_id": self.current_concept.get("id"),
            "mode": "teach_back",
            "score": round(score, 3),
        }
        try:
            os.makedirs(os.path.dirname(self.mastery_file), exist_ok=True)
            if os.path.exists(self.mastery_file):
                with open(self.mastery_file, "r", encoding="utf-8") as f:
                    try:
                        data = json.load(f)
                        if not isinstance(data, list):
                            data = []
                    except Exception:
                        data = []
            else:
                data = []
            data.append(record)
            tmp = self.mastery_file + ".tmp"
            with open(tmp, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            os.replace(tmp, self.mastery_file)
        except Exception:
            pass

        # Ensure TTS uses Ken for teach-back feedback
        await self._ensure_voice(context, self.VOICE_KEN)
        return {"voice": self.VOICE_KEN, "feedback": feedback, "score": score}

    async def _ensure_voice(self, context: Optional[RunContext], voice):
        """Try multiple strategies to switch the runtime TTS voice to `voice`.

        This sets the voice on `context.session.tts` if available and supports `set_voice`,
        or falls back to assigning a `.voice` attribute. It also stores `self._requested_voice`
        so callers can inspect which voice was requested.
        """
        self._requested_voice = voice
        if context is None:
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
