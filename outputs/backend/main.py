"""
Lightspeed Call Intelligence — FastAPI Backend
================================================
Endpoints
---------
POST   /api/upload              Upload a ZIP of audio files for batch processing
GET    /api/jobs/{job_id}       Poll processing job status
GET    /api/calls               List all analysed calls (filterable)
GET    /api/calls/{call_id}     Get single call detail
PATCH  /api/calls/{call_id}/status  Update action status (done/pending/follow_up)
GET    /api/agents              List agents
POST   /api/agents              Add agent
DELETE /api/agents/{agent_id}   Remove agent
GET    /api/stats               Aggregated stats (dashboard widgets)
GET    /health                  Health check

Processing pipeline (background task per uploaded file)
--------------------------------------------------------
1. Extract audio from ZIP
2. Transcribe via OpenAI Whisper
3. Analyse with Claude (structured JSON output)
4. Persist to SQLite
"""

from __future__ import annotations

import asyncio
import io
import json
import logging
import os
import re
import tempfile
import uuid
import zipfile
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional

import aiosqlite
from anthropic import AsyncAnthropic
from dotenv import load_dotenv
from fastapi import BackgroundTasks, FastAPI, File, HTTPException, Query, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from openai import AsyncOpenAI
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings

# ── Config ────────────────────────────────────────────────────────────────────

load_dotenv()


class Settings(BaseSettings):
    anthropic_api_key: str = Field(..., env="ANTHROPIC_API_KEY")
    openai_api_key: str = Field(..., env="OPENAI_API_KEY")
    db_path: str = Field("lightspeed.db", env="DB_PATH")
    allowed_origins: str = Field("*", env="ALLOWED_ORIGINS")
    claude_model: str = Field("claude-sonnet-4-20250514", env="CLAUDE_MODEL")
    whisper_model: str = Field("whisper-1", env="WHISPER_MODEL")

    class Config:
        env_file = ".env"
        extra = "ignore"


settings = Settings()

# ── Logging ───────────────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
)
log = logging.getLogger("lci")

# ── DB helpers ────────────────────────────────────────────────────────────────

DB_PATH = settings.db_path


async def get_db() -> aiosqlite.Connection:
    db = await aiosqlite.connect(DB_PATH)
    db.row_factory = aiosqlite.Row
    await db.execute("PRAGMA journal_mode=WAL")
    await db.execute("PRAGMA foreign_keys=ON")
    return db


async def init_db() -> None:
    async with aiosqlite.connect(DB_PATH) as db:
        db.row_factory = aiosqlite.Row
        await db.executescript("""
            CREATE TABLE IF NOT EXISTS agents (
                id          TEXT PRIMARY KEY,
                name        TEXT NOT NULL,
                role        TEXT DEFAULT 'Agent',
                initials    TEXT,
                color       TEXT DEFAULT '#4f8ef7',
                created_at  TEXT DEFAULT (datetime('now'))
            );

            CREATE TABLE IF NOT EXISTS calls (
                id                          TEXT PRIMARY KEY,
                agent_id                    TEXT REFERENCES agents(id),
                job_id                      TEXT,
                filename                    TEXT,
                customer_name               TEXT,
                business_name               TEXT,
                customer_phone              TEXT,
                primary_topic               TEXT,
                call_outcome                TEXT,
                priority                    TEXT DEFAULT 'medium',
                churn_risk                  TEXT DEFAULT 'medium',
                action_status               TEXT DEFAULT 'pending',
                duration_seconds            INTEGER DEFAULT 0,
                agent_tone_score            REAL DEFAULT 3.0,
                agent_empathy_score         REAL DEFAULT 3.0,
                agent_clarity_score         REAL DEFAULT 3.0,
                overall_customer_sentiment  REAL DEFAULT 3.0,
                agent_talk_ratio            REAL DEFAULT 0.5,
                first_call_resolution       INTEGER DEFAULT 0,
                policy_number               TEXT,
                claim_number                TEXT,
                amount_disputed             TEXT,
                summary                     TEXT,
                coaching_note               TEXT,
                transcript                  TEXT,
                full_analysis               TEXT,
                call_datetime               TEXT DEFAULT (datetime('now')),
                created_at                  TEXT DEFAULT (datetime('now'))
            );

            CREATE TABLE IF NOT EXISTS jobs (
                id          TEXT PRIMARY KEY,
                agent_id    TEXT REFERENCES agents(id),
                filename    TEXT,
                status      TEXT DEFAULT 'queued',
                total       INTEGER DEFAULT 0,
                processed   INTEGER DEFAULT 0,
                errors      INTEGER DEFAULT 0,
                started_at  TEXT,
                finished_at TEXT,
                created_at  TEXT DEFAULT (datetime('now'))
            );

            CREATE INDEX IF NOT EXISTS idx_calls_agent ON calls(agent_id);
            CREATE INDEX IF NOT EXISTS idx_calls_job   ON calls(job_id);
            CREATE INDEX IF NOT EXISTS idx_calls_dt    ON calls(call_datetime);
        """)
        await db.commit()
        log.info("Database initialised at %s", DB_PATH)


# ── Lifespan ──────────────────────────────────────────────────────────────────


@asynccontextmanager
async def lifespan(app: FastAPI):
    await init_db()
    yield


# ── App ───────────────────────────────────────────────────────────────────────

app = FastAPI(
    title="Lightspeed Call Intelligence API",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.allowed_origins.split(","),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── AI clients ────────────────────────────────────────────────────────────────

anthropic_client = AsyncAnthropic(api_key=settings.anthropic_api_key)
openai_client = AsyncOpenAI(api_key=settings.openai_api_key)

# ── Prompts ───────────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """You are an expert insurance call analyst. Analyse the provided call transcript and return a detailed, structured JSON object.

Focus on:
- Customer sentiment and emotional journey throughout the call
- Agent performance metrics (tone, empathy, clarity, professionalism)
- Business outcomes and risk indicators
- Coaching opportunities and actionable feedback

Always respond with valid JSON only, no markdown fences."""

USER_PROMPT_TEMPLATE = """Analyse this insurance call transcript and return a JSON object with exactly this structure:

{{
  "customer_name": "Full name extracted from transcript or 'Unknown Customer'",
  "business_name": "Company/business name or 'Personal'",
  "customer_phone": "Phone number if mentioned or null",
  "primary_topic": "one of: billing_dispute|claim_status|policy_change|cancellation|new_policy|general_enquiry|complaint|coverage_question",
  "call_outcome": "one of: resolved|follow_up_required|escalated|unresolved",
  "priority": "one of: urgent|high|medium|low",
  "churn_risk": "one of: high|medium|low",
  "duration_seconds": <integer seconds>,
  "agent_tone_score": <float 1.0-5.0>,
  "agent_empathy_score": <float 1.0-5.0>,
  "agent_clarity_score": <float 1.0-5.0>,
  "overall_customer_sentiment": <float 1.0-5.0 where 1=very negative, 5=very positive>,
  "agent_talk_ratio": <float 0.0-1.0, proportion of words spoken by agent>,
  "first_call_resolution": <1 if resolved on this call, 0 otherwise>,
  "policy_number": "policy number if mentioned or null",
  "claim_number": "claim number if mentioned or null",
  "amount_disputed": "dollar amount if mentioned or null",
  "summary": "2-3 sentence objective summary of the call",
  "coaching_note": "1-2 sentence encouraging, growth-focused coaching note for the agent. Highlight strengths first, then one specific actionable tip. Use a warm, supportive tone.",
  "turns": [
    {{"speaker": "Agent"|"Customer", "text": "exact quoted text from transcript"}},
    ...
  ]
}}

Transcript filename: {filename}
Transcript:
---
{transcript}
---

Return ONLY the JSON object. No explanation, no markdown."""


# ── Transcription ─────────────────────────────────────────────────────────────

AUDIO_EXTENSIONS = {".mp3", ".mp4", ".m4a", ".wav", ".ogg", ".webm", ".flac"}


async def transcribe_audio(audio_bytes: bytes, filename: str) -> str:
    """Send audio bytes to Whisper and return transcript text."""
    suffix = Path(filename).suffix.lower() or ".mp3"
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp.write(audio_bytes)
        tmp_path = tmp.name
    try:
        with open(tmp_path, "rb") as audio_file:
            response = await openai_client.audio.transcriptions.create(
                model=settings.whisper_model,
                file=(filename, audio_file, "audio/mpeg"),
                response_format="text",
            )
        return str(response)
    finally:
        Path(tmp_path).unlink(missing_ok=True)


# ── Analysis ──────────────────────────────────────────────────────────────────

async def analyse_transcript(transcript: str, filename: str) -> Dict[str, Any]:
    """Send transcript to Claude and return parsed analysis dict."""
    user_prompt = USER_PROMPT_TEMPLATE.format(
        filename=filename, transcript=transcript
    )
    message = await anthropic_client.messages.create(
        model=settings.claude_model,
        max_tokens=2048,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": user_prompt}],
    )
    raw = message.content[0].text.strip()
    # Strip accidental markdown fences
    raw = re.sub(r"^```(?:json)?\s*", "", raw)
    raw = re.sub(r"\s*```$", "", raw)
    return json.loads(raw)


# ── Processing pipeline ───────────────────────────────────────────────────────

async def process_audio_file(
    job_id: str,
    agent_id: str,
    filename: str,
    audio_bytes: bytes,
) -> None:
    """Full pipeline: transcribe → analyse → persist."""
    call_id = str(uuid.uuid4())
    try:
        log.info("[job=%s] Transcribing %s …", job_id, filename)
        transcript = await transcribe_audio(audio_bytes, filename)

        log.info("[job=%s] Analysing %s …", job_id, filename)
        analysis = await analyse_transcript(transcript, filename)

        turns = analysis.pop("turns", [])
        analysis["transcript"] = json.dumps(turns)
        analysis["full_analysis"] = json.dumps(analysis)

        async with aiosqlite.connect(DB_PATH) as db:
            db.row_factory = aiosqlite.Row
            await db.execute(
                """INSERT INTO calls (
                    id, agent_id, job_id, filename,
                    customer_name, business_name, customer_phone,
                    primary_topic, call_outcome, priority, churn_risk,
                    duration_seconds,
                    agent_tone_score, agent_empathy_score, agent_clarity_score,
                    overall_customer_sentiment, agent_talk_ratio,
                    first_call_resolution,
                    policy_number, claim_number, amount_disputed,
                    summary, coaching_note, transcript, full_analysis
                ) VALUES (
                    :id, :agent_id, :job_id, :filename,
                    :customer_name, :business_name, :customer_phone,
                    :primary_topic, :call_outcome, :priority, :churn_risk,
                    :duration_seconds,
                    :agent_tone_score, :agent_empathy_score, :agent_clarity_score,
                    :overall_customer_sentiment, :agent_talk_ratio,
                    :first_call_resolution,
                    :policy_number, :claim_number, :amount_disputed,
                    :summary, :coaching_note, :transcript, :full_analysis
                )""",
                {
                    "id": call_id,
                    "agent_id": agent_id,
                    "job_id": job_id,
                    "filename": filename,
                    **{
                        k: analysis.get(k)
                        for k in [
                            "customer_name", "business_name", "customer_phone",
                            "primary_topic", "call_outcome", "priority", "churn_risk",
                            "duration_seconds",
                            "agent_tone_score", "agent_empathy_score", "agent_clarity_score",
                            "overall_customer_sentiment", "agent_talk_ratio",
                            "first_call_resolution",
                            "policy_number", "claim_number", "amount_disputed",
                            "summary", "coaching_note", "transcript", "full_analysis",
                        ]
                    },
                },
            )
            await db.execute(
                "UPDATE jobs SET processed = processed + 1 WHERE id = ?", (job_id,)
            )
            await db.commit()
        log.info("[job=%s] ✓ Saved call %s", job_id, call_id)

    except Exception as exc:
        log.exception("[job=%s] ✗ Failed on %s: %s", job_id, filename, exc)
        async with aiosqlite.connect(DB_PATH) as db:
            await db.execute(
                "UPDATE jobs SET errors = errors + 1 WHERE id = ?", (job_id,)
            )
            await db.commit()


async def process_zip(
    job_id: str, agent_id: str, zip_bytes: bytes, zip_filename: str
) -> None:
    """Extract ZIP and kick off pipeline for each audio file."""
    async with aiosqlite.connect(DB_PATH) as db:
        await db.execute(
            "UPDATE jobs SET status='processing', started_at=datetime('now') WHERE id=?",
            (job_id,),
        )
        await db.commit()

    try:
        with zipfile.ZipFile(io.BytesIO(zip_bytes)) as zf:
            audio_members = [
                m for m in zf.namelist()
                if Path(m).suffix.lower() in AUDIO_EXTENSIONS
                and not m.startswith("__MACOSX")
            ]
            async with aiosqlite.connect(DB_PATH) as db:
                await db.execute(
                    "UPDATE jobs SET total=? WHERE id=?",
                    (len(audio_members), job_id),
                )
                await db.commit()

            tasks = []
            for member in audio_members:
                audio_bytes = zf.read(member)
                tasks.append(
                    process_audio_file(
                        job_id=job_id,
                        agent_id=agent_id,
                        filename=Path(member).name,
                        audio_bytes=audio_bytes,
                    )
                )
            await asyncio.gather(*tasks, return_exceptions=True)

        async with aiosqlite.connect(DB_PATH) as db:
            await db.execute(
                "UPDATE jobs SET status='complete', finished_at=datetime('now') WHERE id=?",
                (job_id,),
            )
            await db.commit()
        log.info("[job=%s] Batch complete (%d files)", job_id, len(audio_members))

    except Exception as exc:
        log.exception("[job=%s] ZIP processing failed: %s", job_id, exc)
        async with aiosqlite.connect(DB_PATH) as db:
            await db.execute(
                "UPDATE jobs SET status='error', finished_at=datetime('now') WHERE id=?",
                (job_id,),
            )
            await db.commit()


# ── Pydantic models ───────────────────────────────────────────────────────────


class AgentCreate(BaseModel):
    name: str
    role: str = "Agent"
    color: str = "#4f8ef7"


class CallStatusUpdate(BaseModel):
    action_status: Literal["pending", "done", "follow_up"]


# ── Routes ─────────────────────────────────────────────────────────────────────


@app.get("/health")
async def health():
    return {"status": "ok", "timestamp": datetime.now(timezone.utc).isoformat()}


# ── Upload ────────────────────────────────────────────────────────────────────


@app.post("/api/upload", status_code=202)
async def upload_batch(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    agent_id: str = Query(..., description="Agent ID to attribute calls to"),
):
    """
    Upload a ZIP archive of audio files.
    Returns a job_id to poll at GET /api/jobs/{job_id}.
    """
    if not file.filename.endswith(".zip"):
        raise HTTPException(status_code=400, detail="Only .zip files accepted")

    zip_bytes = await file.read()
    job_id = str(uuid.uuid4())

    async with aiosqlite.connect(DB_PATH) as db:
        # Verify agent exists (create on-the-fly if missing — useful for first run)
        row = await db.execute_fetchall(
            "SELECT id FROM agents WHERE id=?", (agent_id,)
        )
        if not row:
            raise HTTPException(status_code=404, detail=f"Agent '{agent_id}' not found")

        await db.execute(
            "INSERT INTO jobs (id, agent_id, filename, status) VALUES (?,?,?,?)",
            (job_id, agent_id, file.filename, "queued"),
        )
        await db.commit()

    background_tasks.add_task(process_zip, job_id, agent_id, zip_bytes, file.filename)
    return {"job_id": job_id, "status": "queued"}


# ── Jobs ──────────────────────────────────────────────────────────────────────


@app.get("/api/jobs/{job_id}")
async def get_job(job_id: str):
    async with aiosqlite.connect(DB_PATH) as db:
        db.row_factory = aiosqlite.Row
        row = await db.execute_fetchall(
            "SELECT * FROM jobs WHERE id=?", (job_id,)
        )
    if not row:
        raise HTTPException(status_code=404, detail="Job not found")
    return dict(row[0])


# ── Calls ─────────────────────────────────────────────────────────────────────


@app.get("/api/calls")
async def list_calls(
    agent_id: Optional[str] = None,
    churn_risk: Optional[str] = None,
    primary_topic: Optional[str] = None,
    call_outcome: Optional[str] = None,
    priority: Optional[str] = None,
    action_status: Optional[str] = None,
    limit: int = Query(100, ge=1, le=500),
    offset: int = Query(0, ge=0),
):
    filters = []
    params: list = []

    if agent_id:
        filters.append("c.agent_id = ?"); params.append(agent_id)
    if churn_risk:
        filters.append("c.churn_risk = ?"); params.append(churn_risk)
    if primary_topic:
        filters.append("c.primary_topic = ?"); params.append(primary_topic)
    if call_outcome:
        filters.append("c.call_outcome = ?"); params.append(call_outcome)
    if priority:
        filters.append("c.priority = ?"); params.append(priority)
    if action_status:
        filters.append("c.action_status = ?"); params.append(action_status)

    where = ("WHERE " + " AND ".join(filters)) if filters else ""
    params += [limit, offset]

    async with aiosqlite.connect(DB_PATH) as db:
        db.row_factory = aiosqlite.Row
        rows = await db.execute_fetchall(
            f"""SELECT c.*, a.name AS agent_name
                FROM calls c
                LEFT JOIN agents a ON a.id = c.agent_id
                {where}
                ORDER BY c.call_datetime DESC
                LIMIT ? OFFSET ?""",
            params,
        )
        total_rows = await db.execute_fetchall(
            f"SELECT COUNT(*) AS n FROM calls c {where}",
            params[:-2],  # exclude limit/offset
        )

    calls = [dict(r) for r in rows]
    # Parse transcript JSON back to list for frontend convenience
    for c in calls:
        if c.get("transcript"):
            try:
                c["turns"] = json.loads(c["transcript"])
            except Exception:
                c["turns"] = []
        else:
            c["turns"] = []

    return {
        "calls": calls,
        "total": dict(total_rows[0])["n"] if total_rows else 0,
        "limit": limit,
        "offset": offset,
    }


@app.get("/api/calls/{call_id}")
async def get_call(call_id: str):
    async with aiosqlite.connect(DB_PATH) as db:
        db.row_factory = aiosqlite.Row
        rows = await db.execute_fetchall(
            """SELECT c.*, a.name AS agent_name
               FROM calls c LEFT JOIN agents a ON a.id=c.agent_id
               WHERE c.id=?""",
            (call_id,),
        )
    if not rows:
        raise HTTPException(status_code=404, detail="Call not found")
    call = dict(rows[0])
    if call.get("transcript"):
        try:
            call["turns"] = json.loads(call["transcript"])
        except Exception:
            call["turns"] = []
    return call


@app.patch("/api/calls/{call_id}/status")
async def update_call_status(call_id: str, body: CallStatusUpdate):
    async with aiosqlite.connect(DB_PATH) as db:
        res = await db.execute(
            "UPDATE calls SET action_status=? WHERE id=?",
            (body.action_status, call_id),
        )
        await db.commit()
        if res.rowcount == 0:
            raise HTTPException(status_code=404, detail="Call not found")
    return {"id": call_id, "action_status": body.action_status}


# ── Agents ────────────────────────────────────────────────────────────────────


def _make_initials(name: str) -> str:
    parts = name.strip().split()
    if len(parts) >= 2:
        return (parts[0][0] + parts[-1][0]).upper()
    return name[:2].upper()


@app.get("/api/agents")
async def list_agents():
    async with aiosqlite.connect(DB_PATH) as db:
        db.row_factory = aiosqlite.Row
        rows = await db.execute_fetchall("SELECT * FROM agents ORDER BY created_at")
    return [dict(r) for r in rows]


@app.post("/api/agents", status_code=201)
async def create_agent(body: AgentCreate):
    agent_id = re.sub(r"[^a-z0-9]+", "-", body.name.lower()).strip("-")
    initials = _make_initials(body.name)
    async with aiosqlite.connect(DB_PATH) as db:
        try:
            await db.execute(
                "INSERT INTO agents (id, name, role, initials, color) VALUES (?,?,?,?,?)",
                (agent_id, body.name, body.role, initials, body.color),
            )
            await db.commit()
        except aiosqlite.IntegrityError:
            raise HTTPException(status_code=409, detail="Agent ID already exists")
    return {"id": agent_id, "name": body.name, "role": body.role, "initials": initials, "color": body.color}


@app.delete("/api/agents/{agent_id}", status_code=204)
async def delete_agent(agent_id: str):
    async with aiosqlite.connect(DB_PATH) as db:
        res = await db.execute("DELETE FROM agents WHERE id=?", (agent_id,))
        await db.commit()
        if res.rowcount == 0:
            raise HTTPException(status_code=404, detail="Agent not found")


# ── Stats ─────────────────────────────────────────────────────────────────────


@app.get("/api/stats")
async def get_stats(agent_id: Optional[str] = None):
    """Aggregated stats for the dashboard widgets."""
    where = "WHERE agent_id=?" if agent_id else ""
    params = [agent_id] if agent_id else []

    async with aiosqlite.connect(DB_PATH) as db:
        db.row_factory = aiosqlite.Row

        totals = await db.execute_fetchall(
            f"""SELECT
                COUNT(*) AS total_calls,
                ROUND(AVG(overall_customer_sentiment),2) AS avg_sentiment,
                ROUND(AVG(agent_tone_score),2) AS avg_tone,
                ROUND(AVG(agent_empathy_score),2) AS avg_empathy,
                ROUND(AVG(agent_clarity_score),2) AS avg_clarity,
                SUM(CASE WHEN call_outcome='resolved' THEN 1 ELSE 0 END) AS resolved,
                SUM(CASE WHEN call_outcome='escalated' THEN 1 ELSE 0 END) AS escalations,
                SUM(CASE WHEN churn_risk='high' THEN 1 ELSE 0 END) AS churn_high,
                SUM(CASE WHEN priority='urgent' THEN 1 ELSE 0 END) AS urgent,
                SUM(CASE WHEN priority='high' THEN 1 ELSE 0 END) AS high_priority,
                SUM(CASE WHEN action_status='done' THEN 1 ELSE 0 END) AS done,
                SUM(CASE WHEN action_status='follow_up' THEN 1 ELSE 0 END) AS follow_up,
                SUM(CASE WHEN first_call_resolution=1 THEN 1 ELSE 0 END) AS fcr_count,
                ROUND(AVG(agent_talk_ratio)*100,1) AS avg_talk_ratio,
                ROUND(AVG(duration_seconds)/60.0,1) AS avg_duration_minutes
            FROM calls {where}""",
            params,
        )

        topic_rows = await db.execute_fetchall(
            f"""SELECT primary_topic, COUNT(*) AS count
                FROM calls {where}
                GROUP BY primary_topic ORDER BY count DESC""",
            params,
        )

        hourly_rows = await db.execute_fetchall(
            f"""SELECT strftime('%H',call_datetime) AS hour,
                ROUND(AVG(overall_customer_sentiment),2) AS avg_sent
                FROM calls {where}
                GROUP BY hour ORDER BY hour""",
            params,
        )

    row = dict(totals[0]) if totals else {}
    total = row.get("total_calls") or 0
    fcr = round((row.get("fcr_count") or 0) / total * 100) if total else 0
    res_rate = round((row.get("resolved") or 0) / total * 100) if total else 0
    esc_rate = round((row.get("escalations") or 0) / total * 100) if total else 0

    return {
        **row,
        "fcr_pct": fcr,
        "resolution_rate": res_rate,
        "escalation_rate": esc_rate,
        "topics": [dict(r) for r in topic_rows],
        "hourly_sentiment": [dict(r) for r in hourly_rows],
    }
