# Lightspeed Call Intelligence — Backend

FastAPI backend that handles audio upload, Whisper transcription, Claude analysis, and REST API for the Lightspeed Call Intelligence frontend.

---

## Quick Start

### 1. Prerequisites

- Python 3.11+
- An [Anthropic API key](https://console.anthropic.com)
- An [OpenAI API key](https://platform.openai.com/api-keys) (for Whisper transcription)

### 2. Install dependencies

```bash
cd backend
pip install -r requirements.txt
```

### 3. Configure environment

```bash
cp .env.example .env
# Edit .env and set ANTHROPIC_API_KEY and OPENAI_API_KEY
```

### 4. Run the server

```bash
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

The API will be available at `http://localhost:8000`.
Interactive docs: `http://localhost:8000/docs`

---

## Connecting the Frontend

In `lightspeed-call-intelligence.html`, find this line near the top of the `<script>` block:

```js
const API_BASE_URL = '';   // e.g. 'http://localhost:8000'
```

Change it to:

```js
const API_BASE_URL = 'http://localhost:8000';
```

The frontend will automatically:
- Probe the backend on load (green "Live" dot in sidebar)
- Fetch real calls instead of demo data when the backend is reachable
- Sync call status updates (done / follow-up) back to the database
- Fall back to demo data silently if the backend is offline

---

## API Reference

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/health` | Health check |
| `POST` | `/api/upload?agent_id=<id>` | Upload ZIP of audio files |
| `GET` | `/api/jobs/{job_id}` | Poll batch processing status |
| `GET` | `/api/calls` | List calls (filterable) |
| `GET` | `/api/calls/{call_id}` | Get single call |
| `PATCH` | `/api/calls/{call_id}/status` | Update action status |
| `GET` | `/api/agents` | List agents |
| `POST` | `/api/agents` | Create agent |
| `DELETE` | `/api/agents/{agent_id}` | Delete agent |
| `GET` | `/api/stats` | Aggregated dashboard stats |

### Upload a batch

```bash
curl -X POST "http://localhost:8000/api/upload?agent_id=rajan" \
  -F "file=@calls.zip"
# Returns: {"job_id": "...", "status": "queued"}
```

### Poll job status

```bash
curl http://localhost:8000/api/jobs/<job_id>
# Returns: {status, total, processed, errors, ...}
```

---

## Processing Pipeline

1. **Upload** — ZIP file received, job record created, background task started
2. **Extract** — Audio files pulled from ZIP (supports mp3, mp4, m4a, wav, ogg, webm, flac)
3. **Transcribe** — Each file sent to OpenAI Whisper (`whisper-1`)
4. **Analyse** — Transcript sent to Claude Sonnet with the insurance call analysis prompt
5. **Persist** — Structured JSON stored in SQLite alongside the raw transcript

All files in a batch are processed concurrently via `asyncio.gather`.

---

## Database

SQLite file at `./lightspeed.db` (configurable via `DB_PATH` env var). Three tables:

- **agents** — team members
- **calls** — analysed call records with all scores, transcript, and coaching notes
- **jobs** — upload batch tracking

---

## Environment Variables

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `ANTHROPIC_API_KEY` | ✅ | — | Claude API key |
| `OPENAI_API_KEY` | ✅ | — | OpenAI key for Whisper |
| `DB_PATH` | | `lightspeed.db` | SQLite file path |
| `CLAUDE_MODEL` | | `claude-sonnet-4-20250514` | Claude model string |
| `WHISPER_MODEL` | | `whisper-1` | Whisper model |
| `ALLOWED_ORIGINS` | | `*` | CORS allowed origins (comma-separated) |
