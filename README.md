# Lightspeed Call Intelligence

This repository contains a FastAPI backend and a generated frontend HTML file for the Lightspeed Call Intelligence project.

## Project Structure

- `outputs/backend/` — FastAPI backend
- `outputs/lightspeed-call-intelligence.html` — frontend demo/output
- `.gitignore` — excludes local secrets, databases, caches, and virtual environments

## Backend Setup

```bash
cd outputs/backend
pip install -r requirements.txt
cp .env.example .env
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

## Environment Variables

Set the following in `outputs/backend/.env`:

- `ANTHROPIC_API_KEY`
- `OPENAI_API_KEY`
- `DB_PATH` (optional)
- `CLAUDE_MODEL` (optional)
- `WHISPER_MODEL` (optional)
- `ALLOWED_ORIGINS` (optional)

## GitHub Deployment

1. Create a new repository on GitHub.
2. Add the remote origin.
3. Push the `main` branch.

Example:

```bash
git remote add origin https://github.com/<your-username>/<repo-name>.git
git push -u origin main
```

## Notes

- Do not commit `.env` or `lightspeed.db`.
- Use `outputs/backend/.env.example` as the template for secrets.
- The FastAPI app entry point is `outputs/backend/main.py`.
