# The Fear Network

One-file horror video factory for GitHub Actions.

## What it does

`the_fear_network.py` runs the full pipeline:

1. Generate a horror concept with OpenRouter (free router) or Grok as fallback.
2. Write a complete English horror narration for long-form or Short mode.
3. Generate the narration with local XTTS v2 using `voice.wav`.
4. Build a visual plan and search Pexels/Pixabay for stock video/photo assets.
5. Build the timeline around the **narration duration**, not the raw media length.
6. Add local music when available, or generate a dark ambient fallback with FFmpeg.
7. Render 16:9 long-form or 9:16 Short.
8. Upload publicly to YouTube.

## Repository layout

```text
The-Fear-Network/
├── the_fear_network.py
├── voice.wav
├── requirements.txt
├── .gitignore
├── README.md
├── music/
└── .github/
    └── workflows/
        └── the_fear_network.yml
```

Put your reference voice at the repository root as `voice.wav`.

## GitHub Secrets

Create these repository secrets:

```text
OPENROUTER_API_KEY
GROK_API_KEY
PEXELS_API_KEY
PIXABAY_API_KEY
YOUTUBE_TOKEN_JSON
```

`YOUTUBE_TOKEN_JSON` is the complete contents of the YouTube OAuth `token.json` for The Fear Network. Do not commit the token to the repository.

## Scheduling

The workflow schedules three runs per day:

- 20:00 UTC → long-form
- 23:00 UTC → long-form
- 02:00 UTC → Short

You can also start a manual run from Actions and select `long` or `short`.

## XTTS model caching

The workflow stores `TTS_HOME` inside the GitHub workspace and caches it with `actions/cache`, so the XTTS model can be restored between workflow runs instead of being downloaded from scratch each time the cache is hit.

Public GitHub-hosted runners currently provide `ubuntu-latest` with 4 CPU cores and 16 GB RAM; standard public-repository runners are free. citeturn467598search3

## Important music note

Pexels documents photos and videos through its API, and Pixabay documents images and videos through its public API; neither public API is a general music/audio search API. The program therefore uses files you place in `music/` and generates a dark ambient fallback when no local track exists. citeturn467598search2turn467598search10

## AI providers

OpenRouter's `openrouter/free` router currently selects among available free models and supports structured outputs when the selected provider supports them. Grok 4.6 is available through the xAI API as `grok-4.6`. citeturn467598search0turn897645search0

## Local testing

```bash
python -m pip install -r requirements.txt
python the_fear_network.py
```

For a local run:

```powershell
$env:VIDEO_MODE="short"
$env:OPENROUTER_API_KEY="..."
$env:GROK_API_KEY="..."
$env:PEXELS_API_KEY="..."
$env:PIXABAY_API_KEY="..."
$env:YOUTUBE_TOKEN_JSON=(Get-Content .\youtube_token.json -Raw)
python .\the_fear_network.py
```
