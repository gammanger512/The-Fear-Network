from __future__ import annotations

import base64
import json
import math
import os
import random
import re
import shutil
import subprocess
import sys
import textwrap
import time
from pathlib import Path
from typing import Any

import requests
import torch
from TTS.api import TTS
from google.oauth2.credentials import Credentials
from google.auth.transport.requests import Request
from googleapiclient.discovery import build
from googleapiclient.http import MediaFileUpload


# ============================================================
# THE FEAR NETWORK — ONE-FILE VIDEO FACTORY
# ============================================================
# Pipeline:
# AI idea -> AI horror story -> scene/visual plan -> XTTS voice
# -> Pexels/Pixabay visual assets -> music -> FFmpeg montage
# -> YouTube upload
#
# GitHub Actions provides the secrets. voice.wav lives beside this file.
# The XTTS model is cached by the workflow through TTS_HOME.
# ============================================================

BASE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = BASE_DIR / "output"
WORK_DIR = BASE_DIR / "work"
MEDIA_DIR = WORK_DIR / "media"
AUDIO_DIR = WORK_DIR / "audio"
MUSIC_DIR = BASE_DIR / "music"
VOICE_WAV = BASE_DIR / "voice.wav"

for folder in (OUTPUT_DIR, WORK_DIR, MEDIA_DIR, AUDIO_DIR, MUSIC_DIR):
    folder.mkdir(parents=True, exist_ok=True)

# ------------------------- Settings --------------------------
VIDEO_MODE = os.getenv("VIDEO_MODE", "long").strip().lower()  # long / short
TOPIC_HINT = os.getenv("TOPIC_HINT", "")

LONG_MINUTES = float(os.getenv("LONG_MINUTES", "10.0"))
SHORT_MINUTES = float(os.getenv("SHORT_MINUTES", "1.5"))

# We generate one visual plan item per sentence. The planner is asked to
# keep scenes visually distinct, but the renderer can reuse assets if needed.
MAX_SCENES_LONG = int(os.getenv("MAX_SCENES_LONG", "95"))
MAX_SCENES_SHORT = int(os.getenv("MAX_SCENES_SHORT", "18"))

VIDEO_W = 1920 if VIDEO_MODE == "long" else 1080
VIDEO_H = 1080 if VIDEO_MODE == "long" else 1920
FPS = 30

# AI providers
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "").strip()
GROK_API_KEY = os.getenv("GROK_API_KEY", "").strip()
OPENROUTER_MODEL = os.getenv("OPENROUTER_MODEL", "openrouter/free")
GROK_MODEL = os.getenv("GROK_MODEL", "grok-4.6")

# Media APIs
PEXELS_API_KEY = os.getenv("PEXELS_API_KEY", "").strip()
PIXABAY_API_KEY = os.getenv("PIXABAY_API_KEY", "").strip()

# YouTube
YOUTUBE_TOKEN_JSON = os.getenv("YOUTUBE_TOKEN_JSON", "").strip()

# Local XTTS
XTTS_MODEL_NAME = "tts_models/multilingual/multi-dataset/xtts_v2"
XTTS_LANGUAGE = "en"
USE_GPU = torch.cuda.is_available()

REQUEST_TIMEOUT = 90
DOWNLOAD_TIMEOUT = 120


# ============================================================
# GENERAL HELPERS
# ============================================================


def log(message: str) -> None:
    print(message, flush=True)


def run_command(command: list[str], *, capture: bool = False) -> subprocess.CompletedProcess[str]:
    log("$ " + " ".join(map(str, command)))
    return subprocess.run(
        command,
        check=True,
        text=True,
        capture_output=capture,
        encoding="utf-8",
        errors="ignore",
    )


def find_executable(name: str) -> str:
    path = shutil.which(name)
    if not path:
        raise FileNotFoundError(f"{name} was not found in PATH.")
    return path


def ffprobe_duration(path: Path) -> float:
    ffprobe = find_executable("ffprobe")
    result = run_command(
        [
            ffprobe,
            "-v",
            "error",
            "-show_entries",
            "format=duration",
            "-of",
            "default=noprint_wrappers=1:nokey=1",
            str(path),
        ],
        capture=True,
    )
    return float(result.stdout.strip())


def safe_filename(value: str, fallback: str = "video") -> str:
    value = re.sub(r"[^A-Za-z0-9._-]+", "_", value.strip())
    value = value.strip("._-")
    return value[:80] or fallback


def http_json(method: str, url: str, *, headers: dict[str, str] | None = None,
              params: dict[str, Any] | None = None, payload: dict[str, Any] | None = None) -> dict[str, Any]:
    response = requests.request(
        method,
        url,
        headers=headers or {},
        params=params,
        json=payload,
        timeout=REQUEST_TIMEOUT,
    )
    response.raise_for_status()
    return response.json()


def download_file(url: str, destination: Path) -> Path:
    destination.parent.mkdir(parents=True, exist_ok=True)
    with requests.get(url, stream=True, timeout=DOWNLOAD_TIMEOUT, headers={"User-Agent": "Mozilla/5.0"}) as response:
        response.raise_for_status()
        with destination.open("wb") as handle:
            for chunk in response.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    handle.write(chunk)
    return destination


# ============================================================
# AI ENGINE
# ============================================================


def extract_json(text: str) -> dict[str, Any]:
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?", "", text, flags=re.I).strip()
        text = re.sub(r"```$", "", text).strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", text, flags=re.S)
        if not match:
            raise
        return json.loads(match.group(0))


def call_openrouter(messages: list[dict[str, str]], *, temperature: float = 0.8,
                    max_tokens: int = 12000) -> str:
    if not OPENROUTER_API_KEY:
        raise RuntimeError("OPENROUTER_API_KEY is not set.")

    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://github.com/",
        "X-Title": "The Fear Network",
    }
    payload = {
        "model": OPENROUTER_MODEL,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "response_format": {"type": "json_object"},
    }
    data = http_json("POST", "https://openrouter.ai/api/v1/chat/completions", headers=headers, payload=payload)
    return str(data["choices"][0]["message"]["content"])


def call_grok(messages: list[dict[str, str]], *, temperature: float = 0.8,
              max_tokens: int = 12000) -> str:
    if not GROK_API_KEY:
        raise RuntimeError("GROK_API_KEY is not set.")

    headers = {
        "Authorization": f"Bearer {GROK_API_KEY}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": GROK_MODEL,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    data = http_json("POST", "https://api.x.ai/v1/chat/completions", headers=headers, payload=payload)
    return str(data["choices"][0]["message"]["content"])


def ai_json(system_prompt: str, user_prompt: str, *, temperature: float = 0.8,
            max_tokens: int = 12000) -> dict[str, Any]:
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]

    providers: list[tuple[str, Any]] = []
    if OPENROUTER_API_KEY:
        providers.append(("OpenRouter", call_openrouter))
    if GROK_API_KEY:
        providers.append(("Grok", call_grok))

    if not providers:
        raise RuntimeError("Set at least OPENROUTER_API_KEY or GROK_API_KEY.")

    last_error: Exception | None = None
    for provider_name, provider in providers:
        try:
            log(f"🤖 AI provider: {provider_name}")
            return extract_json(provider(messages, temperature=temperature, max_tokens=max_tokens))
        except Exception as error:
            last_error = error
            log(f"⚠️ {provider_name} failed: {type(error).__name__}: {error}")

    raise RuntimeError(f"All AI providers failed: {last_error}")


def fallback_idea() -> str:
    ideas = [
        "A night-shift security guard discovers that the cameras show him arriving in the building hours before he actually does.",
        "A woman receives voicemail messages from her own phone recorded exactly one day in the future.",
        "A remote forest road appears on no map, but every driver who takes it hears a child counting from the back seat.",
        "A family moves into an old house where one locked room becomes colder every night and slowly starts answering questions.",
        "A missing-person case is reopened after a new photograph appears online showing the missing man standing behind the detective taking the photo.",
    ]
    return random.choice(ideas)


def generate_idea() -> str:
    prompt = f"""
Create one original horror-video concept for The Fear Network.
Audience: English-speaking viewers, especially the United States.
Style: cinematic, disturbing, suspenseful, plausible, psychologically gripping.
Avoid copying known films, Reddit posts, creepypastas, or famous stories.
Do not use excessive gore. Build a strong hook and escalating mystery.
Topic hint from the user: {TOPIC_HINT or 'none'}
Return JSON only:
{{"idea": "one-sentence concept"}}
"""
    try:
        data = ai_json(
            "You are an expert horror story concept writer for a faceless YouTube channel.",
            prompt,
            temperature=0.9,
            max_tokens=800,
        )
        idea = str(data.get("idea", "")).strip()
        if idea:
            return idea
    except Exception as error:
        log(f"⚠️ Idea generation failed, using fallback: {error}")
    return fallback_idea()


def story_word_target() -> int:
    return int((LONG_MINUTES if VIDEO_MODE == "long" else SHORT_MINUTES) * 145)


def generate_story(idea: str) -> dict[str, Any]:
    target_words = story_word_target()
    max_scenes = MAX_SCENES_LONG if VIDEO_MODE == "long" else MAX_SCENES_SHORT

    prompt = f"""
Write an original professional horror narration for YouTube.

Concept:
{idea}

Format:
- English, natural American storytelling voice.
- No stage directions inside the narration.
- Strong first 1-3 lines as a hook.
- Escalate tension continuously.
- End with a memorable final reveal or disturbing image.
- Target about {target_words} spoken words.
- The actual narration MUST be complete and coherent, not an outline.
- Create no more than {max_scenes} visual scenes.
- Each scene should correspond to one sentence or a very small sentence group.
- Give each scene a concise visual search query for stock footage/photos.
- Visual queries must describe concrete cinematic subjects/locations, not abstract emotions.
- Make queries suitable for Pexels/Pixabay.

Return JSON only in this shape:
{{
  "title": "YouTube title under 100 characters",
  "description": "Short YouTube description with a curiosity hook",
  "tags": ["horror", "scary story", "..."],
  "narration": "FULL narration text",
  "scenes": [
    {{"text": "exact sentence(s) from narration", "visual_query": "dark abandoned hallway night"}},
    {{"text": "exact sentence(s) from narration", "visual_query": "lonely man security office CCTV"}}
  ]
}}
"""

    data = ai_json(
        "You are the head writer and visual director of The Fear Network. Produce production-ready horror stories.",
        prompt,
        temperature=0.85,
        max_tokens=22000,
    )

    narration = str(data.get("narration", "")).strip()
    scenes = data.get("scenes") or []
    title = str(data.get("title", "The Door Was Already Open")).strip()
    description = str(data.get("description", "A disturbing story that gets worse with every minute.")).strip()
    tags = [str(x).strip() for x in (data.get("tags") or []) if str(x).strip()]

    if not narration:
        raise RuntimeError("AI returned an empty narration.")
    if not scenes:
        scenes = [{"text": sentence, "visual_query": "dark cinematic horror scene"} for sentence in split_sentences(narration)]

    # Safety/quality guard: narration must be plausibly long for the requested mode.
    min_words = int((LONG_MINUTES if VIDEO_MODE == "long" else SHORT_MINUTES) * 105)
    if len(narration.split()) < min_words:
        log(f"⚠️ Narration shorter than requested ({len(narration.split())} words). Continuing with returned story.")

    return {
        "title": title[:100],
        "description": description,
        "tags": tags[:30],
        "narration": narration,
        "scenes": scenes[:max_scenes],
    }


def split_sentences(text: str) -> list[str]:
    text = re.sub(r"\s+", " ", text.strip())
    parts = re.split(r"(?<=[.!?])\s+", text)
    return [part.strip() for part in parts if part.strip()]


def normalize_scene_plan(story: dict[str, Any]) -> list[dict[str, str]]:
    narration = story["narration"]
    sentences = split_sentences(narration)
    raw_scenes = story.get("scenes") or []

    normalized: list[dict[str, str]] = []
    used_sentence_indices: set[int] = set()

    # First use AI-provided text and find it in the narration. Any remaining
    # sentences are appended with an automatically generated visual query.
    for item in raw_scenes:
        text = str(item.get("text", "")).strip()
        query = str(item.get("visual_query", "")).strip()
        if not text:
            continue
        normalized.append({"text": text, "visual_query": query or "cinematic dark horror scene"})

        for index, sentence in enumerate(sentences):
            if index not in used_sentence_indices and sentence == text:
                used_sentence_indices.add(index)
                break

    if not normalized:
        normalized = [{"text": sentence, "visual_query": "cinematic dark horror scene"} for sentence in sentences]

    # Keep the scene count bounded. Merge tail sentences if AI overproduced.
    max_scenes = MAX_SCENES_LONG if VIDEO_MODE == "long" else MAX_SCENES_SHORT
    if len(normalized) > max_scenes:
        group_size = math.ceil(len(normalized) / max_scenes)
        merged: list[dict[str, str]] = []
        for start in range(0, len(normalized), group_size):
            group = normalized[start:start + group_size]
            merged.append({
                "text": " ".join(x["text"] for x in group),
                "visual_query": group[0]["visual_query"],
            })
        normalized = merged

    return normalized


# ============================================================
# XTTS v2
# ============================================================


def make_narration(text: str, output_path: Path) -> None:
    if not VOICE_WAV.exists():
        raise FileNotFoundError(
            f"Missing {VOICE_WAV}. Put your reference voice.wav beside the_fear_network.py."
        )

    log("\n🎙️ Loading XTTS v2...")
    log(f"GPU available: {USE_GPU}")
    if not USE_GPU:
        log("⚠️ Using CPU")

    tts = TTS(XTTS_MODEL_NAME)
    if USE_GPU:
        tts = tts.to("cuda")

    log("🎙️ Generating narration...")
    tts.tts_to_file(
        text=text,
        speaker_wav=str(VOICE_WAV),
        language=XTTS_LANGUAGE,
        file_path=str(output_path),
    )
    log(f"✅ Narration created: {output_path}")


# ============================================================
# STOCK MEDIA SEARCH
# ============================================================


def pexels_video(query: str, orientation: str) -> tuple[str, str] | None:
    if not PEXELS_API_KEY:
        return None
    headers = {"Authorization": PEXELS_API_KEY}
    params = {
        "query": query,
        "per_page": 5,
        "orientation": orientation,
        "size": "medium",
    }
    try:
        data = http_json("GET", "https://api.pexels.com/v1/videos/search", headers=headers, params=params)
        videos = data.get("videos") or []
        if not videos:
            return None
        best = videos[0]
        files = best.get("video_files") or []
        if not files:
            return None
        files = sorted(files, key=lambda x: (abs((x.get("width") or 0) - VIDEO_W), -(x.get("height") or 0)))
        return str(files[0]["link"]), f"pexels_video:{best.get('id', '')}"
    except Exception as error:
        log(f"⚠️ Pexels video failed for '{query}': {error}")
        return None


def pexels_photo(query: str) -> tuple[str, str] | None:
    if not PEXELS_API_KEY:
        return None
    headers = {"Authorization": PEXELS_API_KEY}
    params = {"query": query, "per_page": 5, "size": "large"}
    try:
        data = http_json("GET", "https://api.pexels.com/v1/search", headers=headers, params=params)
        photos = data.get("photos") or []
        if not photos:
            return None
        src = photos[0].get("src") or {}
        url = src.get("large2x") or src.get("large") or src.get("original")
        if not url:
            return None
        return str(url), f"pexels_photo:{photos[0].get('id', '')}"
    except Exception as error:
        log(f"⚠️ Pexels photo failed for '{query}': {error}")
        return None


def pixabay_video(query: str) -> tuple[str, str] | None:
    if not PIXABAY_API_KEY:
        return None
    params = {
        "key": PIXABAY_API_KEY,
        "q": query,
        "per_page": 5,
        "safesearch": "true",
    }
    try:
        data = http_json("GET", "https://pixabay.com/api/videos/", params=params)
        hits = data.get("hits") or []
        if not hits:
            return None
        hit = hits[0]
        videos = hit.get("videos") or {}
        candidates = [videos.get("medium"), videos.get("small"), videos.get("tiny")]
        for item in candidates:
            if item and item.get("url"):
                return str(item["url"]), f"pixabay_video:{hit.get('id', '')}"
        return None
    except Exception as error:
        log(f"⚠️ Pixabay video failed for '{query}': {error}")
        return None


def pixabay_image(query: str) -> tuple[str, str] | None:
    if not PIXABAY_API_KEY:
        return None
    params = {
        "key": PIXABAY_API_KEY,
        "q": query,
        "image_type": "photo",
        "orientation": "vertical" if VIDEO_MODE == "short" else "horizontal",
        "per_page": 5,
        "safesearch": "true",
    }
    try:
        data = http_json("GET", "https://pixabay.com/api/", params=params)
        hits = data.get("hits") or []
        if not hits:
            return None
        hit = hits[0]
        url = hit.get("largeImageURL") or hit.get("webformatURL")
        if not url:
            return None
        return str(url), f"pixabay_image:{hit.get('id', '')}"
    except Exception as error:
        log(f"⚠️ Pixabay image failed for '{query}': {error}")
        return None


def visual_for_query(query: str) -> tuple[str, str]:
    orientation = "portrait" if VIDEO_MODE == "short" else "landscape"
    providers = [
        lambda: pexels_video(query, orientation),
        lambda: pixabay_video(query),
        lambda: pexels_photo(query),
        lambda: pixabay_image(query),
    ]
    for provider in providers:
        result = provider()
        if result:
            return result
    raise RuntimeError(f"No visual asset found for: {query}")


# ============================================================
# MUSIC
# ============================================================
# Important: Pexels and Pixabay public APIs document photos/videos, not a
# public music/audio search endpoint. We therefore support local royalty-free
# music files and generate a dark ambient fallback with FFmpeg.
# ============================================================


def choose_local_music() -> Path | None:
    files = []
    for pattern in ("*.mp3", "*.wav", "*.m4a", "*.aac", "*.ogg"):
        files.extend(MUSIC_DIR.glob(pattern))
    return random.choice(files) if files else None


def generate_dark_drone(path: Path, duration: float) -> Path:
    ffmpeg = find_executable("ffmpeg")
    duration = max(duration, 1.0)
    filter_graph = (
        "aevalsrc="
        "0.12*sin(2*PI*55*t)+0.06*sin(2*PI*82.41*t)+0.03*sin(2*PI*110*t):"
        "s=44100:d={dur}," 
        "lowpass=f=1400,afade=t=in:st=0:d=4,afade=t=out:st={fade}:d=5"
    ).format(dur=duration, fade=max(duration - 5, 0))
    run_command([
        ffmpeg, "-y", "-f", "lavfi", "-i", filter_graph,
        "-c:a", "pcm_s16le", str(path)
    ])
    return path


def prepare_music(duration: float) -> Path:
    local = choose_local_music()
    if local:
        log(f"🎵 Using local music: {local.name}")
        return local
    fallback = AUDIO_DIR / "dark_drone.wav"
    if not fallback.exists() or abs(ffprobe_duration(fallback) - duration) > 1.0:
        log("🎵 No local music found; generating a copyright-safe dark ambient drone.")
        generate_dark_drone(fallback, duration)
    return fallback


# ============================================================
# MEDIA PREPARATION / TIMELINE
# ============================================================


def estimate_scene_durations(scenes: list[dict[str, str]], total_audio_duration: float) -> list[float]:
    weights = [max(1, len(scene["text"].split())) for scene in scenes]
    total_weight = sum(weights)
    raw = [total_audio_duration * (w / total_weight) for w in weights]

    # Keep very short scenes readable and then renormalize.
    raw = [max(1.8, value) for value in raw]
    scale = total_audio_duration / sum(raw)
    return [value * scale for value in raw]


def prepare_visuals(scenes: list[dict[str, str]]) -> list[Path]:
    files: list[Path] = []
    for index, scene in enumerate(scenes, start=1):
        query = scene["visual_query"]
        log(f"\n🖼️ Visual {index}/{len(scenes)}: {query}")
        url, source_id = visual_for_query(query)
        extension = ".mp4" if "video" in source_id else ".jpg"
        destination = MEDIA_DIR / f"scene_{index:03d}{extension}"
        try:
            download_file(url, destination)
        except Exception as error:
            log(f"⚠️ Download failed: {error}")
            # Try a generic fallback query once.
            fallback_url, fallback_id = visual_for_query("dark cinematic horror night")
            destination = MEDIA_DIR / f"scene_{index:03d}{'.mp4' if 'video' in fallback_id else '.jpg'}"
            download_file(fallback_url, destination)
        files.append(destination)
    return files


def render_video(
    narration: Path,
    visual_files: list[Path],
    scene_durations: list[float],
    music: Path,
    output: Path,
) -> None:
    ffmpeg = find_executable("ffmpeg")

    if len(visual_files) != len(scene_durations):
        raise RuntimeError("Visual count and duration count differ.")

    concat_inputs: list[Path] = []
    segment_files: list[Path] = []

    for index, (visual, duration) in enumerate(zip(visual_files, scene_durations), start=1):
        segment = WORK_DIR / f"segment_{index:03d}.mp4"
        segment_files.append(segment)

        if visual.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"}:
            command = [
                ffmpeg, "-y",
                "-loop", "1", "-i", str(visual),
                "-t", f"{duration:.3f}",
                "-vf", (
                    f"scale={VIDEO_W}:{VIDEO_H}:force_original_aspect_ratio=increase,"
                    f"crop={VIDEO_W}:{VIDEO_H},setsar=1,fps={FPS}"
                ),
                "-an",
                "-c:v", "libx264", "-preset", "veryfast", "-crf", "24",
                "-pix_fmt", "yuv420p", str(segment),
            ]
        else:
            command = [
                ffmpeg, "-y",
                "-stream_loop", "-1", "-i", str(visual),
                "-t", f"{duration:.3f}",
                "-vf", (
                    f"scale={VIDEO_W}:{VIDEO_H}:force_original_aspect_ratio=increase,"
                    f"crop={VIDEO_W}:{VIDEO_H},setsar=1,fps={FPS}"
                ),
                "-an",
                "-c:v", "libx264", "-preset", "veryfast", "-crf", "24",
                "-pix_fmt", "yuv420p", str(segment),
            ]

        run_command(command)
        concat_inputs.append(segment)

    concat_list = WORK_DIR / "concat.txt"
    with concat_list.open("w", encoding="utf-8") as handle:
        for path in concat_inputs:
            handle.write(f"file '{path.as_posix().replace("'", "'\\''")}'\n")

    visual_track = WORK_DIR / "visual_track.mp4"
    run_command([
        ffmpeg, "-y", "-f", "concat", "-safe", "0", "-i", str(concat_list),
        "-an", "-c", "copy", str(visual_track)
    ])

    # Match final visuals to narration duration exactly, add music under it.
    narration_duration = ffprobe_duration(narration)
    total_duration = max(narration_duration, 1.0)

    run_command([
        ffmpeg, "-y",
        "-i", str(visual_track),
        "-i", str(narration),
        "-stream_loop", "-1", "-i", str(music),
        "-filter_complex",
        (
            f"[2:a]volume=0.075,atrim=duration={total_duration:.3f},"
            f"asetpts=PTS-STARTPTS[m];"
            f"[1:a]acompressor=threshold=-18dB:ratio=3:attack=20:release=250[voice];"
            f"[m][voice]sidechaincompress=threshold=0.02:ratio=8:attack=20:release=400[ducked];"
            f"[voice][ducked]amix=inputs=2:duration=longest:dropout_transition=2[aout]"
        ),
        "-map", "0:v:0",
        "-map", "[aout]",
        "-t", f"{total_duration:.3f}",
        "-c:v", "libx264", "-preset", "veryfast", "-crf", "22",
        "-pix_fmt", "yuv420p", "-r", str(FPS),
        "-c:a", "aac", "-b:a", "192k",
        "-movflags", "+faststart",
        str(output),
    ])


# ============================================================
# YOUTUBE
# ============================================================


def write_youtube_token() -> Path:
    if not YOUTUBE_TOKEN_JSON:
        raise RuntimeError("YOUTUBE_TOKEN_JSON is not set.")
    token_path = WORK_DIR / "youtube_token.json"
    token_path.write_text(YOUTUBE_TOKEN_JSON, encoding="utf-8")
    return token_path


def get_youtube_service():
    token_path = write_youtube_token()
    scopes = [
        "https://www.googleapis.com/auth/youtube.upload",
        "https://www.googleapis.com/auth/youtube",
    ]
    credentials = Credentials.from_authorized_user_file(str(token_path), scopes)

    if credentials.expired and credentials.refresh_token:
        log("🔄 Refreshing YouTube token...")
        credentials.refresh(Request())
        token_path.write_text(credentials.to_json(), encoding="utf-8")

    if not credentials.valid:
        raise RuntimeError("YouTube credentials are invalid or missing a refresh token.")

    return build("youtube", "v3", credentials=credentials, cache_discovery=False)


def upload_to_youtube(video_path: Path, story: dict[str, Any]) -> str:
    youtube = get_youtube_service()

    title = story["title"][:100]
    description = (
        story["description"].strip()
        + "\n\nThe Fear Network — original horror storytelling."
        + "\n\n#shorts" if VIDEO_MODE == "short"
        else story["description"].strip() + "\n\nThe Fear Network — original horror storytelling."
    )

    body = {
        "snippet": {
            "title": title,
            "description": description[:5000],
            "tags": list(dict.fromkeys((story.get("tags") or []) + ["horror", "scary story", "The Fear Network"]))[:30],
            "categoryId": "24",
        },
        "status": {
            "privacyStatus": "private",
            "selfDeclaredMadeForKids": False,
        },
    }

    media = MediaFileUpload(str(video_path), mimetype="video/mp4", resumable=True, chunksize=8 * 1024 * 1024)
    request = youtube.videos().insert(part="snippet,status", body=body, media_body=media)

    response = None
    while response is None:
        status, response = request.next_chunk()
        if status:
            log(f"📤 YouTube upload: {status.progress() * 100:.1f}%")

    video_id = response.get("id")
    if not video_id:
        raise RuntimeError("YouTube returned no video ID.")

    url = f"https://www.youtube.com/watch?v={video_id}"
    log(f"✅ YouTube upload complete: {url}")
    return url


# ============================================================
# CLEANUP
# ============================================================


def clean_work() -> None:
    for child in WORK_DIR.iterdir():
        if child.is_dir():
            shutil.rmtree(child, ignore_errors=True)
        else:
            try:
                child.unlink()
            except OSError:
                pass
    MEDIA_DIR.mkdir(parents=True, exist_ok=True)
    AUDIO_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# MAIN
# ============================================================


def main() -> int:
    log("=" * 72)
    log(f"THE FEAR NETWORK — {VIDEO_MODE.upper()} FACTORY")
    log("=" * 72)

    if VIDEO_MODE not in {"long", "short"}:
        raise RuntimeError("VIDEO_MODE must be 'long' or 'short'.")

    required = {
        "PEXELS_API_KEY": PEXELS_API_KEY,
        "PIXABAY_API_KEY": PIXABAY_API_KEY,
        "YOUTUBE_TOKEN_JSON": YOUTUBE_TOKEN_JSON,
    }
    missing = [name for name, value in required.items() if not value]
    if missing:
        raise RuntimeError("Missing secrets: " + ", ".join(missing))

    if not VOICE_WAV.exists():
        raise FileNotFoundError(f"Missing voice reference: {VOICE_WAV}")

    clean_work()

    idea = generate_idea()
    log(f"\n💡 IDEA:\n{idea}\n")

    story = generate_story(idea)
    log(f"📝 TITLE: {story['title']}")
    log(f"📝 NARRATION WORDS: {len(story['narration'].split())}")

    narration_path = AUDIO_DIR / "narration.wav"
    make_narration(story["narration"], narration_path)
    narration_duration = ffprobe_duration(narration_path)
    log(f"⏱️ Narration duration: {narration_duration:.2f} seconds")

    scenes = normalize_scene_plan(story)
    scene_durations = estimate_scene_durations(scenes, narration_duration)
    log(f"🎬 Scenes: {len(scenes)}")

    visual_files = prepare_visuals(scenes)
    music = prepare_music(narration_duration)

    output_name = safe_filename(story["title"], "fear_video") + ".mp4"
    output_path = OUTPUT_DIR / output_name

    log("\n🎞️ Rendering final video...")
    render_video(narration_path, visual_files, scene_durations, music, output_path)
    log(f"✅ Video created: {output_path}")

    youtube_url = upload_to_youtube(output_path, story)
    log(f"\n🎉 DONE: {youtube_url}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        log("\n⏹️ Interrupted.")
        raise SystemExit(130)
    except Exception as error:
        log(f"\n❌ FATAL: {type(error).__name__}: {error}")
        raise SystemExit(1)
