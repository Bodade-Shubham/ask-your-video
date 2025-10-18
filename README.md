# Ask Your Video

Minimal demo that exposes a Fastify API backed by OpenAI embeddings/completions and a static HTML UI for asking questions about a video transcript. The repository is split into two folders:

- `api/` – Fastify server that embeds transcript segments, ranks them against a question, and calls OpenAI for the final answer.
- `ui/` – Static HTML/CSS/JS client that calls the API and renders answers.

## Prerequisites

- Node.js 18+ (the OpenAI SDK requires a modern runtime).
- An OpenAI API key with access to `text-embedding-3-small` and `gpt-3.5-turbo`.

## 1. Prepare transcript data

The API expects `api/src/segment.json`. The file can be either:

```json
[
  { "text": "…", "start": 0, "end": 9.8 },
  { "text": "…", "start": 9.8, "end": 22.1 }
]
```

or wrapped in an object:

```json
{ "segments": [ /* same objects as above */ ] }
```

Only `text` is required; `start`/`end`/`id` are optional but improve the answer context.

### Generate `segment.json` from an MP4 (Python + Whisper)

The easiest way to build the transcript is with OpenAI's Whisper model via the `openai-whisper` Python package (runs locally). Requirements:

- Python 3.9+
- `ffmpeg` on your PATH (`brew install ffmpeg` on macOS, `apt install ffmpeg` on Debian/Ubuntu)
- GPU is optional; Whisper works on CPU but takes longer

Install the tooling:

```bash
pip install openai-whisper
```

Create `scripts/build_segments.py` with:

```python
#!/usr/bin/env python3
import argparse
import json
from pathlib import Path

import whisper


def transcribe(video_path: Path, model_size: str = "small"):
  model = whisper.load_model(model_size)
  result = model.transcribe(str(video_path), verbose=False)

  segments = []
  for seg in result.get("segments", []):
    text = seg.get("text", "").strip()
    if not text:
      continue
    segments.append(
      {
        "text": text,
        "start": round(seg.get("start", 0.0), 2),
        "end": round(seg.get("end", 0.0), 2),
      }
    )

  return {"segments": segments}


def main():
  parser = argparse.ArgumentParser(description="Transcribe video and emit segment.json")
  parser.add_argument("video", type=Path, help="Path to the input .mp4 file")
  parser.add_argument(
    "-o",
    "--output",
    type=Path,
    default=Path("api/src/segment.json"),
    help="Where to write the JSON (default: api/src/segment.json)",
  )
  parser.add_argument(
    "-m",
    "--model",
    default="small",
    choices=["tiny", "base", "small", "medium", "large"],
    help="Whisper model size (trade-off between speed and accuracy)",
  )
  args = parser.parse_args()

  data = transcribe(args.video, args.model)
  args.output.parent.mkdir(parents=True, exist_ok=True)
  args.output.write_text(json.dumps(data, indent=2), encoding="utf-8")
  print(f"Wrote {len(data.get('segments', []))} segments to {args.output}")


if __name__ == "__main__":
  main()
```

Then run:

```bash
python scripts/build_segments.py videos/my-video.mp4
```

The script loads the requested Whisper model, transcribes the MP4, and writes `api/src/segment.json` with one entry per Whisper segment. Adjust the model size (`--model medium`) if you need higher accuracy and have the hardware.

## 2. Run the API

```bash
cd api
npm install
OPENAI_API_KEY=sk-... npm start
```

The server listens on `PORT` (defaults to 3000). If the port is taken it will try up to 10 additional ports.

### Endpoints

- `GET /health` → `{ "ok": true }`
- `GET /ask?q=your+question` – simple query test endpoint.
- `POST /ask` with body `{ "question": "your question" }`

Successful `/ask` responses look like:

```json
{
  "ok": true,
  "answer": "Concise reply that references transcript timestamps.",
  "sources": [
    { "start": 12.3, "end": 30.7, "score": 0.78, "text": "Snippet…" }
  ]
}
```

Error responses include an `error` string and an HTTP status that explains the failure (missing API key, missing segments, etc.).

## 3. Run the UI

The UI is a single static page. Serve it with any static file server while the API is running:

```bash
cd ui
python3 -m http.server 4173
```

Open `http://localhost:4173` in a browser. Update the `API_BASE` constant in `ui/index.html` if your API runs on a different host or port.

## Project structure

```
api/
  src/
    index.js           # Fastify server + route registration
    services/
      askService.js    # Business logic: embeddings, ranking, OpenAI call
    segment.json       # Transcript segments (sample / replace with your data)
ui/
  index.html           # Static client
videos/                # Placeholder for source media (not used directly)
```

## Development notes

- The API caches segment embeddings in memory. Restart the server if you change `segment.json`.
- For hot reload during development you can use `node --watch src/index.js` or add a watcher (e.g., nodemon) to `package.json`.
- The UI currently expects the API response shape shown above; adjust the client if you rename properties.
