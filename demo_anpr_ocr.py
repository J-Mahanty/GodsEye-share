"""Standalone demo: upload a photo, get the plate back.

Pipeline: YOLOv8-OBB plate detector -> Awiros/anpr-ocr recogniser
(anpr/anpr_ocr_reader.py), the combination measured at 91.0% per-frame
exact-match / 93.8% mean over an 8-seed vehicle split on the real-photo
corpus (see CLAUDE.md). This is evaluation-only wiring, identical in shape
to compare_backends.py's --real-frames path - it does not touch sim/api/
dashboard's default engine.

Needs paddlepaddle, so it must run under .venv-paddle, not the repo's main
.venv:

    .venv-paddle/bin/python demo_anpr_ocr.py

Then open http://localhost:8010
"""
from __future__ import annotations

import io

import cv2
import numpy as np
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import HTMLResponse, JSONResponse

from anpr.anpr_ocr_reader import load as load_anpr_ocr
from anpr.ocr import ANPREngine

app = FastAPI(title="GodsEye ANPR demo")
_engine: ANPREngine | None = None


def get_engine() -> ANPREngine:
    global _engine
    if _engine is None:
        # Do not call anpr_ocr_reader.available() here - it does a bare
        # `import paddle`, and importing paddle before load()'s internal
        # torch model loads (_load_yolo(), crnn.load()) reproducibly
        # corrupts torch's tensors (see CLAUDE.md's load-order hazard).
        # load() does its own availability check internally and returns
        # None if paddle isn't importable.
        reader = load_anpr_ocr()
        if reader is None:
            raise RuntimeError(
                "anpr_ocr backend failed to load (see server logs) - check network access "
                "for the first-run PaddleOCR clone / HF weight download"
            )
        _engine = ANPREngine(anpr_ocr_reader=reader)
    return _engine


PAGE = """<!doctype html>
<html>
<head>
<meta charset="utf-8">
<title>GodsEye ANPR demo</title>
<style>
  :root { color-scheme: light dark; }
  body { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
         max-width: 640px; margin: 48px auto; padding: 0 20px; }
  h1 { font-size: 1.3rem; margin-bottom: 4px; }
  .sub { color: #888; font-size: 0.9rem; margin-bottom: 28px; }
  #drop { border: 2px dashed #999; border-radius: 12px; padding: 36px;
          text-align: center; cursor: pointer; transition: border-color .15s; }
  #drop.over { border-color: #2b6cf6; }
  #drop p { margin: 0; color: #777; }
  #preview { max-width: 100%; margin-top: 20px; border-radius: 8px; display: none; }
  #result { margin-top: 24px; display: none; }
  .plate { font-family: "Courier New", monospace; font-size: 2rem; font-weight: 700;
           letter-spacing: 2px; padding: 12px 20px; border-radius: 8px; display: inline-block;
           border: 3px solid #222; background: #f6c945; color: #111; }
  .meta { margin-top: 12px; font-size: 0.9rem; color: #888; line-height: 1.6; }
  .badge { display: inline-block; padding: 2px 8px; border-radius: 4px; font-size: 0.75rem;
           font-weight: 600; margin-left: 8px; }
  .ok { background: #1e8e3e22; color: #1e8e3e; }
  .low { background: #d9363022; color: #d93630; }
  #err { color: #d93630; margin-top: 16px; display: none; }
  #spin { display: none; margin-top: 16px; color: #888; }
  input[type=file] { display: none; }
</style>
</head>
<body>
<h1>GodsEye plate reader</h1>
<div class="sub">YOLOv8-OBB detector &rarr; anpr-ocr recogniser (91% exact-match, real-photo corpus)</div>

<label id="drop" for="file">
  <p>Drop a photo here, or click to choose one</p>
</label>
<input id="file" type="file" accept="image/*">
<img id="preview">
<div id="spin">reading plate&hellip;</div>
<div id="err"></div>

<div id="result">
  <div class="plate" id="plateText"></div>
  <span class="badge" id="badge"></span>
  <div class="meta" id="meta"></div>
</div>

<script>
const drop = document.getElementById('drop');
const fileInput = document.getElementById('file');
const preview = document.getElementById('preview');
const result = document.getElementById('result');
const err = document.getElementById('err');
const spin = document.getElementById('spin');

drop.addEventListener('dragover', e => { e.preventDefault(); drop.classList.add('over'); });
drop.addEventListener('dragleave', () => drop.classList.remove('over'));
drop.addEventListener('drop', e => {
  e.preventDefault();
  drop.classList.remove('over');
  if (e.dataTransfer.files.length) handleFile(e.dataTransfer.files[0]);
});
fileInput.addEventListener('change', () => {
  if (fileInput.files.length) handleFile(fileInput.files[0]);
});

function handleFile(file) {
  preview.src = URL.createObjectURL(file);
  preview.style.display = 'block';
  result.style.display = 'none';
  err.style.display = 'none';
  spin.style.display = 'block';

  const form = new FormData();
  form.append('image', file);
  fetch('/predict', { method: 'POST', body: form })
    .then(r => r.json())
    .then(data => {
      spin.style.display = 'none';
      if (data.error) { err.textContent = data.error; err.style.display = 'block'; return; }
      document.getElementById('plateText').textContent = data.plate_found ? data.pretty : '— not found —';
      const badge = document.getElementById('badge');
      badge.textContent = data.accepted ? 'accepted' : 'below confidence floor';
      badge.className = 'badge ' + (data.accepted ? 'ok' : 'low');
      document.getElementById('meta').innerHTML =
        `confidence: ${(data.confidence * 100).toFixed(1)}%` +
        (data.reason ? `<br>${data.reason}` : '');
      result.style.display = 'block';
    })
    .catch(e => {
      spin.style.display = 'none';
      err.textContent = 'request failed: ' + e;
      err.style.display = 'block';
    });
}
</script>
</body>
</html>
"""


@app.get("/", response_class=HTMLResponse)
def index():
    return PAGE


def _to_jsonable(obj):
    """boxes/candidates come back as tuples of numpy.int64 - plain json
    doesn't know what to do with those, so coerce recursively."""
    if isinstance(obj, dict):
        return {k: _to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_jsonable(v) for v in obj]
    if isinstance(obj, np.generic):
        return obj.item()
    return obj


@app.post("/predict")
async def predict(image: UploadFile = File(...)):
    raw = await image.read()
    frame = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR)
    if frame is None:
        return JSONResponse({"error": "could not decode image"}, status_code=400)
    try:
        engine = get_engine()
    except RuntimeError as e:
        return JSONResponse({"error": str(e)}, status_code=500)
    result = engine.read_frame(frame)
    return JSONResponse(_to_jsonable(result.as_dict()))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8010)
