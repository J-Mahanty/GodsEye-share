# Running the 91% real-photograph reader — Windows & Linux

The platform's default reader is a CRNN trained on synthetic captures. It reads
**0.9%** of real photographs. The **91.0%** number quoted in `README.md` comes
from a different backend — `anpr_ocr`, a wrapper around
[`Awiros/anpr-ocr`](https://huggingface.co/Awiros/anpr-ocr) (PP-OCRv5,
PPHGNetV2_B4, Apache-2.0), pretrained on 558,767 real Indian plates, driven by
this repo's own grammar-constrained decoder.

That backend needs **PaddlePaddle**, and paddle has **no Python 3.14 wheels**.
If your main environment is 3.13 or 3.14 it physically cannot run this.
Everything below builds a second, separate environment on Python 3.12 alongside
whatever you already have.

Nothing here touches your existing environment, and the platform keeps running
normally without any of it.

> Commands are given for **Windows (PowerShell)** and **Linux (bash)**. Run
> everything from the project root — the corpus loaders resolve by relative
> path. On Windows use PowerShell, not `cmd.exe`.

---

## Before you start

| | |
|---|---|
| Time | ~15 minutes, most of it downloading |
| Disk | **~4 GB** (env + 2.3 GB weights and PaddleOCR clone) |
| Network | required — HF weights and a GitHub clone are fetched on first run |
| `git` | required, and must be on `PATH` — the first run shells out to it |
| Corpus | bundled — `dataset_v1_bbox/`, `indian-number-plates-dataset/`, `real_plate_crops/` |
| GPU | **not used.** See [GPU](#a-note-on-gpus) — the device is pinned to CPU in code |

**Two different goals, and they need different things.** Read this before you
start, because one of them cannot be done from the share bundle:

1. **Run the 91% reader on your own photographs** — needs only the steps below.
   Works from the share bundle.
2. **Reproduce the 91.0% / 93.8% benchmark numbers** — additionally needs the
   real-photo corpus (`dataset_v1_bbox/`, `indian-number-plates-dataset/`,
   `real_plate_crops/`, `real_plate_frame_labels.json`,
   `build_real_crnn_dataset.py`). That corpus **is included** in both the repo
   and the share bundle, ~200 MB of it. `indian-number-plates-dataset` is
   CC BY-NC-ND 4.0 — attribution, non-commercial, no modified redistribution —
   so keep it inside the team. See
   [Reproducing the benchmark](#5-reproducing-the-benchmark-numbers).

---

## 1. Get Python 3.12

Check what you have:

```powershell
# Windows
py -3.12 --version
```
```bash
# Linux
python3.12 --version
```

If that prints `Python 3.12.x`, skip to step 2.

### Windows

Either the official installer from
[python.org/downloads/release/python-31213](https://www.python.org/downloads/release/python-3120/)
— tick **"Add python.exe to PATH"** — or:

```powershell
winget install Python.Python.3.12
```

Then reopen PowerShell and confirm `py -3.12 --version` works. The `py`
launcher is installed with Python on Windows and is the reliable way to pick a
version when several are installed.

Also install **Git for Windows** if you do not have it
([git-scm.com/download/win](https://git-scm.com/download/win)) — step 4 shells
out to `git clone` and will fail without it.

### Linux — Ubuntu / Debian

```bash
sudo add-apt-repository ppa:deadsnakes/ppa
sudo apt update
sudo apt install python3.12 python3.12-venv python3.12-dev git
```

`python3.12-venv` is a separate package on Debian-family systems, and step 2
fails with `ensurepip is not available` without it.

### Linux — Fedora / RHEL

```bash
sudo dnf install python3.12 python3.12-devel git
```

### Any distribution — pyenv

```bash
curl https://pyenv.run | bash
pyenv install 3.12.13
```

> **Why exactly 3.12?** Paddle publishes wheels for 3.9–3.12. 3.13 is
> unreliable, 3.14 has nothing at all. 3.12 is the newest version that simply
> works.

## 2. Create the environment

From the project root:

```powershell
# Windows
py -3.12 -m venv .venv-paddle
```
```bash
# Linux
python3.12 -m venv .venv-paddle
```

This creates `.venv-paddle/` next to any environment you already have. They do
not interact. `.gitignore` does not currently list `.venv-paddle/`, so it will
show as untracked — leave it that way, never commit it.

Verify before going further:

```powershell
# Windows -- must say 3.12.x
.\.venv-paddle\Scripts\python.exe --version
```
```bash
# Linux -- must say 3.12.x
.venv-paddle/bin/python --version
```

> **Do not activate the environment.** Every command below calls the
> interpreter by its full path. That is deliberate: with two environments in
> one project, an activated shell is how you end up running paddle code against
> the wrong interpreter and getting a confusing `ModuleNotFoundError`. Explicit
> paths cannot be got wrong.
>
> If you do want to activate it on Windows, PowerShell blocks the script by
> default; `Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass` for
> that session first.

## 3. Install the dependencies

The full stack, because this environment runs the whole pipeline — the YOLOv8
detector, the CRNN, and paddle.

**Windows**

```powershell
.\.venv-paddle\Scripts\python.exe -m pip install --upgrade pip
.\.venv-paddle\Scripts\python.exe -m pip install -r requirements.txt
.\.venv-paddle\Scripts\python.exe -m pip install paddlepaddle huggingface_hub safetensors
.\.venv-paddle\Scripts\python.exe -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
```

**Linux**

```bash
.venv-paddle/bin/pip install --upgrade pip
.venv-paddle/bin/pip install -r requirements.txt
.venv-paddle/bin/pip install paddlepaddle huggingface_hub safetensors
.venv-paddle/bin/pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
```

`requirements.txt` marks `torch`, `paddlepaddle` and `huggingface_hub` as
`extra == "neural"`, so a plain `-r requirements.txt` install skips them — the
two extra lines are what actually install them.

Check it took:

```bash
# Linux -- on Windows substitute .\.venv-paddle\Scripts\python.exe
.venv-paddle/bin/python -c "import paddle; print('paddle', paddle.__version__)"
.venv-paddle/bin/python -c "import torch; print('torch', torch.__version__)"
```

Known-good versions: paddle 3.3.1, torch 2.14.0, numpy 2.5.2,
opencv 5.0.0.93, Python 3.12.13.

### A note on GPUs

Unlike macOS, Linux and Windows do have CUDA builds of paddle
(`paddlepaddle-gpu`). **Installing one will not make this faster**, because the
reader pins the device in code:

```python
# anpr/anpr_ocr_reader.py:459
paddle.set_device("cpu")
```

Change that line to `paddle.set_device("gpu")` if you have CUDA and want to try
it, and install the matching wheel from
[paddlepaddle.org.cn/install](https://www.paddlepaddle.org.cn/en/install/quick)
— the correct index URL depends on your CUDA version, so take it from that page
rather than copying one from here. This is untested in this repo; the numbers in
`README.md` were all measured on CPU. Recognition is not the bottleneck in the
benchmark anyway — it is a small model reading one crop at a time.

The same applies to torch: the `--index-url .../cpu` above is deliberate. Swap
it for a CUDA index if you want the YOLOv8 detector on a GPU, which does help
on `--real-frames`.

## 4. First run — it downloads ~2.3 GB

```powershell
# Windows
.\.venv-paddle\Scripts\python.exe -c "from anpr import anpr_ocr_reader; print('loaded:', anpr_ocr_reader.load() is not None)"
```
```bash
# Linux
.venv-paddle/bin/python -c "from anpr import anpr_ocr_reader; print('loaded:', anpr_ocr_reader.load() is not None)"
```

The first call does three things, all cached under `models/anpr_ocr/`:

1. git-clones `PaddlePaddle/PaddleOCR` (~246 MB) — the `ppocr` package that
   builds this architecture is not on PyPI standalone, so the model's own
   `test.py` bootstraps it this way and `_ensure_paddleocr()` does the same;
2. downloads `model.safetensors` and `en_dict.txt` from Hugging Face;
3. builds the network and loads the weights.

Expect several minutes. It prints `loaded: True` when it works. Subsequent runs
take a few seconds. None of `models/anpr_ocr/` is committed.

If it prints `loaded: False`, `load()` swallowed an exception — no network, no
`git` on `PATH`, a failed clone, or a partial download. Delete
`models/anpr_ocr/` and retry.

> **Windows: enable long paths first.** The PaddleOCR clone contains paths that
> exceed the legacy 260-character `MAX_PATH` limit, and the clone fails with
> `Filename too long` on a default Windows install. Fix it once, as
> Administrator:
>
> ```powershell
> git config --system core.longpaths true
> New-ItemProperty -Path "HKLM:\SYSTEM\CurrentControlSet\Control\FileSystem" `
>   -Name "LongPathsEnabled" -Value 1 -PropertyType DWORD -Force
> ```
>
> Then reopen PowerShell. Keeping the project close to the drive root
> (`C:\GodsEye`, not a deep folder under `Documents`) avoids the problem too.

### Reading one photograph

```bash
# Linux -- on Windows substitute the Scripts\python.exe path
.venv-paddle/bin/python -c "
import cv2
from anpr.ocr import ANPREngine
from anpr import anpr_ocr_reader

eng = ANPREngine(anpr_ocr_reader=anpr_ocr_reader.load())
img = cv2.imread('real_plate_crops/00_video10.jpg')   # a tight plate crop
print(eng.read(img).text)
"
```

Verified output for that crop: `MH02EK0837`, confidence 0.99.

For a whole photograph rather than a crop, use `eng.read_frame(img)` — that
routes through the YOLOv8-OBB detector to find the plate first.

## 5. Reproducing the benchmark numbers

The corpus ships with both the repo and the share bundle, so these run
anywhere you have unpacked it. Run them **from the project root** — the loaders
resolve the corpus by relative path.

```bash
# 91.0% -- every labelled real crop, no vehicle split. The stable comparison.
.venv-paddle/bin/python -m anpr.compare_backends --real-all

# 93.8% mean / 86.8% min / 97.1% max -- held out by vehicle, 8 splits
.venv-paddle/bin/python -m anpr.compare_backends --real --seeds 8

# 91.5% mean -- whole photographs end to end, detector included
.venv-paddle/bin/python -m anpr.compare_backends --real-frames --seeds 8
```

On Windows, replace `.venv-paddle/bin/python` with
`.\.venv-paddle\Scripts\python.exe` in each. Add `--json out.json` to write the
per-seed results, which is what `models/backend_comparison_real.json` and chart
06 are built from.

> **A single seed is not a number.** With ~118 unique vehicles, which ones land
> in validation moves the result by ~10 points. Always quote the multi-seed
> mean. Note also that `--seed` is synthetic-mode only and `--real-seed` is a
> separate flag — the two used to be silently conflated, which hid a 19-point
> swing.

## 6. The load-order hazard — read this before writing your own script

**Importing `paddle` and then later loading a torch model in the same process
reproducibly corrupts torch's tensors**, with:

```
PreconditionNotMet: Tensor holds no memory
```

It is a binary symbol clash between the two frameworks' compiled extensions,
not a bug in this repo, and it is not platform-specific. `anpr_ocr_reader.load()`
already defends against it — it loads the YOLO detector and the CRNN *first*,
then paddle:

```python
from anpr.ocr import _load_yolo
_load_yolo()
from anpr import crnn as _crnn_mod
_crnn_mod.load()
# ... only now does it import paddle
```

So in your own code: **call `anpr_ocr_reader.load()` before anything that
touches torch**, or let it be the thing that initialises both. Do not `import
paddle` at the top of a module that also uses the CRNN or the detector.

## Troubleshooting

| Symptom | Cause and fix |
|---|---|
| `No matching distribution found for paddlepaddle` | Wrong Python. Check the venv's interpreter says 3.12 — not 3.13 or 3.14. |
| `ensurepip is not available` (Linux) | `sudo apt install python3.12-venv`, then recreate the venv. |
| `Filename too long` during the clone | Windows `MAX_PATH`. Enable long paths — see the box in step 4. |
| `'git' is not recognized` / `git: command not found` | `git` is not on `PATH`. Install Git for Windows, or `apt install git`. |
| `running scripts is disabled on this system` | PowerShell execution policy, only if you activated the venv. `Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass`. |
| `ModuleNotFoundError: No module named 'ppocr'` | First-run clone failed. Delete `models/anpr_ocr/PaddleOCR` and re-run step 4. |
| `PreconditionNotMet: Tensor holds no memory` | Load order. See section 6 — paddle was imported before a torch model loaded. |
| `load()` returns `None` | Exception swallowed by design. Call `_ensure_paddleocr()` and `_ensure_weights()` directly to see the real error. |
| `FileNotFoundError: dataset_v1_bbox/...` | Not run from the project root — the loaders use relative paths. `cd` to the folder containing `config.py` first. |
| Accuracy differs from the README | Check you used `--real-seed`/`--seeds`, not `--seed`. And quote the multi-seed mean, not one split. |
| Very slow | Expected. The device is pinned to CPU — see [GPU](#a-note-on-gpus). The 8-seed sweeps take a while. |

## What this does not change

`anpr_ocr` is **evaluation-only and opt-in**. It is passed in through
`ANPREngine(anpr_ocr_reader=...)` and is never picked up implicitly, so the
simulator, the API and the dashboard are all unaffected and continue to use the
CRNN and classical readers. Setting this up adds a second environment; it does
not alter the running platform.

Fine-tuning `anpr_ocr` on this repo's 439-crop real training set was tried and
**did not help** — validation accuracy plateaued at ~49.5% by epoch 4 while
train loss kept falling, which is overfitting on too little real data. See
`CLAUDE.md`. The 91.0% comes from the base weights plus a better decoder, not
from any retraining.

---

*The behavioural facts here — that `load()` returns `True`, that the sample crop
reads `MH02EK0837`, the pinned versions, the load-order hazard — were verified
by running them. That verification was on macOS/Apple Silicon; the Windows and
Linux command forms are adapted, not separately tested on those platforms. The
Python-level behaviour is identical, but if you hit a packaging difference,
that is where it will be.*
