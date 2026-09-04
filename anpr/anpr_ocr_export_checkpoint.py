"""Save Awiros/anpr-ocr's safetensors weights as a native Paddle checkpoint,
so PaddleOCR's own tools/train.py can start a fine-tune from
Global.pretrained_model (it doesn't read safetensors directly).

    ./.venv-paddle/bin/python -m anpr.anpr_ocr_export_checkpoint

Writes models/anpr_ocr/finetune_data/base.pdparams - the exact same weights
anpr_ocr_reader.py already loads at inference time, just persisted in
Paddle's own format. Never touches the original model.safetensors.
"""
from __future__ import annotations

from anpr import anpr_ocr_reader as m


def main():
    reader = m.load()
    if reader is None:
        raise SystemExit("could not load anpr_ocr - is paddle installed in this venv?")
    out_path = f"{m._CACHE_DIR}/finetune_data/base.pdparams"
    import os
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    reader._paddle.save(reader._model.state_dict(), out_path)
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
