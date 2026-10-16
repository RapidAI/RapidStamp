"""Seal classes and the preprocessing contract shared by train and inference."""

from __future__ import annotations

CLASS_NAMES: list[str] = [
    "round_official",
    "oval",
    "square_personal",
    "finance",
    "invoice",
    "company_latin",
    "notary",
    "rect_stamp",
]

# circle / ellipse use polar unwrapping. grid is a personal-seal cell unroll.
# hline is a horizontal band for rectangular rubber stamps.
CLASS_LAYOUT: dict[str, str] = {
    "round_official": "circle",
    "oval": "ellipse",
    "square_personal": "grid",
    "finance": "ellipse",
    "invoice": "ellipse",
    "company_latin": "circle",
    "notary": "circle",
    "rect_stamp": "hline",
}

DET_SIZE = 320
GRID = 20
REC_H = 32
REC_W = 384
UNWRAP_R_IN = 0.72
UNWRAP_R_OUT = 0.99
UNWRAP_START_DEG = 140.0
UNWRAP_SPAN_DEG = 360.0
PAD_VALUE = 114
SCORE_THRESH = 0.25
NMS_THRESH = 0.45
ANGLE_STEPS = 24
# Raw box-size logits are clamped before exp so a bad output cannot explode.
BOX_LOGIT_LIMIT = 4.0


def class_id(name: str) -> int:
    return CLASS_NAMES.index(name)


def write_preprocess(path: str) -> None:
    """Flat runtime contract. C++ and Python both load this file."""
    lines = [
        f"det_size {DET_SIZE}",
        f"grid {GRID}",
        f"rec_h {REC_H}",
        f"rec_w {REC_W}",
        f"r_in {UNWRAP_R_IN}",
        f"r_out {UNWRAP_R_OUT}",
        f"start_deg {UNWRAP_START_DEG}",
        f"span_deg {UNWRAP_SPAN_DEG}",
        f"pad_value {PAD_VALUE}",
        f"score_thresh {SCORE_THRESH}",
        f"nms_thresh {NMS_THRESH}",
        f"angle_steps {ANGLE_STEPS}",
        f"box_logit_limit {BOX_LOGIT_LIMIT}",
    ]
    for name, layout in CLASS_LAYOUT.items():
        lines.append(f"layout {name} {layout}")
    with open(path, "w", encoding="utf-8") as handle:
        handle.write("\n".join(lines) + "\n")


def load_preprocess(path: str) -> dict:
    data: dict = {"layout": {}}
    with open(path, encoding="utf-8") as handle:
        for raw in handle:
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            key = parts[0]
            if key == "layout" and len(parts) == 3:
                data["layout"][parts[1]] = parts[2]
            elif len(parts) == 2:
                value: int | float
                if "." in parts[1]:
                    value = float(parts[1])
                else:
                    value = int(parts[1])
                data[key] = value
    return data
