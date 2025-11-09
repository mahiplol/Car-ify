# api/app.py
import os, json
from pathlib import Path
from typing import Optional, Dict, Any
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

# ---------- helpers ----------
def _s(x: Any) -> str:
    """Coerce any value to a short safe string."""
    if x is None:
        return "?"
    try:
        return str(x)
    except Exception:
        return "?"

def _i(x: Any, default: int = 0) -> int:
    try:
        return int(x)
    except Exception:
        return default

def _fmt_money(n: Optional[int]) -> str:
    return f"${n:,}" if isinstance(n, int) else "?"

def _specs_public(specs: dict) -> dict:
    """Return a small, string-safe specs dict for the client overlay."""
    return {
        "drivetrain": _s(specs.get("Drivetrain")),
        "engine": _s(specs.get("EngineType")),
        "displacement": _s(specs.get("Displacement")),
        "power": _s(specs.get("SAEhpRPM")),
        "mpg": _s(specs.get("GasMileage")),
        "body": _s(specs.get("BodyStyle")),
    }


# ---------- load data ----------
ROOT = Path(__file__).resolve().parents[1]

def _load_json(p: Path) -> dict:
    return json.load(open(p, "r", encoding="utf-8")) if p.exists() else {}

LABEL_MAP = _load_json(ROOT / "data" / "car_cls_triplet" / "label_map.json")
SPECS_MAP = _load_json(ROOT / "tools" / "_analysis" / "class_specs.json")

# ---------- optional OpenAI ----------
try:
    from openai import OpenAI
    oai = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
except Exception:
    oai = None

app = FastAPI(title="Car Dossier API", version="0.1.1")
_cache: Dict[str, Dict] = {}  # memory cache keyed by label

class DossierReq(BaseModel):
    label: Optional[str] = None
    make: Optional[str] = None
    model: Optional[str] = None
    year: Optional[int] = None

class DossierResp(BaseModel):
    title: str
    price: Optional[Dict[str, Any]] = None
    specs: Dict[str, str] = Field(default_factory=dict)
    highlights: Optional[list[str]] = None
    caveats: Optional[list[str]] = None
    summary: str

def _norm_from_label(label: str):
    meta = LABEL_MAP.get(label)
    if meta:
        return _s(meta.get("make")), _s(meta.get("model")), _i(meta.get("year"), 0)
    parts = label.split("_", 2)
    mk = _s(parts[0]).title() if len(parts) > 0 else "Unknown"
    md = _s(parts[1]).upper() if len(parts) > 1 else "?"
    yr = _i(parts[2], 0) if len(parts) > 2 else 0
    return mk, md, yr

def _find_specs_key(mk: str, md: str, yr: int):
    key = f"{mk.lower()}_{md.lower()}_{yr}"
    if key in SPECS_MAP:
        return key
    # fallback to any year of same make/model
    prefix = f"{mk.lower()}_{md.lower()}_"
    for k in SPECS_MAP.keys():
        if k.startswith(prefix):
            return k
    return None

def _estimate_price(msrp: Optional[str], yr: int):
    # very rough: 15% first year + 10%/yr afterwards, floor at 15% residual
    try:
        base = int(str(msrp))
    except Exception:
        return None
    age = max(0, 2025 - yr)
    d = 0.15 + max(0, age - 1) * 0.10 if age > 0 else 0.0
    d = max(0.0, min(d, 0.85))
    est = int(base * (1 - d))
    return {"low": int(est * 0.9), "high": int(est * 1.1), "note": "rough estimate"}

def _local_summary(mk, md, yr, price, specs):
    title = f"{_s(mk)} {_s(md)} {_s(yr)}"
    price_band = f"{_fmt_money(price['low'])}–{_fmt_money(price['high'])}" if isinstance(price, dict) else "Price unknown"
    bits = [
        title,
        price_band,
        _s(specs.get("Drivetrain")),
        _s(specs.get("EngineType")),
        _s(specs.get("GasMileage")),
        _s(specs.get("BodyStyle")),
    ]
    return " • ".join(bits) + " • Prices vary by mileage, condition, and region."

def _llm_summary(mk, md, yr, price, specs) -> Dict[str, Any]:
    """
    Combined mode:
      - Use LOCAL data for specs and (optional) rough price band.
      - Ask OpenAI ONLY for an expressive, positive summary line.
    """
    title = f"{_s(mk)} {_s(md)} {_s(yr)}"
    specs_out = _specs_public(specs)   # keep your own specs
    price_out = price if isinstance(price, dict) else None  # keep your local price band if available

    # No key: friendly deterministic fallback (but still returns your local specs/price)
    if oai is None or os.getenv("OPENAI_API_KEY") is None:
        return {
            "title": title,
            "price": price_out,
            "specs": specs_out,
            "highlights": None,
            "caveats": None,
            "summary": f"{title}: a well-loved model noted for comfort and refinement.",
        }

    # Ask the model ONLY for a lively, positive summary based on name+year.
    system = (
        "You are an enthusiastic automotive presenter at a live demo. "
        "Given only a car's make, model, and year, write a SHORT, vivid, positive description (1–3 sentences) "
        "highlighting reputation, character, and why people like it. It's okay to be admiring. "
        "Do not include disclaimers like 'price varies'. Do not invent exact specs or numbers. "
        "Return strict JSON with key: summary."
    )
    user = {"make": _s(mk), "model": _s(md), "year": yr}

    try:
        resp = oai.chat.completions.create(
            model=os.getenv("LLM_MODEL", "gpt-4o-mini"),
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": json.dumps(user, ensure_ascii=False)},
            ],
            temperature=0.85,
            max_tokens=120,
            response_format={"type": "json_object"},
        )
        data = json.loads(resp.choices[0].message.content)
        summary = _s(data.get("summary"))
        if not summary:
            summary = f"{title}: praised for its overall appeal."
    except Exception:
        summary = f"{title}: praised for its overall appeal."

    return {
        "title": title,
        "price": price_out,     # from your local estimator (optional)
        "specs": specs_out,     # from your local JSON (string-safe)
        "highlights": None,
        "caveats": None,
        "summary": summary,     # from OpenAI
    }
@app.post("/dossier", response_model=DossierResp)
def dossier(req: DossierReq):
    if req.label:
        mk, md, yr = _norm_from_label(req.label)
        label_key = req.label
    else:
        if not (req.make and req.model and req.year is not None):
            raise HTTPException(400, "Provide either 'label' or all of make/model/year.")
        mk, md, yr = _s(req.make), _s(req.model), _i(req.year, 0)
        label_key = f"{mk.lower()}_{md.lower()}_{yr}"

    if label_key in _cache:
        return _cache[label_key]

    skey = _find_specs_key(mk, md, yr)
    specs = SPECS_MAP.get(skey or "", {})
    price = _estimate_price(specs.get("MSRP"), yr)

    out = _llm_summary(mk, md, yr, price, specs)
    _cache[label_key] = out
    return out
