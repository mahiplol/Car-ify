# tools/prepare_cls_dataset.py
# Build a YOLO classification dataset where class = Make_Model_Year
# from filenames like:
# Make_Model_Year_MSRP_FrontWheelIn_SAEhpRPM_Displacement_EngineType_WidthIn_HeightIn_LengthIn_GasMileage_Drivetrain_PassengerCapacity_PassengerDoors_BodyStyle_[random].jpg

import argparse, csv, json, os, random, shutil
from pathlib import Path
from collections import defaultdict

FIELDS = [
    "Make","Model","Year","MSRP","FrontWheelIn","SAEhpRPM",
    "Displacement","EngineType","WidthIn","HeightIn","LengthIn",
    "GasMileage","Drivetrain","PassengerCapacity","PassengerDoors","BodyStyle"
]
NUMERIC = {
    "MSRP","FrontWheelIn","SAEhpRPM","Displacement",
    "WidthIn","HeightIn","LengthIn","GasMileage",
    "PassengerCapacity","PassengerDoors"
}

def try_float(x):
    try:
        return float(x)
    except Exception:
        return None

def parse_filename(fname: str):
    stem = fname.rsplit(".", 1)[0]
    parts = stem.split("_")
    if len(parts) < 16:
        return None  # malformed
    tokens = parts[:16]
    rec = dict(zip(FIELDS, tokens))
    # normalize
    rec["Make"]  = rec["Make"].strip().title()
    rec["Model"] = rec["Model"].strip().upper()
    rec["Year"]  = rec["Year"].strip()
    rec["EngineType"] = rec["EngineType"].upper()
    rec["Drivetrain"] = rec["Drivetrain"].upper()
    rec["BodyStyle"]  = rec["BodyStyle"].lower()
    for k in NUMERIC:
        rec[k] = try_float(rec[k])
    rec["Triplet"] = f"{rec['Make'].lower()}_{rec['Model'].lower()}_{rec['Year']}"
    return rec

def link_or_copy(src: Path, dst: Path, do_copy: bool):
    dst.parent.mkdir(parents=True, exist_ok=True)
    if do_copy:
        shutil.copy2(src, dst); 
        return
    try:
        if os.name == "nt":
            # Hardlink usually works without admin on Windows; fallback to copy.
            try:
                os.link(src, dst)
            except Exception:
                shutil.copy2(src, dst)
        else:
            os.symlink(src, dst)
    except Exception:
        shutil.copy2(src, dst)

def simple_mode(values):
    """Return the most frequent value (mode) without using Counter."""
    if not values:
        return None
    freq = {}
    for v in values:
        freq[v] = freq.get(v, 0) + 1
    # pick the key with highest count
    best_k, best_c = None, -1
    for k, c in freq.items():
        if c > best_c:
            best_k, best_c = k, c
    return best_k

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", default="data/raw_images", help="Folder with your images")
    ap.add_argument("--out", default="data/car_cls_triplet", help="Output dataset root (train/ & val/)")
    ap.add_argument("--meta_out", default="tools/_analysis", help="Where to write metadata/specs")
    ap.add_argument("--topk", type=int, default=120, help="Keep top-K most frequent Make_Model_Year classes")
    ap.add_argument("--min_per_class", type=int, default=80, help="Drop classes with < this many images")
    ap.add_argument("--max_per_class", type=int, default=800, help="Cap per-class images (0=disable)")
    ap.add_argument("--val_ratio", type=float, default=0.2, help="Validation split")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--copy", action="store_true", help="Copy instead of link/hardlink")
    ap.add_argument("--exts", nargs="+", default=[".jpg",".jpeg",".png",".bmp",".webp"])
    args = ap.parse_args()

    raw = Path(args.raw)
    meta_dir = Path(args.meta_out); meta_dir.mkdir(parents=True, exist_ok=True)

    # 1) Parse filenames → per-image metadata rows
    rows, scanned = [], 0
    for p in raw.rglob("*"):
        if not p.is_file():
            continue
        if p.suffix.lower() not in args.exts:
            continue
        scanned += 1
        rec = parse_filename(p.name)
        if rec:
            rec["RelPath"] = str(p.as_posix())
            rows.append(rec)
    print(f"[prepare] scanned={scanned} parsed={len(rows)}")

    # 2) Write full metadata CSV (debug & provenance)
    meta_csv = meta_dir / "metadata.csv"
    with open(meta_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["RelPath","Triplet"] + FIELDS)
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print(f"[prepare] wrote {meta_csv}")

    # 3) Aggregate per class → class_specs.json (means for numeric, modes for categorical)
    buckets = defaultdict(list)
    for r in rows:
        buckets[r["Triplet"]].append(r)

    class_specs = {}
    for lab, rs in buckets.items():
        agg = {}
        # numeric means
        for k in NUMERIC:
            vals = [r[k] for r in rs if isinstance(r[k], (int, float))]
            agg[k] = (sum(vals) / len(vals)) if vals else None
        # categorical modes
        for k in ["EngineType", "Drivetrain", "BodyStyle"]:
            vals = [r[k] for r in rs if r.get(k)]
            agg[k] = simple_mode(vals)
        # identifiers
        s = rs[0]
        agg["Make"], agg["Model"], agg["Year"] = s["Make"], s["Model"], s["Year"]
        class_specs[lab] = agg

    specs_json = meta_dir / "class_specs.json"
    with open(specs_json, "w", encoding="utf-8") as f:
        json.dump(class_specs, f, indent=2)
    print(f"[prepare] wrote {specs_json}")

    # 4) Choose top-K classes by frequency (manual count, no Counter)
    freq = {}
    for r in rows:
        t = r["Triplet"]
        freq[t] = freq.get(t, 0) + 1
    freq_list = sorted(freq.items(), key=lambda kv: kv[1], reverse=True)
    keep = set([lab for lab, _ in freq_list[:args.topk]])

    # 5) Keep + balance
    random.seed(args.seed)
    per_class = defaultdict(list)
    for r in rows:
        lab = r["Triplet"]
        if lab in keep:
            per_class[lab].append(Path(r["RelPath"]))

    retained = {}
    for lab, files in per_class.items():
        if len(files) < args.min_per_class:
            continue
        files = sorted(files)
        if args.max_per_class and len(files) > args.max_per_class:
            files = random.sample(files, args.max_per_class)
        retained[lab] = files

    # 6) Write folder-per-class dataset
    out = Path(args.out)
    train_root, val_root = out / "train", out / "val"
    if train_root.exists(): shutil.rmtree(train_root)
    if val_root.exists():   shutil.rmtree(val_root)

    total = 0
    for lab, files in retained.items():
        random.shuffle(files)
        n_val = max(1, int(len(files) * args.val_ratio))
        val_files, train_files = files[:n_val], files[n_val:]
        for src in train_files:
            link_or_copy(src, train_root / lab / src.name, args.copy)
        for src in val_files:
            link_or_copy(src, val_root / lab / src.name, args.copy)
        total += len(files)

    # 7) Label map for pretty names
    label_map = {}
    for lab in retained.keys():
        make, model, year = lab.split("_", 3)
        label_map[lab] = {"make": make.title(), "model": model.upper(), "year": year}
    (out / "label_map.json").write_text(json.dumps(label_map, indent=2), encoding="utf-8")
    (out / "README.txt").write_text(
        "Ultralytics classification format: folder-per-class under train/ and val/. Class = make_model_year\n",
        encoding="utf-8"
    )

    print(f"[prepare] classes={len(retained)} images={total}")
    print(f"[prepare] dataset root: {out.resolve()}")

if __name__ == "__main__":
    main()
